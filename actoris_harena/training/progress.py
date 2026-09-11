"""How far a training run has got, read from the only record that exists.

Nothing in this pipeline writes metrics anywhere a program can read them.
``--wandb.enable=false`` is unconditional in ``test/system/long_vla_real.sh``,
this LeRobot ships no ``SummaryWriter``, and no jsonl or csv is produced --
``MetricsTracker.to_dict()`` returns exactly the numbers wanted and is never
called. What there is, is the driver's log, and so that is what this parses.

Two things about that log are not obvious and both are load-bearing.

**The step is abbreviated.** ``format_big_number`` renders steps 10 500 and
10 600 alike as ``10K``, so the ``step:`` field CANNOT be the x-axis of a curve;
above a thousand it is a label, not a number. ``loss``, ``grdn``, ``lr``,
``mem_gb`` and ``smp/s`` are printed at full precision, so only the axis is
affected -- but a plot drawn against ``step:`` would silently pile two thirds of
an 80 000-step run onto eight distinct x-values.

**The two destinations log differently.** tqdm is disabled inside Slurm, so a
CREATE log is clean ``INFO`` lines with no exact step anywhere, while a thanos
or local log is one enormous ``\\r``-separated blob whose progress frames carry
the exact step, the total, the elapsed time and the ETA -- strictly better
information than the line that follows them.

So the step comes from whichever source is present: the tqdm frame immediately
preceding a metric line, else the line's ordinal times ``log_freq``, which is
exact because lerobot logs at ``step % log_freq == 0`` and nowhere else.
``log_freq`` and the run's total are read from the config dump lerobot prints at
the head of its own log, which both destinations have.

Pure. No I/O, no SSH, no aiohttp: which file to read, on which machine, is
``actoris_harena.training.runs``.
"""

from __future__ import annotations

import bisect
import re
from datetime import datetime
from typing import Any

from actoris_harena.training import metrics

# ── The shapes in the log ────────────────────────────────────────────────────

# The config lerobot pprints before training. Only flat integer keys are taken:
# this is a Python repr spread over hundreds of lines, and the parts worth
# having are all `'key': 1234,` at the top level of it.
_HEAD_KEYS = ("steps", "log_freq", "save_freq", "batch_size", "num_workers")
# pprint puts the first key on the same line as the opening brace -- which is
# the INFO line itself -- and every later top-level key one space in. Nested
# keys are indented much further, so this matches the top level and no more.
_HEAD_RE = re.compile(r"(?:^ ?|\{)'(" + "|".join(_HEAD_KEYS) + r")':\s*(\d+),", re.M)

# `INFO 2026-08-24 11:51:09 ot_train.py:596 step:100 smpl:800 ... loss:10.460`
_STAMP = r"(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
_METRIC_LINE_RE = re.compile(_STAMP + r"[^\n\r]*?\bstep:[^\n\r]*")

# Every field is `name:value`, and the value may carry a magnitude suffix. The
# names are enumerated rather than matched generically because the line also
# contains `ot_train.py:596` and a `11:51:09`, both of which a general
# `word:number` pattern happily mistakes for a metric.
_FLOATS = {
    "loss": "loss",
    "grdn": "grad_norm",
    "lr": "lr",
    "updt_s": "update_s",
    "data_s": "dataloading_s",
    "smp/s": "samples_per_s",
    "mem_gb": "gpu_mem_gb",
    "epch": "epochs",
}
# Abbreviated by format_big_number, so kept only as labels and cross-checks.
_COUNTERS = {"step": "step_label", "smpl": "samples_label", "ep": "episodes_label"}
_FIELD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in [*_FLOATS, *_COUNTERS]) + r"):"
    r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?[KMBTQ]?)"
)

# `Training:  89%|███| 71379/80000 [7:56:20<39:15,  3.66step/s]`. tqdm switches
# to seconds-per-step below one step a second, and prints `?` for an ETA it
# cannot yet estimate, so both forms have to be read.
_TQDM_RE = re.compile(
    r"(?P<done>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:]+|\?),\s*"
    r"(?P<rate>[\d.]+)(?P<unit>step/s|s/step)\]"
)

# `INFO 2026-09-05 11:02:07 ot_train.py:627 step 12000: eval_loss=0.0421`.
# lerobot's held-out validation loss, written when --eval_steps and
# --dataset.eval_split are both set. Note `step 12000`, with a SPACE: this is a
# different line from the tracker's `step:12K`, and -- unlike it -- the number
# is exact, because it is an f-string of the integer rather than
# format_big_number. Nothing here has to reconstruct it.
_EVAL_LOSS_RE = re.compile(
    _STAMP + r"[^\n\r]*?\bstep (?P<step>\d+): eval_loss=(?P<loss>[\d.eE+-]+)"
)

_CHECKPOINT_RE = re.compile(r"Checkpoint policy after step (\d+)")
_DONE_RE = re.compile(r"End of training")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")
# The exception line that ends a traceback, and the driver's own verdicts.
_EXCEPTION_RE = re.compile(r"^(\w+(?:\.\w+)*(?:Error|Exception|Interrupt)\b.*)$", re.M)
_ROW_FAILED_RE = re.compile(r"❌\s*row (\d+) FAILED[^\n\r]*")
_ROW_DONE_RE = re.compile(r"✓\s*row (\d+) done")
_REUSED_RE = re.compile(r"↷\s*reusing checkpoint")
#: A run the driver picked back up after the box went down under it. The step it
#: resumed AT is worth surfacing: it is exactly what the interruption cost.
_RESUMED_RE = re.compile(r"↻\s*resuming \S+ at step (\d+) of (\d+)")

# A run is called stalled when its log has been silent for this many times its
# own observed logging interval. Long enough that a slow policy writing every
# few minutes is never libelled, short enough to notice a machine that went away
# -- thanos rebooted mid-run on 27 August 2026 and nothing said so for hours.
STALL_FACTOR = 6.0
STALL_FLOOR_S = 900.0

DEFAULT_LOG_FREQ = 100


# ── Small conversions ────────────────────────────────────────────────────────


def parse_big(text: str) -> float:
    """``'80K'`` -> ``80000.0``. The inverse of lerobot's format_big_number."""
    suffixes = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "Q": 1e15}
    text = text.strip()
    if text and text[-1] in suffixes:
        return float(text[:-1]) * suffixes[text[-1]]
    return float(text)


def parse_clock(text: str) -> "float | None":
    """``'7:56:20'`` or ``'39:15'`` -> seconds. ``'?'`` -> ``None``."""
    if not text or "?" in text:
        return None
    parts = text.split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for value in values:
        seconds = seconds * 60.0 + value
    return seconds


def parse_stamp(text: str) -> "float | None":
    """A log timestamp -> epoch seconds, read in THIS machine's timezone.

    lerobot prints local time with no offset, so a log written on a machine in
    another zone is read here as if it were local. That is why staleness is
    judged from the file's mtime against the remote clock (see
    ``actoris_harena.training.runs``) and never from these stamps: they are exact for
    intervals, which is all they are used for.
    """
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return None


# ── The pieces ───────────────────────────────────────────────────────────────


def parse_head(text: str) -> "dict[str, int]":
    """The run's own configuration, from the dump lerobot prints before training.

    First occurrence wins: the same key name appears again inside nested policy
    and dataset sections, and the top-level one is printed first.
    """
    out: "dict[str, int]" = {}
    for key, value in _HEAD_RE.findall(text):
        out.setdefault(key, int(value))
    return out


def parse_tqdm(text: str) -> "list[dict[str, Any]]":
    """Every progress frame, with its position in the text. Pure."""
    frames = []
    for match in _TQDM_RE.finditer(text):
        rate = float(match.group("rate"))
        if match.group("unit") == "s/step":
            rate = 1.0 / rate if rate else 0.0
        frames.append(
            {
                "at": match.start(),
                "done": int(match.group("done")),
                "total": int(match.group("total")),
                "elapsed_s": parse_clock(match.group("elapsed")),
                "eta_s": parse_clock(match.group("eta")),
                "rate": rate,
            }
        )
    return frames


def parse_points(text: str) -> "list[dict[str, Any]]":
    """One entry per metric line, in order, without a step. Pure.

    The step is deliberately not decided here: it depends on the tqdm frames and
    on the head, and putting it in one place (``resolve_steps``) is what keeps
    the two destinations from growing two parsers.
    """
    points = []
    for match in _METRIC_LINE_RE.finditer(text):
        line = match.group(0)
        point: "dict[str, Any]" = {
            "at": match.start(),
            "time": parse_stamp(match.group("stamp")),
        }
        for key, raw in _FIELD_RE.findall(line):
            if key in _FLOATS:
                try:
                    point[_FLOATS[key]] = float(raw)
                except ValueError:
                    continue
            else:
                point[_COUNTERS[key]] = parse_big(raw)
                if key == "step":
                    # How much the label was rounded by. `10K` stands for
                    # anything from 9500 to 10499, and the cross-check below is
                    # only meaningful if it knows that.
                    point["step_divisor"] = (
                        parse_big("1" + raw[-1]) if raw[-1].isalpha() else 1.0
                    )
        if "loss" in point:
            points.append(point)
    return points


#: Which of lerobot's tracker fields become curves, and under what name. The
#: labels (`step_label` and friends) are deliberately absent: they are rounded,
#: so they are cross-checks rather than series.
TRAIN_SERIES = {
    "loss": "train/loss",
    "grad_norm": "train/grad_norm",
    "lr": "train/lr",
    "update_s": "train/update_s",
    "dataloading_s": "train/dataloading_s",
    "samples_per_s": "train/samples_per_s",
    "gpu_mem_gb": "train/gpu_mem_gb",
    "epochs": "train/epochs",
}


def parse_eval_losses(text: str) -> "list[dict[str, Any]]":
    """lerobot's own held-out validation loss, where the run was given one.

    Off by default: ``eval_steps`` is 0 unless the driver is asked for it, and
    it in turn requires ``dataset.eval_split``, which holds out episodes and so
    changes what is trained. A run without those flags simply has no such line.
    """
    out = []
    for match in _EVAL_LOSS_RE.finditer(text):
        try:
            value = float(match.group("loss"))
        except ValueError:
            continue
        out.append(
            {
                "at": match.start(),
                "step": int(match.group("step")),
                "time": parse_stamp(match.group("stamp")),
                "values": {"eval/loss": value},
            }
        )
    return out


def build_series(
    points: "list[dict[str, Any]]", extra: "list[dict[str, Any]]"
) -> "dict[str, list[dict[str, Any]]]":
    """Every curve in one shape, so the page needs no list of metric names.

    ``points`` are lerobot's tracker lines, whose steps have already been
    resolved; ``extra`` is everything carrying its own exact step -- the
    ``so101-metric`` channel and lerobot's eval line. They are merged onto one
    step axis rather than kept apart, because a panel grid that had to know
    which parser produced a series would be a fourth place to edit when a
    policy adds a curve, which is the thing this is for.
    """
    series: "dict[str, list[dict[str, Any]]]" = {}

    def add(name: str, step: "int | None", time: "float | None", value: float) -> None:
        if step is None:
            return
        series.setdefault(name, []).append({"step": step, "time": time, "value": value})

    for point in points:
        for key, name in TRAIN_SERIES.items():
            if key in point:
                add(name, point.get("step"), point.get("time"), point[key])
    for record in extra:
        for name, value in (record.get("values") or {}).items():
            add(name, record.get("step"), record.get("time"), value)
    for entries in series.values():
        entries.sort(key=lambda e: e["step"])
    return series


def parse_resumes(text: str) -> "list[dict[str, Any]]":
    """Where the driver picked a run back up, and at which step. Pure.

    A resumed run APPENDS to the log it already had, so one file can hold two or
    more attempts end to end. These markers are the only thing that says where
    one stops and the next begins -- and therefore the only reliable way to give
    the second attempt's lines their true step (see :func:`resolve_steps`).
    """
    return [
        {"at": m.start(), "step": int(m.group(1)), "total": int(m.group(2))}
        for m in _RESUMED_RE.finditer(text)
    ]


def resolve_steps(
    points: "list[dict[str, Any]]",
    frames: "list[dict[str, Any]]",
    log_freq: int = DEFAULT_LOG_FREQ,
    total_steps: "int | None" = None,
    resumes: "list[dict[str, Any]] | None" = None,
) -> "list[str]":
    """Give every point its exact step, in place. Returns any disagreements.

    Two sources, in order of preference:

    * the tqdm frame immediately BEFORE the line -- exact, and present on any
      machine that is not a Slurm node;
    * the line's ordinal times ``log_freq`` -- exact too, because lerobot logs
      when ``step % log_freq == 0`` and at no other time, which is the only
      source a CREATE log has.

    A RESUMED run breaks both, and does so invisibly: lerobot builds its bar
    with ``total=cfg.steps - step``, so the second attempt's frames count from
    zero again, and its metric lines restart their ordinal too. The log holds
    both attempts one after the other, so the offset a line needs is decided by
    WHERE it is: the step named by the last ``resuming`` marker before it, and
    zero for a line before any. Inferring the offset from the totals instead --
    which is what this did first -- moves the FIRST attempt's points as well,
    because both attempts happen to carry the same total when the target has not
    changed.

    Falls back to the total-difference heuristic when a log was resumed without
    the driver's marker, which is what a resume done by hand looks like.

    The abbreviated ``step:`` field is used for neither; it is only checked, and
    a mismatch is returned rather than hidden, because a curve drawn against a
    wrong axis looks exactly like a curve.
    """
    positions = [f["at"] for f in frames]
    marks = sorted(resumes or [], key=lambda r: r["at"])
    mark_positions = [r["at"] for r in marks]
    problems: "list[str]" = []
    # The ordinal a `count`-sourced step is derived from restarts at each
    # resume, so it is counted per segment rather than over the whole file.
    ordinal = 0
    segment = 0
    for index, point in enumerate(points):
        here = bisect.bisect_right(mark_positions, point["at"])
        if here != segment:
            segment, ordinal = here, 0
        ordinal += 1
        base = marks[here - 1]["step"] if here else 0

        frame = None
        slot = bisect.bisect_left(positions, point["at"])
        if slot:
            frame = frames[slot - 1]
        if frame is not None:
            offset = base
            if (
                not marks
                and total_steps
                and frame["total"]
                and total_steps > frame["total"]
            ):
                offset = total_steps - frame["total"]
            point["step"] = offset + frame["done"]
            point["source"] = "tqdm"
        else:
            point["step"] = base + ordinal * log_freq
            point["source"] = "count"

        label = point.get("step_label")
        if label is not None and problems == []:
            # format_big_number divides by a thousand per suffix and rounds to
            # a whole number, so `10K` stands for anything in [9500, 10500).
            # Half the divisor is therefore the whole tolerance -- a looser one
            # (the label's digit count, which was the first attempt) accepts a
            # curve drawn 9 600 steps away from where it happened.
            tolerance = max(point.get("step_divisor", 1.0) / 2.0, 1.0)
            if abs(point["step"] - label) > tolerance:
                problems.append(
                    f"step {point['step']} disagrees with the log's own "
                    f"'step:{int(label)}' at line {index + 1}"
                )
    return problems


def parse_markers(text: str) -> "dict[str, Any]":
    """Everything in the log that is not a metric: checkpoints, restarts, endings."""
    failure = None
    if _TRACEBACK_RE.search(text):
        exceptions = _EXCEPTION_RE.findall(text)
        failure = exceptions[-1].strip() if exceptions else "training raised"
    row_failed = _ROW_FAILED_RE.search(text)
    if row_failed and not failure:
        failure = row_failed.group(0).strip()
    resumed = _RESUMED_RE.findall(text)
    return {
        "checkpoints": [int(s) for s in _CHECKPOINT_RE.findall(text)],
        "finished": bool(_DONE_RE.search(text)),
        "failure": failure,
        "rows_done": [int(s) for s in _ROW_DONE_RE.findall(text)],
        "reused": bool(_REUSED_RE.search(text)),
        # One entry per interruption, so a run that was cut short twice says so
        # twice rather than reading as one clean run.
        "resumed_at": [int(m[0]) for m in resumed],
        # The step target the RESUME was given. This is not decoration: a
        # resumed run appends to the log it already had, so the config dump at
        # the head still states the total the FIRST attempt was given, and a
        # target that has since been raised would be read from a stale number.
        "resumed_total": int(resumed[-1][1]) if resumed else None,
    }


# ── The whole thing ──────────────────────────────────────────────────────────


def parse_log(
    text: str,
    log_freq: "int | None" = None,
    head_text: "str | None" = None,
) -> "dict[str, Any]":
    """A log's text -> its configuration, its curve and how it ended. Pure.

    ``head_text`` is the configuration dump when it was fetched SEPARATELY from
    the metric lines, which is how ``actoris_harena.training.runs`` reads a remote log.
    Passing the two as one string is what a local read does, and is also how a
    first attempt at the remote read got 887 points out of an 800-point run: the
    head prefix already contained the first 87 metric lines, and they were then
    counted again from the filtered body. Whichever way the caller has it, the
    ordinal a step is derived from must count each line exactly once.
    """
    # tqdm separates its frames with \r, so a log from a machine that had one
    # enabled is a single "line" megabytes long. Everything downstream wants
    # lines, and nothing wants the overwritten frames back.
    text = text.replace("\r", "\n")
    head = parse_head(head_text if head_text is not None else text)
    frames = parse_tqdm(text)
    points = parse_points(text)
    markers = parse_markers(text)
    # A resume states the target it was actually given, and it is the later
    # word: see parse_markers. Without this a run resumed with a raised target
    # has every point of its second half plotted at the wrong step, because the
    # tqdm frames of a resumed run count what is LEFT rather than what is done.
    total = (
        markers.get("resumed_total")
        or head.get("steps")
        or (frames[-1]["total"] if frames else None)
    )
    freq = log_freq or head.get("log_freq") or DEFAULT_LOG_FREQ
    problems = resolve_steps(points, frames, freq, total, parse_resumes(text))
    if total and points and points[-1]["step"] > total:
        # Counted past the end of the run. The curve is then drawn against an
        # axis that does not exist, so say so rather than plot it.
        problems.append(
            f"counted {len(points)} logged steps, which is past the run's "
            f"{total}: are some lines in the text twice?"
        )
    # Everything that carries its own exact step, merged onto the same axis.
    # A `so101-metric` line has no timestamp of its own -- it is a bare print,
    # not one of lerobot's INFO lines -- so it borrows the time of the nearest
    # preceding tracker line, which is at most `log_freq` steps away. That is
    # what makes it plottable against wall-clock; it is a stamp, not a
    # measurement, and nothing derives a rate from it.
    extra = metrics.parse(text) + parse_eval_losses(text)
    stamps = [(p["at"], p.get("time")) for p in points if p.get("time") is not None]
    offsets = [at for at, _ in stamps]
    for record in extra:
        if record.get("time") is None and offsets:
            index = bisect.bisect_right(offsets, record["at"]) - 1
            record["time"] = stamps[max(index, 0)][1]
    series = build_series(points, extra)
    for point in points:
        point.pop("at", None)
    return {
        "config": head,
        "log_freq": freq,
        "total_steps": total,
        "points": points,
        "series": series,
        "metrics": sorted(series),
        "last_frame": frames[-1] if frames else None,
        "problems": problems,
        **markers,
    }


def summarise(
    parsed: "dict[str, Any]", age_s: "float | None" = None
) -> "dict[str, Any]":
    """The one-line state of a run: where it is, how fast, and whether it lives.

    ``age_s`` is how long ago the log was last written, measured on the machine
    that holds it -- never from the timestamps inside, which are in that
    machine's timezone and cannot be compared with this one's clock.
    """
    points = parsed.get("points") or []
    total = parsed.get("total_steps")
    frame = parsed.get("last_frame")
    losses = [p["loss"] for p in points if "loss" in p]
    step = points[-1]["step"] if points else 0
    if frame is not None:
        # The bar is written every step and the metric line every hundredth, so
        # between two log lines the bar is the more current of the two.
        step = max(step, frame["done"])

    interval = _median_interval(points)
    rate = frame["rate"] if frame else _rate_from(points)
    eta = None
    if frame is not None and frame["eta_s"] is not None:
        eta = frame["eta_s"]
    elif rate and total and step < total:
        eta = (total - step) / rate

    out = {
        "step": step,
        "total_steps": total,
        "fraction": (step / total) if total else None,
        "eta_s": eta,
        "rate": rate,
        "loss_last": losses[-1] if losses else None,
        "loss_first": losses[0] if losses else None,
        "loss_min": min(losses) if losses else None,
        "gpu_mem_gb_max": _peak(points, "gpu_mem_gb"),
        "samples_per_s": points[-1].get("samples_per_s") if points else None,
        "epochs": points[-1].get("epochs") if points else None,
        "checkpoint": max(parsed.get("checkpoints") or [0]) or None,
        "points": len(points),
        # What this run actually has curves for. The page builds its panel grid
        # from this rather than from a list of its own, so a policy that emits
        # a new metric gets a panel without the front end being taught its name.
        "metrics": parsed.get("metrics") or [],
        "eval_loss_last": ((parsed.get("series") or {}).get("eval/loss") or [{}])[
            -1
        ].get("value"),
        "age_s": age_s,
        # How often the run was picked back up, and what the last interruption
        # cost. A curve with a restart in it is a different thing from a clean
        # one, and the panel should not present them as the same.
        "resumes": len(parsed.get("resumed_at") or []),
        "resumed_at": (parsed.get("resumed_at") or [None])[-1],
    }
    out["state"] = _state(parsed, interval, age_s)
    return out


def _state(
    parsed: "dict[str, Any]", interval: "float | None", age_s: "float | None"
) -> str:
    if parsed.get("failure"):
        return "failed"
    if parsed.get("finished"):
        return "done"
    if age_s is None:
        return "running"
    # A log that has stopped growing has not necessarily stopped training -- a
    # policy that logs every few minutes is silent between lines -- so the
    # threshold follows the run's OWN cadence, with a floor under it.
    limit = max(STALL_FLOOR_S, STALL_FACTOR * (interval or 0.0))
    return "stalled" if age_s > limit else "running"


def _median_interval(points: "list[dict[str, Any]]") -> "float | None":
    times = [p["time"] for p in points if p.get("time") is not None]
    gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    return gaps[len(gaps) // 2] if gaps else None


def _rate_from(points: "list[dict[str, Any]]") -> "float | None":
    """Steps per second over the whole run, when there is no progress bar."""
    timed = [p for p in points if p.get("time") is not None]
    if len(timed) < 2:
        return None
    span = timed[-1]["time"] - timed[0]["time"]
    steps = timed[-1]["step"] - timed[0]["step"]
    return (steps / span) if span > 0 and steps > 0 else None


def _peak(points: "list[dict[str, Any]]", key: str) -> "float | None":
    values = [p[key] for p in points if key in p]
    return max(values) if values else None


def thin(points: "list[dict[str, Any]]", limit: int = 1200) -> "list[dict[str, Any]]":
    """At most ``limit`` points, keeping the first and last. Pure.

    A 100 000-step diffusion run logs a thousand points, which a canvas a few
    hundred pixels wide cannot show anyway; this only matters when several runs
    are overlaid.
    """
    if limit <= 2 or len(points) <= limit:
        return list(points)
    stride = len(points) / float(limit - 1)
    kept = [points[int(i * stride)] for i in range(limit - 1)]
    kept.append(points[-1])
    return kept
