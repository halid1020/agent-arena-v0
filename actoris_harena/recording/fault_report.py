"""Turning a session's device failures into one honest explanation.

Each device reports its own failure and knows nothing about the others, so a hub
going down surfaces as several unrelated complaints: two cameras that cannot be
reopened, a serial bus that will not write, a storage volume returning I/O
errors. Reported that way it reads as several broken cables, and an operator
acting on it reseats three connectors that were never loose. Reported together it
is one hub, and the remedy is a different socket or a power supply.

The decision is only about what to blame. Several devices failing on one hub is a
hub or power fault, and the message should name the hub and everything behind it
-- including the devices that did not happen to fail, because they are the ones
that will next time. A single device failing, however often, really is that
device, and the cable-and-port advice is right for it.

Pure and unit-tested: the caller supplies what failed and where each device
sits, and gets back the lines to print.
"""

from __future__ import annotations

from typing import Mapping

from actoris_harena.recording.usb_topology import UsbLocation, group_by_hub


def _times(n: int) -> str:
    return "once" if n == 1 else f"{n} times"


def session_fault_report(
    faults: "Mapping[str, int]",
    locations: "Mapping[str, UsbLocation | None]",
) -> "list[str]":
    """Lines explaining this session's device failures, most-likely cause first.

    ``faults`` maps a device label to how many times it failed; entries with a
    zero count are ignored so a caller can pass its whole device list.
    ``locations`` places every KNOWN device, failed or not, so the report can
    name the others behind an implicated hub.
    """
    failed = {label: n for label, n in faults.items() if n > 0}
    if not failed:
        return []
    hubs = group_by_hub(dict(locations))
    lines: list[str] = []
    blamed: set[str] = set()

    for hub, members in sorted(hubs.items()):
        hit = sorted(label for label in members if label in failed)
        if len(hit) < 2:
            continue
        blamed.update(hit)
        others = sorted(set(members) - set(hit))
        lines.append(
            f"⚠️  {len(hit)} devices on one USB hub ({hub}) failed this session: "
            + ", ".join(hit)
        )
        if others:
            lines.append(
                "    also on that hub, and at the same risk: " + ", ".join(others)
            )
        lines.append(
            "    That is the hub or its power, not the individual cables — give "
            "it its own powered supply, or move some devices to a port on "
            "another controller."
        )

    for label, n in sorted(failed.items()):
        if label in blamed:
            continue
        where = locations.get(label)
        seat = f" (at {where.describe()})" if where is not None else ""
        urgency = (
            "worth a look before the next session"
            if n == 1
            else "check this before collecting more"
        )
        lines.append(
            f"⚠️  {label} dropped off the bus {_times(n)} this session{seat} — "
            f"{urgency}: reseat its connector, try a port on another "
            "controller, or power its hub."
        )
    return lines
