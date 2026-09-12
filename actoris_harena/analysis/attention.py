"""What ACT's decoder looked at, per action of the chunk.

ACT is a transformer whose decoder cross-attends from one query per planned
action to the encoder's tokens -- and :mod:`actoris_harena.analysis.streams` knows which
token belongs to which camera. So the attention weights give something no other
method here can: a *per action step* reading of which stream the policy consulted
when planning action 0, action 50, action 99. That is the shape of a plan's
reliance over its own horizon, for the price of one forward pass.

Getting it costs nothing extra. ``nn.MultiheadAttention`` computes the averaged
weights whether or not they are wanted, and ACT's decoder discards them with a
``[0]``; a plain forward hook picks them back up.

**Attention is not attribution**, and this module will not let itself be read as
if it were. High attention on a token does not establish that the token changed
the output -- the value it carries may be near zero, and a residual path can
route around it entirely. It is reported BESIDE integrated gradients and the
occlusion effect, and where the three disagree that disagreement is the finding.
Only occlusion actually intervenes on the policy; the other two are inferences
about it.

MEASURED on the five-camera ACT checkpoint over 206 frames of six episodes, and
the reason :func:`deviation` exists: this decoder's cross-attention has almost no
DYNAMIC RANGE. Each camera owns 300 of 1 502 tokens, so uniform would give it
0.1997 of the mass -- and the pooled shares run 0.192 to 0.219, a spread of
1.14x across the five, where occlusion over the same frames spans 1.4 % to
57.1 %, a spread of 42x. Pooled, the two rank the cameras alike (+0.90); PER
FRAME they often do not, agreeing at a mean of only +0.17 with 35 % of frames
correlating negatively. So the raw mass is dominated by TOKEN COUNT and is far
too flat to read as importance, however well it happens to sort.

What carries information is the DEVIATION from uniform, and how it varies across
the chunk's own horizon -- `central`'s share runs about 0.31 for the first
actions of a plan and 0.08 for the last, which no token count explains.

Diffusion has no equivalent: it conditions a UNet on a concatenated feature
vector with no attention over cameras at all, which is one of the concrete ways
the two policies must be analysed differently.
"""

from __future__ import annotations

import numpy as np


def decoder_layers(policy) -> list:
    """ACT's decoder layers, or an empty list for a policy that has none."""
    decoder = getattr(getattr(policy, "model", None), "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        return []
    return [layer for layer in layers if hasattr(layer, "multihead_attn")]


def cross_attention(inference, batch: dict) -> "np.ndarray | None":
    """Cross-attention as ``(layers, actions, tokens)``, or None if there is none.

    Averaged over heads, which is what ``nn.MultiheadAttention`` returns by
    default. Layers are kept separate rather than summed: ACT's decoder has one
    layer by default here, but where there are several they do different jobs and
    averaging them is a choice the caller should make knowingly.
    """
    layers = decoder_layers(inference.policy)
    if not layers:
        return None
    torch = inference.torch
    collected: "list[np.ndarray]" = []

    def keep(_module, _inputs, output):
        # (attn_output, attn_weights); the weights are (B, queries, keys).
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            collected.append(output[1][0].detach().to("cpu").numpy())

    handles = [layer.multihead_attn.register_forward_hook(keep) for layer in layers]
    try:
        with torch.no_grad():
            inference.chunk_tensor(dict(batch))
    finally:
        for handle in handles:
            handle.remove()
    return np.stack(collected) if collected else None


def by_stream(weights: np.ndarray, spans) -> "dict[str, np.ndarray]":
    """Fold token-level attention onto streams: ``{stream: (actions,)}``.

    SUMMED over the tokens of a stream, not averaged. Attention over the token
    axis sums to one, so a sum keeps that property -- the streams' shares of one
    action's attention add up to one, minus whatever went to the VAE latent --
    while a mean would make a 300-token camera look 300 times less consulted
    than the single state token for the same total attention.
    """
    if weights.ndim == 3:
        weights = weights.mean(axis=0)  # over layers
    out: "dict[str, np.ndarray]" = {}
    for span in spans:
        columns = [c for c in span.indices() if c < weights.shape[1]]
        out[span.stream.name] = (
            weights[:, columns].sum(axis=1) if columns else np.zeros(weights.shape[0])
        )
    return out


def summarise(per_action: "dict[str, np.ndarray]") -> "dict[str, float]":
    """One number per stream: its mean attention across the whole chunk."""
    return {name: float(np.mean(values)) for name, values in per_action.items()}


def spatial(
    weights: np.ndarray, spans, camera: str, hw: "tuple[int, int]"
) -> "np.ndarray | None":
    """One camera's attention laid back out as its feature map, summed over actions.

    The tokens of a camera are its ResNet feature map flattened row-major
    (``einops`` ``"b c h w -> (h w) b c"``), so they fold back to ``hw`` in the
    same order -- which is what makes an attention map comparable, cell for cell,
    with a Grad-CAM map of the same frame.
    """
    if weights.ndim == 3:
        weights = weights.mean(axis=0)
    for span in spans:
        if span.stream.name != camera:
            continue
        columns = [c for c in span.indices() if c < weights.shape[1]]
        rows, cols = hw
        if len(columns) != rows * cols:
            return None
        return weights[:, columns].sum(axis=0).reshape(rows, cols)
    return None


def uniform_share(spans, total_tokens: int) -> "dict[str, float]":
    """What each stream's attention would be if every token got the same weight.

    The null this is read against. A camera owns 300 of 1 502 tokens on the
    five-camera ACT checkpoint, so uniform hands it 0.1997 whether the policy
    consults it or not.
    """
    return {
        span.stream.name: (span.width / total_tokens if total_tokens else 0.0)
        for span in spans
    }


def deviation(
    per_stream: "dict[str, float]", spans, total_tokens: int
) -> "dict[str, float]":
    """Attention mass MINUS what token count alone would give it.

    Positive means consulted more than its size accounts for. This -- not the
    raw mass -- is the number worth plotting: see the module docstring for the
    measurement that makes the raw mass unreadable.
    """
    null = uniform_share(spans, total_tokens)
    return {name: value - null.get(name, 0.0) for name, value in per_stream.items()}
