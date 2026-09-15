"""Two static SVG charts for the test report: precision-recall and reliability.

Plain SVG strings, no plotting library — the charts are small, deterministic, and
committed next to the numbers they draw. Rendered by GitHub as images, so there
is no hover layer; the report's tables carry every value these charts show.

Colour follows the reference data-viz palette, validated with its six checks for
all pairs of the three series in both modes (worst CVD ΔE 9.2 light / 9.4 dark).
Light-mode aqua sits at 2.74:1 against the surface, which the relief rule covers:
the values are in a table beside each chart. Identity is carried by the legend,
its line keys, and markers — direct labels are drawn only where they fit, and
precision-recall curves, which crowd along the top edge, do not leave room.
Dark mode is its own set of steps, selected by ``prefers-color-scheme``, not an
automatic inversion. The background is painted, so the chart never borrows the
page behind it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

__all__ = ["Series", "line_chart_svg"]

WIDTH, HEIGHT = 600, 440
LEFT, RIGHT, TOP, BOTTOM = 60, 28, 92, 56
MIN_LABEL_GAP = 15

_STYLE = """
text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.surface { fill: #fcfcfb; }
.grid { stroke: #e1e0d9; stroke-width: 1; }
.axis { stroke: #c3c2b7; stroke-width: 1; }
.reference { stroke: #898781; stroke-width: 1; }
.ink { fill: #0b0b0b; }
.ink2 { fill: #52514e; }
.muted { fill: #898781; }
.tick { font-variant-numeric: tabular-nums; }
.halo { paint-order: stroke; stroke: #fcfcfb; stroke-width: 4px; stroke-linejoin: round; }
.ring { stroke: #fcfcfb; stroke-width: 2; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.s1 { stroke: #2a78d6; } .f1 { fill: #2a78d6; }
.s2 { stroke: #eb6834; } .f2 { fill: #eb6834; }
.s3 { stroke: #1baf7a; } .f3 { fill: #1baf7a; }
@media (prefers-color-scheme: dark) {
  .surface { fill: #1a1a19; }
  .grid { stroke: #2c2c2a; }
  .axis { stroke: #383835; }
  .ink { fill: #ffffff; }
  .ink2 { fill: #c3c2b7; }
  .halo { stroke: #1a1a19; }
  .ring { stroke: #1a1a19; }
  .s1 { stroke: #3987e5; } .f1 { fill: #3987e5; }
  .s2 { stroke: #d95926; } .f2 { fill: #d95926; }
  .s3 { stroke: #199e70; } .f3 { fill: #199e70; }
}
"""


@dataclass(frozen=True)
class Series:
    name: str
    #: (x, y) in data units, both within [0, 1], in drawing order.
    points: Sequence[tuple[float, float]]
    #: Categorical slot, 1-3, in fixed order. Never cycled.
    slot: int
    markers: bool = False
    #: Index into ``points`` where the direct label anchors.
    label_at: int | None = None
    #: Draw as a step: each y holds from the previous x up to this point's x.
    #: Precision-recall curves need this — average precision is the area under
    #: exactly that step, and straight lines between points overstate it.
    step: bool = False


def _px(value: float) -> float:
    return LEFT + value * (WIDTH - LEFT - RIGHT)


def _py(value: float) -> float:
    return TOP + (1.0 - value) * (HEIGHT - TOP - BOTTOM)


def _fmt(value: float) -> str:
    return f"{value:.1f}"


def _step(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Vertical, then horizontal: the interval up to each x takes that point's y.

    Points are drawn in the order given, not re-sorted. A precision-recall curve
    can visit the same recall more than once as its threshold falls; sorting by
    (recall, precision) reorders those visits and draws spikes that are not there.
    """
    out: list[tuple[float, float]] = list(points[:1])
    for x, y in points[1:]:
        out.append((out[-1][0], y))
        out.append((x, y))
    return out


def line_chart_svg(
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    series: Sequence[Series],
    diagonal_label: str | None = None,
) -> str:
    """A unit-square line chart: hairline grid, 2px lines, ringed markers, direct labels."""
    if not 1 <= len(series) <= 3:
        raise ValueError("these charts carry one to three series")
    if len({s.slot for s in series}) != len(series) or any(not 1 <= s.slot <= 3 for s in series):
        raise ValueError("each series needs its own categorical slot, 1-3")
    for s in series:
        if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in s.points):
            raise ValueError(f"series {s.name!r} has a point outside the unit square")

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(subtitle)}</desc>',
        f"<style>{_STYLE}</style>",
        f'<rect class="surface" width="{WIDTH}" height="{HEIGHT}" rx="8"/>',
        f'<text class="ink" x="{LEFT}" y="30" font-size="15" font-weight="600">'
        f"{escape(title)}</text>",
        f'<text class="ink2" x="{LEFT}" y="50" font-size="12">{escape(subtitle)}</text>',
    ]

    # Legend: always present for two or more series. A single series needs none —
    # the title names it.
    if len(series) > 1:
        x = float(LEFT)
        for s in series:
            parts.append(
                f'<line class="line s{s.slot}" x1="{_fmt(x)}" y1="70" x2="{_fmt(x + 18)}" y2="70"/>'
            )
            parts.append(
                f'<text class="ink2" x="{_fmt(x + 24)}" y="74" font-size="12">'
                f"{escape(s.name)}</text>"
            )
            x += 34 + 7.0 * len(s.name)

    # Grid and ticks: hairline, solid, recessive.
    for step in range(6):
        value = step / 5
        gx, gy = _px(value), _py(value)
        parts.append(
            f'<line class="grid" x1="{_fmt(gx)}" y1="{TOP}" x2="{_fmt(gx)}" '
            f'y2="{HEIGHT - BOTTOM}"/>'
        )
        parts.append(
            f'<line class="grid" x1="{LEFT}" y1="{_fmt(gy)}" x2="{WIDTH - RIGHT}" y2="{_fmt(gy)}"/>'
        )
        parts.append(
            f'<text class="muted tick" x="{_fmt(gx)}" y="{HEIGHT - BOTTOM + 18}" font-size="11" '
            f'text-anchor="middle">{value:.1f}</text>'
        )
        parts.append(
            f'<text class="muted tick" x="{LEFT - 10}" y="{_fmt(gy + 4)}" font-size="11" '
            f'text-anchor="end">{value:.1f}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{LEFT}" y1="{HEIGHT - BOTTOM}" x2="{WIDTH - RIGHT}" '
        f'y2="{HEIGHT - BOTTOM}"/>'
    )
    parts.append(
        f'<text class="ink2" x="{_fmt((LEFT + WIDTH - RIGHT) / 2)}" y="{HEIGHT - 16}" '
        f'font-size="12" text-anchor="middle">{escape(x_label)}</text>'
    )
    mid_y = _fmt((TOP + HEIGHT - BOTTOM) / 2)
    parts.append(
        f'<text class="ink2" x="18" y="{mid_y}" font-size="12" text-anchor="middle" '
        f'transform="rotate(-90 18 {mid_y})">{escape(y_label)}</text>'
    )

    if diagonal_label is not None:
        parts.append(
            f'<line class="reference" x1="{_fmt(_px(0))}" y1="{_fmt(_py(0))}" '
            f'x2="{_fmt(_px(1))}" y2="{_fmt(_py(1))}"/>'
        )
        parts.append(
            f'<text class="muted halo" x="{_fmt(_px(0.97))}" y="{_fmt(_py(0.97) + 16)}" '
            f'font-size="11" text-anchor="end">{escape(diagonal_label)}</text>'
        )

    for s in series:
        path = _step(s.points) if s.step else list(s.points)
        coords = " ".join(f"{_fmt(_px(x))},{_fmt(_py(y))}" for x, y in path)
        parts.append(f'<polyline class="line s{s.slot}" points="{coords}"/>')
        if s.markers:
            for x, y in s.points:
                parts.append(
                    f'<circle class="f{s.slot} ring" cx="{_fmt(_px(x))}" cy="{_fmt(_py(y))}" '
                    f'r="4"><title>{escape(s.name)}: ({x:.2f}, {y:.2f})</title></circle>'
                )

    # Direct labels in ink, never in the series colour; spread so none collide.
    labels: list[tuple[float, float, Series]] = []
    for s in series:
        if s.label_at is None or not s.points:
            continue
        x, y = s.points[min(max(s.label_at, 0), len(s.points) - 1)]
        labels.append((_px(x), _py(y), s))
    labels.sort(key=lambda item: item[1])
    placed: list[float] = []
    for lx, ly, s in labels:
        # Below the anchor: curves that hug the top edge leave the space under
        # them empty, and a label above would sit on the line.
        target = ly + 20
        if placed and target < placed[-1] + MIN_LABEL_GAP:
            target = placed[-1] + MIN_LABEL_GAP
        target = min(max(target, TOP + 12), HEIGHT - BOTTOM - 6)
        placed.append(target)
        anchor = "end" if lx > WIDTH - RIGHT - 120 else "start"
        tx = lx - 10 if anchor == "end" else lx + 10
        parts.append(
            f'<text class="ink2 halo" x="{_fmt(tx)}" y="{_fmt(target)}" font-size="12" '
            f'text-anchor="{anchor}">{escape(s.name)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"
