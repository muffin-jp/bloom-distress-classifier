"""The charts are well-formed, and draw what the numbers say."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dc.cascade import CostModel, Thresholds, route
from dc.evaluate import Prediction, compute_results, write_plots
from dc.plots import Series, _step, line_chart_svg  # pyright: ignore[reportPrivateUsage]

SVG = "{http://www.w3.org/2000/svg}"


def chart(*series: Series) -> ET.Element:
    return ET.fromstring(
        line_chart_svg(title="t", subtitle="s", x_label="x", y_label="y", series=list(series))
    )


def test_the_svg_parses_and_names_itself() -> None:
    root = chart(Series("a", [(0.0, 0.0), (1.0, 1.0)], slot=1))
    assert root.tag == f"{SVG}svg"
    assert root.find(f"{SVG}title") is not None
    assert root.attrib["role"] == "img"


def test_the_background_is_painted() -> None:
    # An image that relies on the page behind it disappears in one of the themes.
    root = chart(Series("a", [(0.0, 0.0), (1.0, 1.0)], slot=1))
    assert any(el.attrib.get("class") == "surface" for el in root.iter(f"{SVG}rect"))


def test_dark_mode_is_its_own_set_of_steps() -> None:
    svg = line_chart_svg(
        title="t", subtitle="s", x_label="x", y_label="y",
        series=[Series("a", [(0.0, 0.0), (1.0, 1.0)], slot=1)],
    )  # fmt: skip
    assert "@media (prefers-color-scheme: dark)" in svg
    assert "#2a78d6" in svg and "#3987e5" in svg


def test_a_single_series_has_no_legend() -> None:
    root = chart(Series("only", [(0.0, 0.0), (1.0, 1.0)], slot=1))
    assert not any((el.text or "") == "only" for el in root.iter(f"{SVG}text"))


def test_multiple_series_have_a_legend() -> None:
    root = chart(Series("a", [(0.0, 1.0)], slot=1), Series("b", [(1.0, 0.0)], slot=2))
    texts = {el.text for el in root.iter(f"{SVG}text")}
    assert {"a", "b"} <= texts


def test_series_share_no_colour_slot() -> None:
    with pytest.raises(ValueError, match="own categorical slot"):
        chart(Series("a", [(0.0, 0.0)], slot=1), Series("b", [(1.0, 1.0)], slot=1))


def test_points_outside_the_unit_square_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        chart(Series("a", [(1.2, 0.5)], slot=1))


def test_a_step_holds_each_value_up_to_its_point() -> None:
    assert _step([(0.0, 1.0), (0.5, 0.8), (1.0, 0.3)]) == [
        (0.0, 1.0), (0.0, 0.8), (0.5, 0.8), (0.5, 0.3), (1.0, 0.3),
    ]  # fmt: skip


def test_a_step_keeps_drawing_order_at_a_repeated_x() -> None:
    # A PR curve can revisit a recall value as its threshold falls; sorting would
    # reorder those visits and draw a spike upward that the data does not have.
    path = _step([(0.5, 0.9), (0.5, 0.7), (1.0, 0.6)])
    ys = [y for x, y in path if x == 0.5]
    assert ys == sorted(ys, reverse=True)


def test_the_report_plots_are_written(tmp_path: Path) -> None:
    bands = Thresholds(0.1, 0.8)
    predictions = [
        Prediction(
            id=f"r{i}", category="distress" if i % 3 == 0 else "nonsense", label=int(i % 3 == 0),
            golden=i < 6, feeling="custom", text="x", score=(0.9 if i % 3 == 0 else 0.2) - i / 1000,
            keyword=float(i % 6 == 0), route=route(0.5, bands).value,
            teacher_votes=() if i < 6 else ((1, 1, 0) if i % 3 == 0 else (0, 0, 0)),
        )
        for i in range(60)
    ]  # fmt: skip
    results = compute_results(predictions, bands, costs=CostModel(20.0, 1.0), n_bootstrap=0)
    paths = write_plots(predictions, results, tmp_path)
    assert {p.name for p in paths} == {"pr_curve.svg", "reliability.svg"}
    for path in paths:
        ET.fromstring(path.read_text())
