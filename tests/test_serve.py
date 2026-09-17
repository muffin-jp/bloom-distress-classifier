"""The rule as served: which statistic decides which band, and how notes are cut up.

The red team defeated the whole-note rule by diluting a crisis clause with game
talk until the mean-pooled score fell below ``low``. These tests pin the fix and,
just as importantly, pin what the fix must *not* do: a bad-looking fragment must
never by itself route a note to support.
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from dc.cascade import Route, Thresholds, route_note, skip_statistic
from dc.features import EMBED_MODEL, EMBED_MODEL_REVISION
from dc.model import DistressModel, FeatureLayout
from dc.serve import flatten_segments, score_notes, segment_scores
from dc.text import SEGMENTATION, ScopeRule, Segmentation, split_segments

BANDS = Thresholds(low=0.1, high=0.8)


class WordEmbedder:
    """A stub whose score rises with the share of alarming words in a text.

    That is exactly the property MiniLM's mean pooling has and the reason dilution
    works, so a stub with it can exercise the routing rule without any weights.
    """

    dim = 2
    ALARMING = {"ending", "worthless", "unsafe"}

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), 2), dtype=np.float32)
        for row, text in enumerate(texts):
            words = text.split()
            share = sum(word in self.ALARMING for word in words) / max(len(words), 1)
            out[row] = (share, 1.0)
        return out


def model(weight: float = 12.0, intercept: float = -6.0) -> DistressModel:
    layout = FeatureLayout(EMBED_MODEL, EMBED_MODEL_REVISION, 2, False)
    return DistressModel(coef=np.array([weight, 0.0]), intercept=intercept, layout=layout)


# --- route_note: which statistic decides which band ------------------------------------


def test_a_note_skips_only_when_every_segment_is_below_low() -> None:
    assert route_note(0.01, 0.05, BANDS) is Route.SKIP_LLM


def test_a_calm_note_hiding_an_alarming_clause_escalates() -> None:
    """The red-team finding, as a rule: the whole note is below `low` and it still escalates."""
    assert route_note(0.0078, 0.479, BANDS) is Route.ESCALATE


def test_support_is_decided_by_the_whole_note_not_by_its_worst_fragment() -> None:
    """A quoted or hypothetical clause must not by itself trigger the support message.

    `high` was fitted on whole-note scores, and routing on fragments would add a
    class of false alarm that neither the cost model nor the gate has measured.
    """
    assert route_note(0.3, 0.99, BANDS) is Route.ESCALATE


def test_a_note_above_high_goes_to_support() -> None:
    assert route_note(0.95, 0.99, BANDS) is Route.SUPPORT


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.5, "0.5", None, True])
def test_an_unusable_whole_note_score_escalates(bad: object) -> None:
    assert route_note(bad, 0.01, BANDS) is Route.ESCALATE


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.5, "0.5", None, True])
def test_an_unusable_segment_score_escalates_rather_than_skipping(bad: object) -> None:
    """Failing to score the segments is failing to prove the note is safe to skip."""
    assert route_note(0.01, bad, BANDS) is Route.ESCALATE


def test_scores_exactly_on_a_threshold_escalate() -> None:
    assert route_note(BANDS.low, BANDS.low, BANDS) is Route.ESCALATE
    assert route_note(BANDS.high, BANDS.high, BANDS) is Route.ESCALATE


def test_the_skip_band_can_only_narrow_never_widen() -> None:
    """Whatever the segments say, a note the whole-note rule escalated is never skipped."""
    rng = np.random.default_rng(0)
    whole = rng.uniform(0, 1, 500)
    worst = np.maximum(whole, rng.uniform(0, 1, 500))
    for w, s in zip(whole, worst, strict=True):
        if route_note(float(w), float(s), BANDS) is Route.SKIP_LLM:
            assert float(w) < BANDS.low


# --- skip_statistic: the wiring guard --------------------------------------------------


def test_swapping_the_two_score_arrays_is_refused() -> None:
    """A segment maximum includes the whole note, so it can never be lower.

    Passing them the wrong way round would widen the skip band and produce
    entirely plausible numbers, so it is caught rather than trusted.
    """
    whole = np.array([0.2, 0.4, 0.6])
    worst = np.array([0.5, 0.9, 0.7])
    assert np.array_equal(skip_statistic(whole, worst), worst)
    with pytest.raises(ValueError, match="swapped"):
        skip_statistic(worst, whole)


def test_no_skip_scores_means_the_whole_note_score() -> None:
    whole = np.array([0.2, 0.4])
    assert np.array_equal(skip_statistic(whole, None), whole)


# --- the splitter ----------------------------------------------------------------------


def test_the_segmentation_rejects_parameters_it_cannot_apply() -> None:
    with pytest.raises(ValueError, match="window and stride"):
        Segmentation(window=0)
    with pytest.raises(ValueError, match="regular expression"):
        Segmentation(boundary="([unclosed")


def test_split_segments_uses_the_fitted_parameters() -> None:
    text = "one two three. four five six"
    assert split_segments(text) == SEGMENTATION.split(text)


def test_flatten_segments_records_which_note_each_segment_came_from() -> None:
    texts, owners = flatten_segments(["first note here", "second. note here"])
    assert len(texts) == len(owners)
    assert owners[0] == 0
    assert set(owners.tolist()) == {0, 1}
    assert texts[int(np.flatnonzero(owners == 1)[0])] == "second. note here"


def test_a_note_with_no_text_contributes_no_segments() -> None:
    texts, owners = flatten_segments(["", "   ", "real note"])
    assert set(owners.tolist()) == {2}
    assert texts == ["real note"]


# --- score_notes -----------------------------------------------------------------------


def test_whole_note_is_scored_first_and_worst_is_the_maximum() -> None:
    embedder = WordEmbedder()
    note = "cleared the level finally. thinking about ending it"
    [scored] = score_notes([note], model(), embedder)
    assert scored.worst >= scored.whole
    assert "ending" in scored.worst_segment


def test_dilution_lowers_the_whole_note_score_but_not_the_worst_segment() -> None:
    """The attack and its defeat, in one comparison."""
    embedder = WordEmbedder()
    short, diluted = score_notes(
        [
            "thinking about ending it",
            "cleared the level finally tries and tries and tries and tries "
            "and tries and tries. thinking about ending it",
        ],
        model(),
        embedder,
    )
    assert diluted.whole < short.whole  # game talk averages the clause away
    assert diluted.worst == pytest.approx(short.whole, abs=1e-6)  # the clause survives as a segment


def test_every_note_costs_one_embedding_call_however_many_segments() -> None:
    embedder = WordEmbedder()
    score_notes(["one. two. three", "four. five", "six"], model(), embedder)
    assert embedder.calls == 1


def test_a_note_with_no_scorable_text_escalates() -> None:
    [scored] = score_notes(["   "], model(), WordEmbedder())
    assert not scored.scorable
    assert scored.route(BANDS) is Route.ESCALATE


def test_segment_scoring_is_refused_for_a_model_that_reads_the_feeling_chip() -> None:
    """A sentence inside a note has no chip of its own, so there is nothing to supply."""
    layout = FeatureLayout(EMBED_MODEL, EMBED_MODEL_REVISION, 2, True)
    with_feeling = DistressModel(coef=np.zeros(2 + 7), intercept=0.0, layout=layout)
    with pytest.raises(ValueError, match="no chip of its own"):
        segment_scores(["a note"], with_feeling, WordEmbedder())


def test_scoring_no_notes_calls_nothing() -> None:
    embedder = WordEmbedder()
    assert score_notes([], model(), embedder) == []
    assert embedder.calls == 0


def test_float32_noise_is_tolerated_but_a_real_swap_is_not() -> None:
    """The same note embedded among notes and among segments differs in the last bit.

    Measured at 1.2e-7 over the training rows, because embeddings are float32 and
    the two batches sum in a different order. A swapped pair of arrays moves scores
    by 0.1 or more, so noise is absorbed and a swap still raises.
    """
    whole = np.array([0.5, 0.9, 0.3])
    noisy = whole - 1.2e-7
    assert np.array_equal(skip_statistic(whole, noisy), whole)  # restored to the definition

    with pytest.raises(ValueError, match="swapped"):
        skip_statistic(whole, whole - 0.01)


# --- the scope rule --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "もう生きていたくない",
        "나 이제 못 버티겠어",
        "больше не могу",
        "stage 12 done. 다 끝내고 싶어",
    ],
)
def test_a_note_in_a_non_latin_script_escalates(text: str) -> None:
    """A low score from a model that cannot read the input is not evidence of anything."""
    [scored] = score_notes([text], model(), WordEmbedder())
    assert scored.out_of_scope
    assert scored.route(BANDS) is Route.ESCALATE


@pytest.mark.parametrize(
    "text",
    ["café au lait", "don't — really… 😊", "naïve but fine", "ich will nicht mehr leben", "ok 123"],
)
def test_latin_script_notes_stay_in_scope(text: str) -> None:
    """Accents, emoji and curly punctuation are not foreign scripts.

    German is Latin script and so stays in scope: the rule declines text the
    embedder cannot read, and it cannot tell that it reads German badly.
    """
    [scored] = score_notes([text], model(), WordEmbedder())
    assert not scored.out_of_scope


def test_the_scope_rule_outranks_a_skippable_score() -> None:
    rule = ScopeRule()
    assert rule.out_of_scope("もう生きていたくない")
    assert route_note(0.001, 0.001, BANDS) is Route.SKIP_LLM  # the score alone would skip
    [scored] = score_notes(["もう生きていたくない"], model(), WordEmbedder())
    assert scored.route(BANDS) is Route.ESCALATE


def test_the_scope_rule_can_be_turned_off_explicitly() -> None:
    """It is a recorded parameter, not a hard-coded behaviour, so it is testable both ways."""
    off = ScopeRule(escalate_non_latin_letters=False)
    [scored] = score_notes(["もう生きていたくない"], model(), WordEmbedder(), scope=off)
    assert not scored.out_of_scope
