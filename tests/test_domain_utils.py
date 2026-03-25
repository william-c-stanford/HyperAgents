"""Tests for utils/domain_utils.py — all pure functions, no mocking needed."""
import pytest
from utils.domain_utils import (
    get_domain_score_key,
    get_domain_splits,
    can_domain_ensembled,
    get_domain_eval_subset,
    get_domain_test_subset,
    get_domain_stagedeval_samples,
    get_domain_stagedeval_frac,
    has_domain_val_subset,
)


# ---------------------------------------------------------------------------
# get_domain_score_key
# ---------------------------------------------------------------------------

class TestGetDomainScoreKey:
    def test_search_arena(self):
        assert get_domain_score_key("search_arena") == "overall_accuracy"

    def test_paper_review(self):
        assert get_domain_score_key("paper_review") == "overall_accuracy"

    def test_imo_grading(self):
        assert get_domain_score_key("imo_grading") == "overall_accuracy"

    def test_balrog_substring(self):
        assert get_domain_score_key("balrog_babyai") == "average_progress"
        assert get_domain_score_key("balrog_minihack") == "average_progress"

    def test_genesis_substring(self):
        assert get_domain_score_key("genesis_locomotion") == "average_fitness"

    def test_polyglot_substring(self):
        assert get_domain_score_key("polyglot_python") == "accuracy_score"
        assert get_domain_score_key("polyglot") == "accuracy_score"

    def test_imo_proof(self):
        assert get_domain_score_key("imo_proof") == "points_percentage"


# ---------------------------------------------------------------------------
# get_domain_splits
# ---------------------------------------------------------------------------

class TestGetDomainSplits:
    def test_human_pref_no_test(self):
        assert get_domain_splits("search_arena") == ["train", "val"]
        assert get_domain_splits("paper_review") == ["train", "val"]
        assert get_domain_splits("imo_grading") == ["train", "val"]

    def test_human_pref_with_test(self):
        splits = get_domain_splits("search_arena", eval_test=True)
        assert splits == ["train", "val", "test"]

    def test_balrog_single_split(self):
        assert get_domain_splits("balrog_babyai") == ["train"]

    def test_genesis_single_split(self):
        assert get_domain_splits("genesis_locomotion") == ["train"]

    def test_polyglot_single_split(self):
        assert get_domain_splits("polyglot") == ["train"]

    def test_imo_proof_single_split(self):
        assert get_domain_splits("imo_proof") == ["train"]

    def test_eval_test_flag_ignored_for_single_split_domains(self):
        assert get_domain_splits("balrog_babyai", eval_test=True) == ["train"]
        assert get_domain_splits("polyglot", eval_test=True) == ["train"]


# ---------------------------------------------------------------------------
# can_domain_ensembled
# ---------------------------------------------------------------------------

class TestCanDomainEnsembled:
    def test_ensembleable_domains(self):
        assert can_domain_ensembled("search_arena") is True
        assert can_domain_ensembled("paper_review") is True
        assert can_domain_ensembled("imo_grading") is True

    def test_non_ensembleable_domains(self):
        assert can_domain_ensembled("balrog_babyai") is False
        assert can_domain_ensembled("genesis_locomotion") is False
        assert can_domain_ensembled("polyglot") is False
        assert can_domain_ensembled("imo_proof") is False


# ---------------------------------------------------------------------------
# get_domain_eval_subset
# ---------------------------------------------------------------------------

class TestGetDomainEvalSubset:
    def test_filtered_subset_domains(self):
        assert get_domain_eval_subset("search_arena") == "_filtered_100_train"
        assert get_domain_eval_subset("paper_review") == "_filtered_100_train"
        assert get_domain_eval_subset("imo_grading") == "_filtered_100_train"

    def test_empty_subset_domains(self):
        assert get_domain_eval_subset("balrog_babyai") == ""
        assert get_domain_eval_subset("genesis_locomotion") == ""
        assert get_domain_eval_subset("polyglot") == ""
        assert get_domain_eval_subset("imo_proof") == ""


# ---------------------------------------------------------------------------
# get_domain_test_subset
# ---------------------------------------------------------------------------

class TestGetDomainTestSubset:
    def test_filtered_test_subset(self):
        assert get_domain_test_subset("search_arena") == "_filtered_100_test"
        assert get_domain_test_subset("paper_review") == "_filtered_100_test"
        assert get_domain_test_subset("imo_grading") == "_filtered_100_test"

    def test_empty_test_subset(self):
        assert get_domain_test_subset("balrog_babyai") == ""
        assert get_domain_test_subset("genesis_locomotion") == ""
        assert get_domain_test_subset("polyglot") == ""
        assert get_domain_test_subset("imo_proof") == ""


# ---------------------------------------------------------------------------
# get_domain_stagedeval_samples
# ---------------------------------------------------------------------------

class TestGetDomainStagedevalSamples:
    def test_human_pref_domains(self):
        assert get_domain_stagedeval_samples("search_arena") == 10
        assert get_domain_stagedeval_samples("paper_review") == 10
        assert get_domain_stagedeval_samples("imo_grading") == 10
        assert get_domain_stagedeval_samples("imo_proof") == 10

    def test_balrog(self):
        assert get_domain_stagedeval_samples("balrog_babyai") == 1

    def test_genesis(self):
        assert get_domain_stagedeval_samples("genesis_locomotion") == 3

    def test_polyglot(self):
        assert get_domain_stagedeval_samples("polyglot") == 10


# ---------------------------------------------------------------------------
# get_domain_stagedeval_frac
# ---------------------------------------------------------------------------

class TestGetDomainStagedevalFrac:
    def test_human_pref_domains(self):
        assert get_domain_stagedeval_frac("search_arena") == pytest.approx(10 / 100)
        assert get_domain_stagedeval_frac("paper_review") == pytest.approx(10 / 100)
        assert get_domain_stagedeval_frac("imo_grading") == pytest.approx(10 / 100)

    def test_balrog_babyai(self):
        assert get_domain_stagedeval_frac("balrog_babyai") == pytest.approx(1 / 10)

    def test_balrog_minihack(self):
        assert get_domain_stagedeval_frac("balrog_minihack") == pytest.approx(1 / 5)

    def test_genesis(self):
        assert get_domain_stagedeval_frac("genesis_locomotion") == pytest.approx(3 / 6)

    def test_polyglot(self):
        assert get_domain_stagedeval_frac("polyglot") == pytest.approx(10 / 60)

    def test_imo_proof(self):
        assert get_domain_stagedeval_frac("imo_proof") == pytest.approx(10 / 60)


# ---------------------------------------------------------------------------
# has_domain_val_subset
# ---------------------------------------------------------------------------

class TestHasDomainValSubset:
    def test_has_val(self):
        assert has_domain_val_subset("search_arena") is True
        assert has_domain_val_subset("paper_review") is True
        assert has_domain_val_subset("imo_grading") is True

    def test_no_val(self):
        assert has_domain_val_subset("balrog_babyai") is False
        assert has_domain_val_subset("genesis_locomotion") is False
        assert has_domain_val_subset("polyglot") is False
        assert has_domain_val_subset("imo_proof") is False
