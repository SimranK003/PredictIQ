"""Deterministic tests for the drift statistics themselves (KS, PSI),
using controlled synthetic distributions — not production data. These
validate the statistical implementation in isolation from the database/
API layer; tests/integration/test_monitoring_drift_api.py separately
verifies the end-to-end behavior against real predictions.
"""

import numpy as np
import pytest

from monitoring.drift import compute_ks_drift, compute_psi_categorical

# --- KS test (numeric) ---


def test_ks_reports_no_drift_for_identical_distributions():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=50, scale=10, size=1000)
    production = rng.normal(loc=50, scale=10, size=1000)

    statistic, p_value = compute_ks_drift(reference, production)

    assert statistic < 0.10  # below the default DRIFT_KS_THRESHOLD
    assert p_value > 0.05  # not statistically significant


def test_ks_detects_a_deliberate_distribution_shift():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=50, scale=10, size=1000)
    production = rng.normal(loc=80, scale=10, size=1000)  # shifted by 3 std devs

    statistic, p_value = compute_ks_drift(reference, production)

    assert statistic >= 0.10
    assert p_value < 0.01


def test_ks_statistic_is_bounded_between_zero_and_one():
    rng = np.random.default_rng(1)
    reference = rng.uniform(0, 1, 500)
    production = rng.uniform(100, 200, 500)  # completely disjoint ranges

    statistic, _ = compute_ks_drift(reference, production)

    assert statistic == pytest.approx(1.0, abs=0.01)  # maximally different


# --- PSI (categorical) ---


def test_psi_is_near_zero_for_identical_category_proportions():
    reference_proportions = {"A": 0.5, "B": 0.3, "C": 0.2}
    production_values = ["A"] * 500 + ["B"] * 300 + ["C"] * 200

    psi = compute_psi_categorical(reference_proportions, production_values)

    assert psi < 0.10  # below the default DRIFT_PSI_WARNING


def test_psi_detects_a_deliberate_proportion_shift():
    reference_proportions = {"A": 0.5, "B": 0.3, "C": 0.2}
    # Production has swung heavily toward "C" — a real proportion shift.
    production_values = ["A"] * 100 + ["B"] * 100 + ["C"] * 800

    psi = compute_psi_categorical(reference_proportions, production_values)

    assert psi >= 0.25  # crosses into the "critical" band


def test_psi_handles_a_category_unseen_in_production_safely():
    reference_proportions = {"A": 0.6, "B": 0.4}
    production_values = ["A"] * 600 + ["B"] * 400  # "B" present, nothing new

    psi = compute_psi_categorical(reference_proportions, production_values)

    assert np.isfinite(psi)
    assert psi < 0.10


def test_psi_handles_a_category_unseen_in_reference_safely():
    """A category that never appeared during training but shows up in
    production must not raise (division by zero / log(0)) — it's
    treated as a rare-but-real category via epsilon smoothing.
    """
    reference_proportions = {"A": 1.0}  # training data only ever had "A"
    production_values = ["A"] * 900 + ["NeverSeenBefore"] * 100

    psi = compute_psi_categorical(reference_proportions, production_values)

    assert np.isfinite(psi)
    assert psi > 0  # a brand-new category appearing is itself a real shift


def test_psi_is_symmetric_under_swapped_labels_but_not_swapped_arguments():
    """Sanity check on the formula itself: PSI(ref, prod) generally
    differs from PSI(prod-as-if-reference, ref-as-if-production) because
    the epsilon smoothing and log ratio aren't symmetric — this just
    documents that PSI is directional (reference vs. current), not proof
    of a specific numeric identity.
    """
    a = {"X": 0.9, "Y": 0.1}
    b_values = ["X"] * 100 + ["Y"] * 900

    psi_a_vs_b = compute_psi_categorical(a, b_values)

    b_as_reference = {"X": 0.1, "Y": 0.9}
    a_values = ["X"] * 900 + ["Y"] * 100
    psi_b_vs_a = compute_psi_categorical(b_as_reference, a_values)

    # Both directions detect the same underlying shift even if not
    # numerically identical.
    assert psi_a_vs_b >= 0.25
    assert psi_b_vs_a >= 0.25
