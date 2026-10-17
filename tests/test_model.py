"""Einheiten von bld_model.py, jede an der Mini-Instanz oder an einer kleinen Aufgabe von Hand gerechnet (DEMO-PLAYBOOK §1b). Orakel gegen andere Rechenwege: test_oracle_bld.py."""

import math

import numpy as np
import pytest

import bld_constants as C
import bld_model as m


# ----------------------------------------------------------------------------- Kerne und Faltungsmatrix

def test_toeplitz_is_the_hand_matrix_and_gives_b(mini):
    kappa, x, b = mini
    a = m.toeplitz(kappa, 3)
    assert a == pytest.approx(np.array([[0.5, 0.25, 0.0], [0.25, 0.5, 0.25], [0.0, 0.25, 0.5]]), abs=1e-15)
    assert a @ x == pytest.approx(b, abs=1e-15)


def test_toeplitz_orientation_with_an_asymmetric_kernel():
    a = m.toeplitz(np.array([0.1, 0.3, 0.6]), 3)                      # A_ij = kappa[i - j + 1]
    assert a[0, 0] == pytest.approx(0.3) and a[1, 0] == pytest.approx(0.6) and a[0, 1] == pytest.approx(0.1) and a[2, 0] == 0.0 and a[0, 2] == 0.0 and a[2, 1] == pytest.approx(0.6)


def test_shift_columns_are_the_shifted_signal(mini):
    _, x, b = mini
    cols = m.shift_columns(x, 1)
    assert cols[:, 0] == pytest.approx([2.0, 1.0, 0.0]) and cols[:, 1] == pytest.approx([1.0, 2.0, 1.0]) and cols[:, 2] == pytest.approx([0.0, 1.0, 2.0])
    assert cols @ np.array([0.25, 0.5, 0.25]) == pytest.approx(b)


def test_delta_box_and_gauss_kernels():
    assert m.delta_kernel(2).tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert m.box_kernel(1) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    k = m.gauss_kernel(0.1, 10, 1)                                    # Verhältnis Rand zu Mitte exp(-(0,1)² / (2·0,01)) = exp(-0,5)
    r = math.exp(-0.5)
    assert k == pytest.approx([r / (1 + 2 * r), 1 / (1 + 2 * r), r / (1 + 2 * r)]) and k.sum() == pytest.approx(1.0) and k[1] == pytest.approx(0.45186, abs=1e-5)


def test_kernel_width_is_the_standard_deviation_of_the_offsets():
    assert m.kernel_width(np.array([0.25, 0.5, 0.25]), 4) == pytest.approx(math.sqrt(0.5 * (1 / 4) ** 2))    # sqrt(0,03125) = 0,176777
    assert m.kernel_width(m.delta_kernel(3), 48) == 0.0
    assert m.kernel_width(np.array([0.5, 0.0, 0.5]), 2) == pytest.approx(0.5)


def test_true_kernel_is_the_normalized_middle_row_and_the_gauss_kernel():
    k = m.true_kernel(0.03, 48, 6)
    assert k.sum() == pytest.approx(1.0) and np.allclose(k, k[::-1]) and k[6] == k.max()
    assert k == pytest.approx(m.gauss_kernel(0.03, 48, 6), abs=1e-12)
    assert k[5] / k[6] == pytest.approx(math.exp(-(1 / 48) ** 2 / (2 * 0.03 ** 2)), rel=1e-12)


def test_op_is_cached_read_only_and_rounds_the_width():
    a = m.op(48, 0.03)
    assert a is m.op(48, 0.03 + 1e-9) and not a.flags.writeable and a.shape == (48, 48)


# ----------------------------------------------------------------------------- NNLS und Kernschritt

def test_nnls_hand_cases():
    assert m.nnls(np.eye(2), np.array([1.0, -1.0])) == pytest.approx([1.0, 0.0])
    assert m.nnls(np.eye(2), np.array([-1.0, -1.0])) == pytest.approx([0.0, 0.0])
    # A = [[1, 0], [1, 1]], b = (2, 0): ohne Nebenbedingung x = (2, -2); mit ihr x2 = 0, x1 = 1 (Gradient der zweiten Spalte -1 <= 0)
    assert m.nnls(np.array([[1.0, 0.0], [1.0, 1.0]]), np.array([2.0, 0.0])) == pytest.approx([1.0, 0.0])
    assert m.nnls(np.array([[1.0, 0.0], [1.0, 1.0]]), np.array([1.0, 3.0])) == pytest.approx([1.0, 2.0])      # Lösung im Innern


def test_nnls_result_is_never_negative_and_zero_columns_stay_zero():
    a = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    x = m.nnls(a, np.array([1.0, 1.0, 1.0]))
    assert (x >= 0.0).all() and x[1] == 0.0


def test_kernel_step_recovers_the_hand_kernel(mini):
    kappa, x, b = mini
    assert m.kernel_step(x, b, 1) == pytest.approx(kappa, abs=1e-9)


def test_kernel_step_has_sum_one_and_is_not_negative():
    rng = np.random.default_rng(0)
    for _ in range(5):
        x, b = rng.normal(size=12), rng.normal(size=12)
        k = m.kernel_step(x, b, 3)
        assert k.sum() == pytest.approx(1.0, abs=1e-12) and (k >= 0.0).all()


# ----------------------------------------------------------------------------- x-Schritte und Zielfunktion

def test_lam_and_mu_are_the_documented_scalings():
    assert m.lam_of(0.05) == pytest.approx(0.1)
    assert m.mu_of(1.0, 48) == pytest.approx(0.5 * math.sqrt(2.0 * math.log(48.0))) and m.mu_of(1.0, 48) == pytest.approx(1.3912, abs=1e-4)
    assert m.mu_of(2.0, 48) == pytest.approx(2.0 * m.mu_of(1.0, 48))


def test_tikhonov_matches_the_hand_values(mini):
    kappa, x, b = mini
    a = m.toeplitz(kappa, 3)
    assert m.tikhonov(a, b, 0.1) == pytest.approx([1.04951561, 1.88374596, 1.04951561], abs=1e-7)
    assert m.tikhonov(a, b, 0.01) == pytest.approx([1.00079554, 1.99840637, 1.00079554], abs=1e-7)


def test_lasso_hand_cases():
    assert m.lasso(np.eye(3), np.array([3.0, -0.5, 1.0]), 1.0) == pytest.approx([2.0, 0.0, 0.0])           # Weichschwelle
    assert m.lasso(2.0 * np.eye(2), np.array([3.0, -3.0]), 1.0) == pytest.approx([1.25, -1.25])           # w = (2 y - mu) / 4
    assert m.lasso(np.eye(2), np.array([0.5, -0.5]), 1.0) == pytest.approx([0.0, 0.0])
    assert m.lasso(np.eye(2), np.array([1.0, 0.0]), 1.0) == pytest.approx([0.0, 0.0])                      # genau auf der Schwelle


def test_x_step_dispatches_on_the_prior(mini):
    kappa, x, b = mini
    a = m.toeplitz(kappa, 3)
    assert m.x_step(a, b, "tikhonov", 0.05) == pytest.approx(m.tikhonov(a, b, 0.1))
    assert m.x_step(a, b, "l1", 0.05) == pytest.approx(m.lasso(a, b, m.mu_of(0.05, 3)))
    assert m.x_step(np.eye(3), np.array([3.0, 0.2, 0.0]), "l1", 1.0 / (0.5 * math.sqrt(2.0 * math.log(3.0)))) == pytest.approx([2.0, 0.0, 0.0])


def test_objective_is_data_plus_penalty(mini):
    kappa, x, b = mini
    a = m.toeplitz(kappa, 3)
    assert m.objective(a, x, b, "tikhonov", 0.05) == pytest.approx(0.01 * 6.0)                      # Datenterm 0, lam² |x|² = 0,01·6
    assert m.objective(a, x, b, "l1", 0.05) == pytest.approx(2.0 * m.mu_of(0.05, 3) * 4.0)           # 2 mu |x|_1
    assert m.objective(a, np.zeros(3), b, "tikhonov", 0.05) == pytest.approx(4.25)                   # |b|²


def test_trivial_pair_has_the_smaller_objective_than_the_true_pair(mini):
    kappa, x, b = mini
    k, xt, j = m.trivial_pair(b, 0.05, "tikhonov", 1)
    assert k.tolist() == [0.0, 1.0, 0.0] and xt == pytest.approx(b / 1.01) and j == pytest.approx(0.042079, abs=1e-6)
    a = m.toeplitz(kappa, 3)
    xo = m.x_step(a, b, "tikhonov", 0.05)
    assert m.objective(a, xo, b, "tikhonov", 0.05) == pytest.approx(0.058665, abs=1e-6) and j < m.objective(a, xo, b, "tikhonov", 0.05)
    _, _, j2 = m.trivial_pair(b, 0.005, "tikhonov", 1)
    assert j2 == pytest.approx(0.000425, abs=1e-6)
    assert m.objective(a, m.x_step(a, b, "tikhonov", 0.005), b, "tikhonov", 0.005) == pytest.approx(0.000600, abs=1e-6)
    _, xl, jl = m.trivial_pair(b, 0.2, "l1", 1)                                                      # A = I: Weichschwelle mit mu = 0,5·0,2·sqrt(2 ln 3)
    mu = m.mu_of(0.2, 3)
    assert xl == pytest.approx(np.maximum(np.abs(b) - mu, 0.0) * np.sign(b)) and jl == pytest.approx(np.sum((xl - b) ** 2) + 2.0 * mu * np.abs(xl).sum())


# ----------------------------------------------------------------------------- Alternierung

def test_alternate_shapes_start_and_monotone_objective(mini):
    kappa, x, b = mini
    k0 = np.array([0.1, 0.8, 0.1])
    run = m.alternate(b, 0.05, "tikhonov", k0, 4)
    assert len(run["kappas"]) == len(run["xs"]) == len(run["js"]) == 5 and run["kappas"][0].tolist() == k0.tolist()
    assert all(run["js"][i + 1] <= run["js"][i] + 1e-12 for i in range(4))
    assert all(abs(k.sum() - 1.0) < 1e-12 and (k >= 0).all() for k in run["kappas"])
    a0 = m.toeplitz(k0, 3)
    assert run["xs"][0] == pytest.approx(m.tikhonov(a0, b, 0.1)) and run["js"][0] == pytest.approx(m.objective(a0, run["xs"][0], b, "tikhonov", 0.05))
    assert run["kappas"][1] == pytest.approx(m.kernel_step(run["xs"][0], b, 1))


def test_alternate_with_zero_rounds_is_one_x_step(mini):
    kappa, x, b = mini
    run = m.alternate(b, 0.05, "tikhonov", kappa, 0)
    assert len(run["kappas"]) == 1 and run["xs"][0] == pytest.approx(m.tikhonov(m.toeplitz(kappa, 3), b, 0.1))


def test_alternate_with_l1_prior_runs_and_never_increases(mini):
    kappa, x, b = mini
    run = m.alternate(b, 0.05, "l1", np.array([0.2, 0.6, 0.2]), 3)
    assert all(run["js"][i + 1] <= run["js"][i] + 1e-12 for i in range(3)) and run["xs"][0].shape == (3,)


# ----------------------------------------------------------------------------- Profil und parametrische Alternierung

def test_snap_picks_the_nearest_grid_value_and_the_smaller_on_a_tie():
    assert m.snap(0.031, (0.03, 0.04)) == 0.03 and m.snap(0.039, (0.03, 0.04)) == 0.04 and m.snap(5.0, (4.0, 6.0)) == 4.0 and m.snap(0.0, (0.03, 0.04)) == 0.03


def test_profile_costs_are_the_objective_at_each_width():
    b = m.op(8, 0.05) @ np.arange(8.0)
    costs = m.profile_costs(b, 0.01, "tikhonov", (0.03, 0.05))
    for c, w in zip(costs, (0.03, 0.05)):
        a = m.op(8, w)
        assert c == pytest.approx(m.objective(a, m.tikhonov(a, b, 0.02), b, "tikhonov", 0.01))
    assert costs[0] < costs[1]                                                                       # der schmalere Kern hat die kleineren Kosten, auch wenn die Daten mit 0,05 entstanden
    assert m.profile_costs(b, 0.01, "l1", (0.05,))[0] == pytest.approx(m.objective(m.op(8, 0.05), m.lasso(m.op(8, 0.05), b, m.mu_of(0.01, 8)), b, "l1", 0.01))


def test_alternate_width_returns_snapped_start_and_aligned_histories():
    b = m.op(12, 0.05) @ np.linspace(0.0, 1.0, 12) ** 2
    run = m.alternate_width(b, 0.01, "tikhonov", 0.0501, (0.04, 0.05, 0.06), 3)
    assert run["widths"][0] == 0.05 and len(run["widths"]) == len(run["xs"]) == len(run["js"]) == 4
    assert all(w in (0.04, 0.05, 0.06) for w in run["widths"])
    assert run["xs"][0] == pytest.approx(m.tikhonov(m.op(12, 0.05), b, 0.02))
    assert all(run["js"][i + 1] <= run["js"][i] + 1e-12 for i in range(3))


# ----------------------------------------------------------------------------- Evidenz

def test_exp_kernel_and_prior_cov_hand_values():
    k = m.exp_kernel(np.array([0.0, 0.12, 0.24]), 0.12)
    assert k[0, 1] == pytest.approx(math.exp(-1.0)) and k[0, 2] == pytest.approx(math.exp(-2.0)) and np.allclose(np.diag(k), 1.0) and np.allclose(k, k.T)
    p = m.prior_cov(np.array([0.0, 0.12]))
    assert p[0, 0] - 0.25 == pytest.approx(1e-10, rel=1e-3) and p[0, 1] == pytest.approx(0.25 * math.exp(-1.0)) and p[1, 0] == p[0, 1]


def test_log_evidence_scalar_hand_values():
    p = np.array([[1.0]])
    assert m.log_evidence(np.array([[math.sqrt(0.99)]]), np.array([1.0]), 0.1, p) == pytest.approx(-0.5)      # a² p + sigma² = 1: -1/2 b²/1 - 1/2 log 1
    e = [m.log_evidence(np.array([[math.sqrt(v)]]), np.array([1.0]), 0.1, p) for v in (0.9, 0.99, 1.1)]
    assert e[1] > e[0] and e[1] > e[2]                                                                # Maximum bei a² p = b² - sigma²
    assert m.log_evidence(np.array([[1.0]]), np.array([2.0]), 1.0, p) == pytest.approx(-0.5 * 4.0 / 2.0 - 0.5 * math.log(2.0))


def test_posterior_mean_scalar_hand_value():
    assert m.posterior_mean(np.array([[1.0]]), np.array([1.0]), 0.1, np.array([[1.0]])) == pytest.approx([1.0 / 1.01])
    assert m.posterior_mean(np.array([[2.0]]), np.array([1.0]), 1.0, np.array([[0.5]])) == pytest.approx([0.5 * 2.0 / (4.0 * 0.5 + 1.0)])


def test_evidence_curve_has_one_value_per_width_and_uses_the_generator_operator():
    b = m.op(12, 0.05) @ np.linspace(0.0, 1.0, 12)
    curve = m.evidence_curve(b, 0.01, (0.04, 0.05))
    from inverse_generator import grid

    p = m.prior_cov(grid(12))
    assert curve.shape == (2,) and curve[1] == pytest.approx(m.log_evidence(m.op(12, 0.05), b, 0.01, p))


# ----------------------------------------------------------------------------- Mehrdeutigkeit

def test_hills_data_hand_value():
    assert m.hills_data(np.array([0.5]), 0.03, ((0.5, 0.04, 1.0),))[0] == pytest.approx(0.8)             # b / sqrt(b² + w²) = 0,04 / 0,05
    assert m.hills_data(np.array([0.5]), 0.0, ((0.5, 0.04, 2.0),))[0] == pytest.approx(2.0)
    assert m.hills_data(np.array([0.5 + 0.05]), 0.03, ((0.5, 0.04, 1.0),))[0] == pytest.approx(0.8 * math.exp(-0.5 * (0.05 / 0.05) ** 2))


def test_box_data_edges_and_middle():
    t = np.array([0.2, 0.4, 0.6, 0.0, 1.0])
    out = m.box_data(t, 0.001, (0.2, 0.6, 1.0))
    assert out[:3] == pytest.approx([0.5, 1.0, 0.5], abs=1e-9) and out[3] == pytest.approx(0.0, abs=1e-9) and out[4] == pytest.approx(0.0, abs=1e-9)


def test_ambiguous_pair_hand_values_and_invariant():
    new, wp = m.ambiguous_pair(((0.5, 0.05, 1.0),), 0.03, 0.0009)
    (c, bp, hp), = new
    assert c == 0.5 and bp == pytest.approx(0.04) and hp == pytest.approx(1.25) and wp == pytest.approx(math.sqrt(0.0018))
    assert bp ** 2 + wp ** 2 == pytest.approx(0.05 ** 2 + 0.03 ** 2)


def test_ambiguity_gap_is_zero_for_hills_and_positive_with_the_box():
    assert m.ambiguity_gap(0.0004, 0.03, False) < 1e-12 and m.ambiguity_gap(0.0, 0.03, True) < 1e-12
    assert m.ambiguity_gap(0.0004, 0.03, True) > 0.01 and m.ambiguity_gap(0.0008, 0.03, True) > m.ambiguity_gap(0.0004, 0.03, True)


def test_shift_vec():
    v = np.array([1.0, 2.0, 3.0])
    assert m.shift_vec(v, 1).tolist() == [0.0, 1.0, 2.0] and m.shift_vec(v, -1).tolist() == [2.0, 3.0, 0.0] and m.shift_vec(v, 0).tolist() == [1.0, 2.0, 3.0]
    assert m.shift_vec(v, 3).tolist() == [0.0, 0.0, 0.0] and m.shift_vec(v, -2).tolist() == [3.0, 0.0, 0.0]


def test_constants_used_by_the_model_are_consistent():
    assert C.N == 48 and C.KERNEL_HALF == 6 and C.PROFILE_GRID[0] == pytest.approx(0.01) and C.PROFILE_GRID[-1] == pytest.approx(0.06) and len(C.PROFILE_GRID) == 26
    assert C.PARAM_GRID[0] == pytest.approx(0.005) and C.PARAM_GRID[-1] == pytest.approx(0.08) and len(C.PARAM_GRID) == 76
