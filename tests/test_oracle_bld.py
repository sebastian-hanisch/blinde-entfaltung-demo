"""Orakel: jede Einheit gegen einen anderen Rechenweg (scipy nur hier im Test), gegen np.convolve, gegen die analytischen Mehrdeutigkeiten und gegen die Mini-Instanz.
Linux-CI: in der letzten Stelle nichts festlegen; Medianwerte und Schranken statt Einzelwerte."""

import math

import numpy as np
import pytest
from scipy.optimize import nnls as scipy_nnls
from scipy.stats import multivariate_normal

import bld_constants as C
import bld_evaluation as ev
import bld_model as m
from inverse_generator import SplitMix64, grid


# ----------------------------------------------------------------------------- 1. Faltungsmatrix

def test_toeplitz_matches_np_convolve_same():
    rng = np.random.default_rng(3)
    for half in (1, 3, 6):
        kappa, x = rng.random(2 * half + 1), rng.normal(size=20)
        assert m.toeplitz(kappa, 20) @ x == pytest.approx(np.convolve(x, kappa, mode="same"), abs=1e-12)


def test_toeplitz_of_the_true_kernel_is_the_generator_operator():
    n, w = 48, 0.02
    a = m.op(n, w)
    row = np.array([a[n // 2, n // 2 + d] for d in range(-6, 7)])
    assert np.abs(m.toeplitz(row, n) - a).max() < 1e-8                                                # nur die Ausläufer jenseits von sechs Zellen fehlen
    assert np.abs(m.toeplitz(m.true_kernel(w, n, 6) * row.sum(), n) - m.toeplitz(row, n)).max() < 1e-12


# ----------------------------------------------------------------------------- 2. NNLS und Kernschritt

def test_nnls_matches_scipy_on_random_problems():
    rng = np.random.default_rng(5)
    for _ in range(30):
        a = rng.normal(size=(25, 8))
        b = rng.normal(size=25)
        assert m.nnls(a, b) == pytest.approx(scipy_nnls(a, b)[0], abs=1e-9)


def test_nnls_matches_scipy_on_the_kernel_step_systems():
    n, mm = C.N, C.KERNEL_HALF
    for obj in C.OBJECTS:
        for seed in (1, 2, 3):
            x, b, _, _ = ev.instance(obj, 1e-2, 0.03, seed)
            cols = m.shift_columns(x, mm)
            sys_a = np.vstack([cols, C.NNLS_HEAVY * np.ones((1, 2 * mm + 1))])
            sys_b = np.append(b, C.NNLS_HEAVY)
            ref = scipy_nnls(sys_a, sys_b)[0]
            assert m.kernel_step(x, b, mm) == pytest.approx(ref / ref.sum(), abs=1e-9)


def test_kernel_step_recovers_the_true_kernel_without_noise():
    n, mm = C.N, C.KERNEL_HALF
    kt = m.true_kernel(0.03, n, mm)
    for obj in ("spikes", "phantom"):
        x = ev.truth_signal(obj)
        assert m.kernel_step(x, m.toeplitz(kt, n) @ x, mm) == pytest.approx(kt, abs=1e-8)


def test_the_kernel_step_with_a_binding_constraint_is_not_negative_and_beats_the_clipped_solution():
    n, mm = C.N, C.KERNEL_HALF
    x = ev.truth_signal("spikes")
    kt = m.true_kernel(0.03, n, mm)
    b = m.toeplitz(kt, n) @ x + SplitMix64(4).normal(n) * 0.3                                         # viel Rauschen: der freie Kern hätte negative Werte
    k = m.kernel_step(x, b, mm)
    free = np.linalg.lstsq(m.shift_columns(x, mm), b, rcond=None)[0]
    assert (free < 0).any() and (k >= 0).all() and k.sum() == pytest.approx(1.0)
    cols = m.shift_columns(x, mm)
    assert np.sum((cols @ k - b) ** 2) <= np.sum((cols @ (np.maximum(free, 0) / np.maximum(free, 0).sum()) - b) ** 2) + 1e-12


# ----------------------------------------------------------------------------- 3. Skala und Versatz

def test_scale_symmetry_is_exact():
    n = C.N
    kt = m.true_kernel(0.03, n, 6)
    for obj in C.OBJECTS:
        x = ev.truth_signal(obj)
        for c in (0.5, 3.0, 17.0):
            assert m.toeplitz(kt / c, n) @ (c * x) == pytest.approx(m.toeplitz(kt, n) @ x, abs=1e-12)


def test_shift_symmetry_is_exact_when_the_supports_fit():
    n = C.N
    kt = m.true_kernel(0.03, n, 6)
    x = ev.truth_signal("spikes")                                                                     # die Spitzen liegen in der Mitte: nichts wird abgeschnitten
    k2 = m.shift_vec(kt, 1)
    assert k2.sum() == pytest.approx(kt.sum() - kt[-1])                                              # nur der letzte Kernwert (Ausläufer) fällt heraus
    d1 = m.toeplitz(kt, n) @ x
    d2 = m.toeplitz(k2, n) @ m.shift_vec(x, -1)
    assert np.abs(d2 - d1).max() < 1e-3 * np.abs(d1).max()
    k3 = np.zeros(13)
    k3[5], k3[6], k3[7] = 0.2, 0.5, 0.3
    k4 = m.shift_vec(k3, 1)
    assert m.toeplitz(k4, n) @ m.shift_vec(x, -1) == pytest.approx(m.toeplitz(k3, n) @ x, abs=1e-12)


# ----------------------------------------------------------------------------- 4. Exakte Mehrdeutigkeit der Hügel

def test_hill_ambiguity_is_exact_for_every_delta():
    for delta in (0.0, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 8.9e-4):
        assert m.ambiguity_gap(delta, 0.03, False) < 1e-12


def test_the_box_breaks_the_ambiguity():
    gaps = [m.ambiguity_gap(d, 0.03, True) for d in (2e-4, 4e-4, 6e-4, 8e-4)]
    assert gaps == sorted(gaps) and min(gaps) > 0.01 and max(gaps) < 0.1


def test_the_ambiguity_also_holds_on_the_grid_up_to_the_discretization():
    from inverse_generator import blurred_exact

    t = grid(400)
    new, wp = m.ambiguous_pair(m.BUMPS, 0.03, 4e-4)
    assert m.hills_data(t, 0.03, m.BUMPS) == pytest.approx(blurred_exact(t, 0.03) - m.box_data(t, 0.03), abs=1e-12)
    assert m.hills_data(t, wp, new) == pytest.approx(m.hills_data(t, 0.03, m.BUMPS), abs=1e-12)


# ----------------------------------------------------------------------------- 5. Zielfunktion

@pytest.mark.parametrize("obj", C.OBJECTS)
def test_the_trivial_pair_beats_the_true_pair_at_one_percent_noise(obj):
    wins = 0
    for seed in range(1, 9):
        x, b, sigma, _ = ev.instance(obj, 1e-2, 0.03, seed)
        a = m.op(C.N, 0.03)
        wins += m.trivial_pair(b, sigma, "tikhonov", 6)[2] < m.objective(a, m.x_step(a, b, "tikhonov", sigma), b, "tikhonov", sigma)
    assert wins == 8


@pytest.mark.parametrize("prior", C.PRIORS)
@pytest.mark.parametrize("start", ["delta", "box", "gauss"])
def test_the_objective_never_increases_over_the_rounds(prior, start):
    x, b, sigma, _ = ev.instance("spikes" if prior == "l1" else "phantom", 1e-2, 0.03, 7)
    run = m.alternate(b, sigma, prior, ev.start_kernel(start, 0.05), 25)
    js = np.array(run["js"])
    assert (np.diff(js) <= 1e-12 * js[0]).all()


def test_alternate_matches_a_scipy_reimplementation():
    x0, b, sigma, _ = ev.instance("phantom", 1e-2, 0.03, 7)
    lam = sigma / C.PRIOR_TAU
    kap = ev.start_kernel("gauss", 0.05)
    run = m.alternate(b, sigma, "tikhonov", kap, 5)
    for r in range(5):
        a = m.toeplitz(kap, C.N)
        x = np.linalg.solve(a.T @ a + lam ** 2 * np.eye(C.N), a.T @ b)
        assert run["xs"][r] == pytest.approx(x, abs=1e-8)
        cols = m.shift_columns(x, 6)
        sol = scipy_nnls(np.vstack([cols, 1e3 * np.ones((1, 13))]), np.append(b, 1e3))[0]
        kap = sol / sol.sum()
        assert run["kappas"][r + 1] == pytest.approx(kap, abs=1e-8)


# ----------------------------------------------------------------------------- 6. Evidenz

def test_log_evidence_is_the_gaussian_density_up_to_a_constant():
    n = 12
    a = m.op(n, 0.05)
    p = m.prior_cov(grid(n))
    b = SplitMix64(9).normal(n) * 0.3
    sigma = 0.1
    cov = a @ p @ a.T + sigma ** 2 * np.eye(n)
    assert m.log_evidence(a, b, sigma, p) - 0.5 * n * math.log(2.0 * math.pi) == pytest.approx(multivariate_normal(mean=np.zeros(n), cov=cov).logpdf(b), abs=1e-9)


def test_the_evidence_maximum_is_near_the_true_width_for_prior_draws():
    n, w, sigma = C.N, 0.03, 0.02
    p = m.prior_cov(grid(n))
    chol = np.linalg.cholesky(p)
    gw = np.array(C.PROFILE_GRID)
    a = m.op(n, w)
    hats = []
    for rep in range(30):
        rng = SplitMix64(500 + rep)
        b = a @ (chol @ rng.normal(n)) + rng.normal(n) * sigma
        hats.append(gw[int(np.argmax(m.evidence_curve(b, sigma, gw)))])
    assert abs(float(np.median(hats)) - w) <= 0.05 * w


def test_posterior_mean_matches_the_information_form():
    n = 10
    a = m.op(n, 0.05)
    p = m.prior_cov(grid(n))
    b = SplitMix64(2).normal(n)
    sigma = 0.1
    info = np.linalg.inv(np.linalg.inv(p) + a.T @ a / sigma ** 2) @ a.T @ b / sigma ** 2
    assert m.posterior_mean(a, b, sigma, p) == pytest.approx(info, abs=1e-6)


# ----------------------------------------------------------------------------- 7. x-Prior

def test_tikhonov_satisfies_the_normal_equations():
    a = m.op(C.N, 0.03)
    b = SplitMix64(1).normal(C.N)
    x = m.tikhonov(a, b, 0.2)
    assert a.T @ (a @ x - b) + 0.04 * x == pytest.approx(np.zeros(C.N), abs=1e-10)


def test_lasso_satisfies_the_kkt_conditions():
    rng = np.random.default_rng(8)
    for mu in (0.5, 2.0, 5.0):
        a = rng.normal(size=(20, 12))
        y = rng.normal(size=20)
        w = m.lasso(a, y, mu)
        g = a.T @ (a @ w - y)
        assert np.allclose(g[w != 0.0], -mu * np.sign(w[w != 0.0]), atol=1e-8) and (np.abs(g[w == 0.0]) <= mu + 1e-8).all()


def test_the_prior_swap_gives_the_hand_values_of_stueck_2_and_6(mini):
    kappa, x, b = mini
    a = m.toeplitz(kappa, 3)
    assert m.x_step(a, b, "tikhonov", 0.05) == pytest.approx([1.04951561, 1.88374596, 1.04951561], abs=1e-7)
    assert m.lasso(np.eye(3), np.array([3.0, -0.5, 1.0]), 1.0) == pytest.approx([2.0, 0.0, 0.0])


# ----------------------------------------------------------------------------- 8. Parametrische Alternierung

@pytest.mark.parametrize("obj", ["phantom", "hills"])
@pytest.mark.parametrize("w0", [0.01, 0.03, 0.06])
def test_every_start_is_a_fixed_point_of_the_parametric_alternation(obj, w0):
    x, b, sigma, _ = ev.instance(obj, 1e-2, 0.03, 7)
    run = m.alternate_width(b, sigma, "tikhonov", w0, C.PARAM_GRID, 10)
    assert run["widths"] == [m.snap(w0, C.PARAM_GRID)] * 11


def test_the_parametric_w_step_minimizes_the_data_error_over_the_grid():
    x, b, sigma, _ = ev.instance("phantom", 1e-2, 0.03, 7)
    run = m.alternate_width(b, sigma, "tikhonov", 0.045, C.PARAM_GRID, 1)
    xs0 = run["xs"][0]
    errs = [np.sum((m.op(C.N, v) @ xs0 - b) ** 2) for v in C.PARAM_GRID]
    assert run["widths"][1] == C.PARAM_GRID[int(np.argmin(errs))]
