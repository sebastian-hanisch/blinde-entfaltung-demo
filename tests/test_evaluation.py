"""Formatierung, Welt, Instanz, Analyse einer Einstellung und die Verzweigungen des Urteils (jede Schwelle auf der Kante)."""

import dataclasses
import math

import numpy as np
import pytest

import bld_constants as C
import bld_evaluation as ev
import bld_model as m
from inverse_generator import SplitMix64


def S(**kw) -> ev.Settings:
    return ev.Settings(**kw)


# ----------------------------------------------------------------------------- Formatierung

def test_de_uses_comma_and_typographic_minus():
    assert ev.de(-1.5, 1) == "−1,5" and ev.de(0.04, 3) == "0,040" and ev.de(1234.5, 0) in ("1234", "1235")


def test_sci_formats_powers_of_ten_and_rounds_up_the_mantissa():
    assert ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(0.00039) == "3,9·10⁻⁴" and ev.sci(9.96e-3) == "1,0·10⁻²" and ev.sci(0.0) == "0,0" and ev.sci(math.inf) == "unendlich"
    assert ev.sci(-2.5e-3) == "−2,5·10⁻³" and ev.sci(1.0) == "1,0·10⁰"


def test_err_pct_has_four_ranges_and_a_factor_form():
    assert ev.err_pct(0.5) == "50,0 %" and ev.err_pct(1.19) == "119 %" and ev.err_pct(0.0123) == "1,2 %" and ev.err_pct(0.00123) == "0,12 %" and ev.err_pct(1e-5) == "1,0·10⁻³ %"
    assert ev.err_pct(12.0) == "1,2·10¹-fach" and ev.err_pct(math.inf) == "unendlich" and ev.err_pct(9.99) == "999 %" and ev.err_pct(0.9951) == "100 %"


def test_noise_label_and_width_label():
    assert ev.noise_label(0.01) == "1 %" and ev.noise_label(0.001) == "0,1 %" and ev.noise_label(0.1) == "10 %" and ev.noise_label(0.0) == "0 %"
    assert ev.width_label(0.0297) == "0,0297" and ev.width_label(0.03) == "0,0300"


def test_rel_err_is_relative_to_the_truth_norm():
    assert ev.rel_err(np.array([3.0, 0.0]), np.array([0.0, 4.0])) == pytest.approx(5.0 / 4.0)
    assert ev.rel_err(np.ones(3), np.ones(3)) == 0.0


# ----------------------------------------------------------------------------- Welt und Instanz

def test_truth_signals():
    s = np.arange(48)
    ph, hi, sp = ev.truth_signal("phantom"), ev.truth_signal("hills"), ev.truth_signal("spikes")
    box = (m.grid(48) >= 0.72) & (m.grid(48) < 0.84)
    assert box.sum() == 5 and ph[box].min() > 0.5 and np.abs(hi[box]).max() < 1e-4 and hi[~box] == pytest.approx(ph[~box])
    assert np.flatnonzero(sp).tolist() == [12, 24, 33] and sp[[12, 24, 33]] == pytest.approx([0.04 * 48, 0.03 * 48, 0.05 * 48])
    assert ph.max() == pytest.approx(1.0, abs=0.02) and not ph.flags.writeable and ev.truth_signal("phantom") is ph and s.size == 48


def test_instance_noise_scale_and_determinism():
    x, b, sigma, clean = ev.instance("phantom", 1e-2, 0.03, 7)
    assert clean == pytest.approx(m.op(48, 0.03) @ x) and sigma == pytest.approx(0.01 * np.sqrt(np.mean(clean ** 2)))
    assert b - clean == pytest.approx(SplitMix64(7).normal(48) * sigma)
    assert ev.instance("phantom", 1e-2, 0.03, 7) is ev.instance("phantom", 1e-2, 0.03, 7)
    assert not np.allclose(b, ev.instance("phantom", 1e-2, 0.03, 8)[1]) and ev.instance("phantom", 1e-1, 0.03, 7)[2] == pytest.approx(10.0 * sigma)
    assert ev.noise_sigma(np.array([3.0, 4.0]), 0.1) == pytest.approx(0.1 * math.sqrt(12.5))


def test_start_kernels():
    assert ev.start_kernel("delta", 0.05).tolist() == m.delta_kernel(6).tolist()
    assert ev.start_kernel("box", 0.05) == pytest.approx(np.full(13, 1 / 13))
    assert ev.start_kernel("gauss", 0.05) == pytest.approx(m.gauss_kernel(0.05, 48, 6)) and ev.start_kernel("gauss", 0.01)[6] > ev.start_kernel("gauss", 0.05)[6]


def test_default_settings_equal_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


# ----------------------------------------------------------------------------- Analyse

def test_analysis_of_the_standard_instance():
    a = ev.analyse(S())
    x, b, sigma, _ = ev.instance("phantom", 1e-2, 0.03, 7)
    assert a.x is x and a.b is b and a.sigma == sigma and a.s == pytest.approx(m.grid(48))
    assert len(a.js) == len(a.widths_hist) == len(a.xerrs_hist) == len(a.kerrs_hist) == 61
    assert a.kappa_hat.sum() == pytest.approx(1.0) and a.kappa_true == pytest.approx(m.true_kernel(0.03, 48, 6)) and a.kappa_start == pytest.approx(m.gauss_kernel(0.04, 48, 6))
    assert a.kerr == pytest.approx(ev.rel_err(a.kappa_hat, a.kappa_true)) and a.xerr == pytest.approx(ev.rel_err(a.x_hat, x)) and a.xerr_known == pytest.approx(ev.rel_err(a.x_known, x))
    assert a.xerr_triv == pytest.approx(ev.rel_err(a.x_triv, x)) and a.w_hat == a.widths_hist[-1] and a.w_true == 0.03 and a.kerr_start == pytest.approx(ev.rel_err(a.kappa_start, a.kappa_true))
    assert a.score_kind == "J" and a.score_hat == a.js[-1] and a.score_true == pytest.approx(m.objective(m.op(48, 0.03), a.x_known, b, "tikhonov", sigma))
    assert a.score_triv == pytest.approx(m.trivial_pair(b, sigma, "tikhonov", 6)[2]) and not a.at_edge
    assert a.grid_w.tolist() == list(C.PROFILE_GRID) and len(a.prof) == len(a.evid) == 26 and a.w_prof == pytest.approx(0.01) and a.w_evid == pytest.approx(0.036)
    assert a.prof == pytest.approx(m.profile_costs(b, sigma, "tikhonov", C.PROFILE_GRID)) and a.evid == pytest.approx(m.evidence_curve(b, sigma, C.PROFILE_GRID))


def test_rounds_set_the_history_length():
    assert len(ev.analyse(S(rounds=1)).js) == 2 and len(ev.analyse(S(rounds=5)).kerrs_hist) == 6


def test_delta_start_stays_delta_and_ends_at_the_trivial_pair():
    a = ev.analyse(S(start="delta"))
    assert a.w_hat == 0.0 and a.kappa_hat == pytest.approx(m.delta_kernel(6), abs=1e-6) and a.score_hat == pytest.approx(a.score_triv, rel=1e-9) and a.score_hat < a.score_true


def test_profile_analysis():
    a = ev.analyse(S(method="profile"))
    assert a.js == [] and a.widths_hist == [] and a.score_kind == "J" and a.score_hat == pytest.approx(float(a.prof.min())) and a.w_hat == a.w_prof == pytest.approx(0.01) and a.at_edge
    assert a.kappa_hat == pytest.approx(m.gauss_kernel(0.01, 48, 6)) and a.x_hat == pytest.approx(m.x_step(m.op(48, 0.01), a.b, "tikhonov", a.sigma))


def test_evidence_analysis():
    a = ev.analyse(S(method="evidence"))
    p = m.prior_cov(m.grid(48))
    assert a.js == [] and a.score_kind == "evidence" and a.score_hat == pytest.approx(float(a.evid.max())) and math.isnan(a.score_triv) and a.w_hat == a.w_evid == pytest.approx(0.036) and not a.at_edge
    assert a.score_true == pytest.approx(m.log_evidence(m.op(48, 0.03), a.b, a.sigma, p)) and a.x_hat == pytest.approx(m.posterior_mean(m.op(48, 0.036), a.b, a.sigma, p))
    assert a.x_known == pytest.approx(m.posterior_mean(m.op(48, 0.03), a.b, a.sigma, p)) and a.score_hat > a.score_true


def test_evidence_at_the_grid_edge(monkeypatch):
    monkeypatch.setattr(C, "PROFILE_GRID", (0.04, 0.05, 0.06))
    assert ev.analyse(S(method="evidence")).at_edge
    monkeypatch.setattr(C, "PROFILE_GRID", (0.01, 0.02, 0.03))
    assert ev.analyse(S(method="evidence")).at_edge
    monkeypatch.setattr(C, "PROFILE_GRID", (0.03, 0.036, 0.04))
    assert not ev.analyse(S(method="evidence")).at_edge


def test_parametric_alternation_analysis():
    a = ev.analyse(S(kernel="gauss", start_w=0.03))
    assert len(a.js) == 61 and a.widths_hist[0] == pytest.approx(0.03) and a.w_hat == a.widths_hist[-1] and a.kappa_hat == pytest.approx(m.gauss_kernel(a.w_hat, 48, 6))
    assert a.kerrs_hist[0] == pytest.approx(0.0, abs=1e-12) and a.kerr_start == pytest.approx(0.0, abs=1e-12) and a.score_hat == a.js[-1]
    assert a.x_hat == pytest.approx(m.x_step(m.op(48, a.w_hat), a.b, "tikhonov", a.sigma))


def test_parametric_start_is_snapped_to_the_grid_but_the_free_start_is_not():
    free = ev.analyse(S(start_w=0.0304, rounds=1))
    par = ev.analyse(S(kernel="gauss", start_w=0.0304, rounds=1))
    assert free.kerr_start > 1e-3 and par.kerr_start < 1e-9 and par.widths_hist[0] == pytest.approx(0.03)


def test_l1_prior_is_used_in_the_x_step_and_the_score():
    a = ev.analyse(S(obj="spikes", prior="l1", start_w=0.06))
    assert a.score_true == pytest.approx(m.objective(m.op(48, 0.03), a.x_known, a.b, "l1", a.sigma)) and a.x_known == pytest.approx(m.lasso(m.op(48, 0.03), a.b, m.mu_of(a.sigma, 48)))
    assert (np.abs(a.x_known) < 1e-12).sum() > 30


# ----------------------------------------------------------------------------- Verdikt: jede Verzweigung

def _v(**kw):
    return ev.verdict_instance(dataclasses.replace(ev.analyse(S()), **kw))


def test_verdict_found_the_kernel():
    kind, text = ev.verdict_instance(ev.analyse(S(**C.PRESETS["Spitzen mit L1"])))
    assert kind == "success" and "Der Kern ist gefunden" in text and "vom Start (47,8 %) auf 1,7 %" in text


def test_verdict_confirmed_only_the_start():
    kind, text = ev.verdict_instance(ev.analyse(S(**C.PRESETS["Start passt"])))
    assert kind == "warning" and "Start bestätigt, nicht den Kern gefunden" in text and "Kern-Fehler null" in text


def test_verdict_trivial_pair():
    kind, text = ev.verdict_instance(ev.analyse(S(**C.PRESETS["Start bei Delta"])))
    assert kind == "warning" and "Das Ergebnis ist das Trivialpaar" in text and "Startbreite" not in text


def test_verdict_the_objective_does_not_prefer_the_truth_and_the_width_stayed():
    kind, text = ev.verdict_instance(ev.analyse(S()))
    assert kind == "warning" and "das Ziel zeichnet die Wahrheit nicht aus" in text and "Die Breite ist bei der Startbreite 0,0400 geblieben." in text


def test_verdict_stuck_in_the_start():
    kind, text = ev.verdict_instance(ev.analyse(S(**C.PRESETS["Start zu breit"])))
    assert kind == "warning" and "im Start steckengeblieben" in text and "Die Breite ist bei der Startbreite 0,0600 geblieben." in text
    box = ev.verdict_instance(ev.analyse(S(start="box")))[1]
    assert "im Start steckengeblieben" in box and "bei der Startbreite" not in box
    gauss = ev.verdict_instance(ev.analyse(S(kernel="gauss")))[1]
    assert "im Start steckengeblieben" in gauss and "bei der Startbreite" not in gauss


def test_verdict_profile_and_evidence():
    kind, text = ev.verdict_instance(ev.analyse(S(**C.PRESETS["Profil über die Breite"])))
    assert kind == "warning" and "Profil, Phantom, Rauschen 1 %" in text and "am Rand des Gitters (0,0100)" in text
    kind, text = ev.verdict_instance(ev.analyse(S(**C.PRESETS["Evidenz"])))
    assert kind == "warning" and "Evidenz, Phantom" in text and "um mehr als 10 %" in text and "Rand des Gitters" not in text and "log p(b | w′) = 161,9 bei ŵ, 156,4 bei der wahren Breite" in text
    kind, text = ev.verdict_instance(ev.analyse(S(method="evidence", noise=0.1)))
    assert kind == "success" and "innerhalb von 10 % der wahren" in text


def test_verdict_text_mentions_scores_of_the_method():
    text = ev.verdict_instance(ev.analyse(S()))[1]
    assert "Zielfunktion J = " in text and "Trivialpaar 2,4·10⁻⁴" in text and "kleiner ist besser" in text and "log p" not in text
    assert "Fehler der Rekonstruktion 25,9 % (bei bekanntem Kern 29,3 %, Trivialpaar 21,8 %)" in text and "Kern-Fehler 23,3 %" in text


def test_kernel_ok_threshold_is_inclusive(monkeypatch):
    monkeypatch.setattr(C, "KERNEL_OK", 0.5)
    assert _v(kerr=0.5, kerr_start=0.9)[0] == "success" and _v(kerr=0.5, kerr_start=0.5)[1].count("bestätigt") == 1
    assert "falsch" in _v(kerr=0.5000001, kerr_start=0.9)[1] and "Der Kern ist gefunden" not in _v(kerr=0.5000001, kerr_start=0.9)[1]
    assert "Der Kern ist gefunden" in _v(kerr=0.5, kerr_start=0.5000001)[1]


def test_start_error_counts_as_null_only_below_the_threshold(monkeypatch):
    monkeypatch.setattr(C, "NULL_ERR", 0.1)
    assert "Kern-Fehler null" in _v(kerr=0.05, kerr_start=0.09)[1] and "Kern-Fehler 10,0 %" in _v(kerr=0.05, kerr_start=0.1)[1]


def test_width_ok_threshold_is_inclusive(monkeypatch):
    monkeypatch.setattr(C, "WIDTH_OK", 0.25)
    base = dataclasses.replace(ev.analyse(S(method="evidence")), w_true=1.0)
    assert ev.verdict_instance(dataclasses.replace(base, w_hat=1.25))[0] == "success" and ev.verdict_instance(dataclasses.replace(base, w_hat=0.75))[0] == "success"
    assert ev.verdict_instance(dataclasses.replace(base, w_hat=1.2500001))[0] == "warning" and ev.verdict_instance(dataclasses.replace(base, w_hat=0.7499999))[0] == "warning"
    assert "Rand des Gitters" in ev.verdict_instance(dataclasses.replace(base, w_hat=3.0, at_edge=True))[1] and "Rand des Gitters" not in ev.verdict_instance(dataclasses.replace(base, w_hat=3.0))[1]


def test_stays_threshold_is_inclusive(monkeypatch):
    monkeypatch.setattr(C, "STAYS_TOL", 0.25)
    st = dataclasses.replace(ev.analyse(S()).settings, start_w=1.0)
    a = dataclasses.replace(ev.analyse(S()), settings=st, kerr=0.9, kerr_start=0.9)
    assert "bei der Startbreite" in ev.verdict_instance(dataclasses.replace(a, w_hat=1.25))[1] and "bei der Startbreite" in ev.verdict_instance(dataclasses.replace(a, w_hat=0.75))[1]
    assert "bei der Startbreite" not in ev.verdict_instance(dataclasses.replace(a, w_hat=1.2500001))[1] and "bei der Startbreite" not in ev.verdict_instance(dataclasses.replace(a, w_hat=0.7499999))[1]


def test_trivial_width_threshold_is_strict(monkeypatch):
    monkeypatch.setattr(C, "TRIVIAL_WIDTH", 0.5)
    assert "Trivialpaar (Kern fast ein Delta" in _v(kerr=0.9, w_hat=0.4999999)[1] and "Trivialpaar (Kern fast ein Delta" not in _v(kerr=0.9, w_hat=0.5)[1]


def test_score_comparison_is_inclusive():
    assert "nicht größer" in _v(kerr=0.9, w_hat=0.03, score_hat=1.0, score_true=1.0)[1] and "im Start steckengeblieben" in _v(kerr=0.9, w_hat=0.03, score_hat=1.0000001, score_true=1.0)[1]


def test_by_finds_the_row_with_all_keys():
    rows = [{"a": 1, "b": 2}, {"a": 1, "b": 3}, {"a": 2, "b": 3}]
    assert ev._by(rows, a=1, b=3) is rows[1] and ev._by(rows, a=2) is rows[2]
