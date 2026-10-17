"""Aussagen aus README, Verdikten und Hilfetexten, einzeln an den festen Strömen gemessen (Seeds ab SEED_BASE). Ändert sich eine Zahl, muss der Text mit.
Linux-CI: in der letzten Stelle nichts festlegen; Medianwerte mit Toleranz, Schranken statt Einzelwerte."""

import pytest

import bld_constants as C
import bld_evaluation as ev


@pytest.fixture(scope="module")
def ambiguity():
    return ev.ambiguity_experiment()


@pytest.fixture(scope="module")
def objective():
    return ev.objective_experiment()


@pytest.fixture(scope="module")
def alternation():
    return ev.alternation_experiment()


@pytest.fixture(scope="module")
def evidence():
    return ev.evidence_experiment()


def cell(rows, **kw):
    return ev._by(rows, **kw)


# ----------------------------------------------------------------------------- Experiment 1

def test_ambiguity_numbers(ambiguity):
    assert [r["delta"] for r in ambiguity["rows"]] == list(C.AMBIGUITY_DELTAS)
    assert [round(r["w_new"], 4) for r in ambiguity["rows"]] == pytest.approx([0.0332, 0.0361, 0.0387, 0.0412], abs=1e-4)
    assert all(r["gap_hills"] < 1e-12 for r in ambiguity["rows"])
    assert [100 * r["gap_box"] for r in ambiguity["rows"]] == pytest.approx([1.5, 2.85, 4.08, 5.21], abs=0.05)
    assert all(r["scale"] < 1e-12 for r in ambiguity["sym"]) and max(r["shift"] for r in ambiguity["sym"]) == pytest.approx(1.7e-4, abs=0.3e-4)
    assert len(ambiguity["steps"]) == 9 and len(ambiguity["sym"]) == 3


def test_kernel_step_numbers(ambiguity):
    steps = ambiguity["steps"]
    sp = [cell(steps, obj="spikes", noise=nz)["median"] for nz in C.NOISE_CHOICES]
    ph = [cell(steps, obj="phantom", noise=nz)["median"] for nz in C.NOISE_CHOICES]
    hi = [cell(steps, obj="hills", noise=nz)["median"] for nz in C.NOISE_CHOICES]
    assert sp == pytest.approx([0.0005, 0.0047, 0.044], abs=0.003) and ph == pytest.approx([0.0048, 0.042, 0.32], abs=0.02) and hi == pytest.approx([0.082, 0.30, 0.83], abs=0.05)
    assert all(sp[i] < ph[i] < hi[i] for i in range(3))
    assert all(r["max"] >= r["median"] for r in steps)
    text = ev.verdict_ambiguity(ambiguity)[1]
    assert "bei δ = 0,0002 bis 0,0008 (Breite 0,0332 bis 0,0412 statt 0,0300)" in text and "unter 10⁻¹²" in text and "mit ihm sind es 1,5 bis 5,2 %" in text
    assert "bei Spitzen auf 0,47 %" in text and "beim Phantom auf 4,2 %" in text and "bei den Hügeln allein nur auf 29,6 %" in text and "im Innern bis 1,7·10⁻⁴" in text


# ----------------------------------------------------------------------------- Experiment 2

def test_the_trivial_pair_wins_the_objective_in_every_cell(objective):
    rows = objective["rows"]
    assert len(rows) == 9 and all(r["share_triv_smaller"] == 1.0 and r["j_triv"] < r["j_true"] for r in rows)
    assert all(r["w_prof"] == pytest.approx(0.01) and r["share_edge"] == 1.0 for r in rows)


def test_objective_numbers(objective):
    ph, sp, hi = (cell(objective["rows"], obj=o, noise=1e-2) for o in ("phantom", "spikes", "hills"))
    assert (ph["j_true"], ph["j_triv"]) == pytest.approx((4.7e-4, 2.4e-4), rel=0.05) and (sp["j_true"], sp["j_triv"]) == pytest.approx((2.2e-4, 4.2e-5), rel=0.06)
    assert (ph["xerr_true"], ph["xerr_triv"]) == pytest.approx((0.275, 0.219), abs=0.01) and (sp["xerr_true"], sp["xerr_triv"]) == pytest.approx((0.529, 0.801), abs=0.01)
    assert (hi["xerr_true"], hi["xerr_triv"]) == pytest.approx((0.293, 0.172), abs=0.01)
    assert cell(objective["rows"], obj="phantom", noise=1e-1)["j_true"] == pytest.approx(0.059, abs=0.003) and cell(objective["rows"], obj="phantom", noise=1e-1)["j_triv"] == pytest.approx(0.0244, abs=0.002)
    assert ph["xerr_triv"] < ph["xerr_true"] and sp["xerr_triv"] > sp["xerr_true"]
    text = ev.verdict_objective(objective)[1]
    assert "in 9 von 9 Zellen in allen 20 Strömen" in text and "2,4·10⁻⁴ gegen 4,7·10⁻⁴" in text and "4,2·10⁻⁵ gegen 2,2·10⁻⁴" in text and "9 von 9 Zellen in allen Strömen den Rand des Gitters (0,0100)" in text
    assert "21,9 % gegen 27,5 %" in text and "80,1 % gegen 52,9 %" in text


# ----------------------------------------------------------------------------- Experiment 3

def test_alternation_stays_at_the_start(alternation):
    rows = alternation["rows"]
    assert len(rows) == 21 and len(alternation["param"]) == 9 and len(alternation["l1"]) == 6
    widths = [cell(rows, obj="phantom", start="gauss", start_w=w0)["width"] for w0 in (0.01, 0.02, 0.03, 0.04, 0.06)]
    assert widths == pytest.approx([0.0085, 0.0197, 0.0297, 0.0396, 0.0561], abs=0.0006)
    for obj in C.OBJECTS:
        assert all(abs(cell(rows, obj=obj, start="gauss", start_w=w0)["width"] - w0) <= 0.15 * w0 for w0 in (0.01, 0.02, 0.03, 0.04, 0.06))
        assert [cell(rows, obj=obj, start="gauss", start_w=w0)["share_ok"] for w0 in (0.01, 0.02, 0.03, 0.04, 0.06)] == [0.0, 0.0, 1.0, 0.0, 0.0]
        assert cell(rows, obj=obj, start="delta")["width"] < 1e-9 and cell(rows, obj=obj, start="delta")["kerr"] == pytest.approx(1.81, abs=0.02)
    assert cell(rows, obj="phantom", start="box")["width"] == pytest.approx(0.0747, abs=0.003) and cell(rows, obj="phantom", start="box")["j_ratio"] > 10.0
    assert cell(rows, obj="phantom", start="gauss", start_w=0.03)["xerr"] < cell(rows, obj="phantom", start="gauss", start_w=0.03)["xerr_known"]            # ein richtiger Start ist besser als bekannt
    assert [cell(rows, obj="phantom", start="gauss", start_w=w0)["j_ratio"] for w0 in (0.01, 0.03, 0.06)] == pytest.approx([0.555, 0.676, 2.37], abs=0.05)
    assert cell(rows, obj="spikes", start="delta")["xerr"] == pytest.approx(0.80, abs=0.02) and cell(rows, obj="spikes", start="delta")["j_ratio"] < 0.25


def test_parametric_alternation_is_a_fixed_point(alternation):
    for r in alternation["param"]:
        assert r["width"] == pytest.approx(r["w0"], abs=1e-9)
    assert max(r["moved"] for r in alternation["param"]) <= 0.126 and sum(r["moved"] > 0 for r in alternation["param"]) <= 1
    assert cell(alternation["param"], obj="phantom", w0=0.06)["xerr"] == pytest.approx(2.1, abs=0.15)
    assert cell(alternation["param"], obj="spikes", w0=0.06)["moved"] == pytest.approx(0.125) and cell(alternation["param"], obj="phantom", w0=0.01)["xerr"] == pytest.approx(0.216, abs=0.01)
    pins = {("phantom", 0.01): 0.2163, ("phantom", 0.03): 0.2625, ("phantom", 0.06): 2.095, ("spikes", 0.01): 0.7936, ("spikes", 0.03): 0.5298, ("spikes", 0.06): 7.043, ("hills", 0.03): 0.2729}
    for (obj, w0), value in pins.items():
        assert cell(alternation["param"], obj=obj, w0=w0)["xerr"] == pytest.approx(value, abs=0.002), (obj, w0)


def test_l1_on_spikes_finds_the_kernel_only_from_a_wide_start_at_medium_noise(alternation):
    l1 = alternation["l1"]
    good = cell(l1, noise=1e-2, start="gauss", start_w=0.05)
    assert good["width"] == pytest.approx(0.0292, abs=0.001) and good["kerr"] == pytest.approx(0.0145, abs=0.01) and good["xerr"] == pytest.approx(0.030, abs=0.01) and good["xerr_known"] == pytest.approx(0.0104, abs=0.004)
    assert good["share_ok"] == 1.0 and cell(l1, noise=1e-2, start="delta")["xerr"] == pytest.approx(0.80, abs=0.02) and cell(l1, noise=1e-2, start="gauss", start_w=0.02)["share_ok"] == 0.0
    low = cell(l1, noise=1e-3, start="gauss", start_w=0.05)
    assert low["width"] == pytest.approx(0.0485, abs=0.003) and low["share_ok"] == 0.0 and low["xerr_known"] == pytest.approx(0.001, abs=0.001)
    text = ev.verdict_alternation(alternation)[1]
    assert "0,0085 / 0,0197 / 0,0297 / 0,0396 / 0,0561" in text and "0,0100 / 0,0300 / 0,0600" in text and "Breite 0,0292, Kern-Fehler 1,5 %, Fehler 3,0 % gegen 1,0 %" in text
    assert "aus Delta nicht (Fehler 80,1 %)" in text and "bleibt die Breite aus dem Start 0,05 bei 0,0485" in text


# ----------------------------------------------------------------------------- Experiment 4

def test_evidence_numbers(evidence):
    rows = evidence["rows"]
    assert len(rows) == 6
    pr = [cell(rows, truth="prior", noise=nz) for nz in C.NOISE_CHOICES]
    ph = [cell(rows, truth="phantom", noise=nz) for nz in C.NOISE_CHOICES]
    assert [r["w_evid"] for r in pr] == pytest.approx([0.03, 0.03, 0.03], abs=0.001) and [r["w_evid"] for r in ph] == pytest.approx([0.034, 0.036, 0.038], abs=0.001)
    assert all(r["w_prof"] == pytest.approx(0.01) for r in rows)
    assert [r["err_evid"] for r in pr] == pytest.approx([0.142, 0.185, 0.256], abs=0.01) and [r["err_known"] for r in pr] == pytest.approx([0.139, 0.177, 0.256], abs=0.01)
    assert [r["err_evid"] for r in ph] == pytest.approx([0.198, 0.203, 0.238], abs=0.01) and [r["err_known"] for r in ph] == pytest.approx([0.094, 0.121, 0.191], abs=0.01)
    assert all(r["q25"] <= r["w_evid"] <= r["q75"] for r in rows) and all(r["err_evid"] >= r["err_known"] - 0.01 for r in rows)
    assert all(abs(r["w_evid"] - 0.03) < abs(q["w_evid"] - 0.03) + 1e-9 for r, q in zip(pr, ph))
    text = ev.verdict_evidence(evidence)[1]
    assert "Median 0,0300 / 0,0300 / 0,0300" in text and "14,2 % / 18,5 % / 25,6 % gegen 13,9 % / 17,7 % / 25,6 %" in text and "immer 0,0100" in text
    assert "0,0340 / 0,0360 / 0,0380" in text and "19,8 % / 20,3 % / 23,8 % gegen 9,4 % / 12,1 % / 19,1 %" in text
