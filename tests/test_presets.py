"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen und die dokumentierten Ausgänge."""

import pytest

import bld_constants as C
import bld_evaluation as ev
import bld_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds_and_on_the_choice_lists():
    for name, p in C.PRESETS.items():
        assert p["obj"] in C.OBJECTS and p["noise"] in C.NOISE_CHOICES and p["kernel"] in C.KERNELS and p["method"] in C.METHODS and p["prior"] in C.PRIORS, name
        assert p["start"] in C.STARTS and p["start_w"] in C.START_W_CHOICES and p["rounds"] in C.ROUNDS_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name
        assert C.W_MIN <= p["w"] <= C.W_MAX and (p["w"] - C.W_MIN) / C.W_STEP == pytest.approx(round((p["w"] - C.W_MIN) / C.W_STEP)), name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_settings_fields_follow_the_preset_keys():
    import dataclasses

    assert [f.name for f in dataclasses.fields(ev.Settings)] == list(P.PRESET_KEYS)


def test_the_presets_show_the_documented_outcomes():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    std, delta, wide, fit, l1, prof, evid = (res[k] for k in C.PRESETS)
    assert ev.verdict_instance(std)[0] == "warning" and std.w_hat == pytest.approx(0.0396, abs=0.0005) and std.kerr == pytest.approx(0.233, abs=0.01) and std.score_hat <= std.score_true
    assert delta.w_hat == 0.0 and delta.xerr == pytest.approx(0.218, abs=0.01) and delta.xerr_known == pytest.approx(0.293, abs=0.01) and delta.score_hat < delta.score_true
    assert wide.w_hat == pytest.approx(0.0559, abs=0.001) and wide.xerr == pytest.approx(1.19, abs=0.05) and wide.score_hat / wide.score_true == pytest.approx(2.77, abs=0.1)
    assert fit.kerr_start < 1e-9 and fit.xerr == pytest.approx(0.127, abs=0.01) and ev.verdict_instance(fit)[0] == "warning"
    assert ev.verdict_instance(l1)[0] == "success" and l1.kerr == pytest.approx(0.017, abs=0.005) and l1.kerr_start == pytest.approx(0.478, abs=0.01)
    assert prof.w_hat == pytest.approx(0.01) and prof.at_edge and evid.w_hat == pytest.approx(0.036) and evid.kerr == pytest.approx(0.15, abs=0.01)


def test_experiment_constants_are_consistent():
    assert C.EXP_REPS >= 20 and C.ALT_REPS >= 8 and C.EVID_REPS >= 30 and C.N == 48 and C.KERNEL_HALF == 6 and set(C.NOISE_CHOICES) == {1e-3, 1e-2, 1e-1}
    assert C.DEFAULT_START_W in C.START_W_CHOICES and C.DEFAULT_ROUNDS in C.ROUNDS_CHOICES and C.DEFAULT_NOISE in C.NOISE_CHOICES and C.W_MIN <= C.DEFAULT_W <= C.W_MAX
    assert all(sw in C.START_W_CHOICES or sw in (0.05,) for _, sw in C.ALT_STARTS + C.L1_STARTS if sw is not None) and C.KERNEL_OK == 0.2 and C.WIDTH_OK == 0.1
    assert max(C.AMBIGUITY_DELTAS) < min(b for _, b, _ in __import__("inverse_generator").BUMPS) ** 2
    assert set(C.SHOW_RANGE) == set(C.OBJECTS) and set(C.OBJECT_LABELS) == set(C.OBJECT_SHORT) == set(C.OBJECTS)


def test_label_dictionaries_cover_their_choices():
    assert set(C.KERNEL_LABELS) == set(C.KERNEL_SHORT) == set(C.KERNELS) and set(C.METHOD_LABELS) == set(C.METHOD_SHORT) == set(C.METHODS)
    assert set(C.PRIOR_LABELS) == set(C.PRIOR_SHORT) == set(C.PRIORS) and set(C.START_LABELS) == set(C.START_SHORT) == set(C.STARTS)
