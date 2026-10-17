"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden
Regler an seine Grenzen und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei st.plotly_chart ohne eindeutiges key=)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
AUTOPLAY_MARKER = "▶"


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_option_of_every_select_slider_does_not_raise():
    import bld_constants as C

    for key, options in (("noise_select", C.NOISE_CHOICES), ("startw_select", C.START_W_CHOICES), ("rounds_select", C.ROUNDS_CHOICES)):
        for value in options:
            at = _fresh_app()
            at.select_slider(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_every_object_method_kernel_prior_and_start_does_not_raise():
    import bld_constants as C

    for key, options in (("obj_select", C.OBJECTS), ("method_select", C.METHODS), ("kernel_select", C.KERNELS), ("prior_select", C.PRIORS), ("start_select", C.STARTS)):
        for value in options:
            at = _fresh_app()
            at.selectbox(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_the_corners_of_the_grid_do_not_raise():
    at = _fresh_app()
    at.slider(key="w_slider").set_value(0.04)
    at.select_slider(key="noise_select").set_value(1e-1)
    at.selectbox(key="obj_select").set_value("spikes")
    at.selectbox(key="prior_select").set_value("l1")
    at.select_slider(key="rounds_select").set_value(1)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    at = _fresh_app()
    at.slider(key="w_slider").set_value(0.02)
    at.select_slider(key="noise_select").set_value(1e-3)
    at.selectbox(key="obj_select").set_value("hills")
    at.selectbox(key="start_select").set_value("box")
    at.selectbox(key="kernel_select").set_value("gauss")
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_every_preset_is_applied_to_the_widgets():
    import bld_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.selectbox(key="obj_select").value == preset["obj"]
        assert at.select_slider(key="noise_select").value == preset["noise"]
        assert abs(at.slider(key="w_slider").value - preset["w"]) < 1e-9
        assert at.selectbox(key="kernel_select").value == preset["kernel"]
        assert at.selectbox(key="method_select").value == preset["method"]
        assert at.selectbox(key="prior_select").value == preset["prior"]
        assert at.selectbox(key="start_select").value == preset["start"]
        assert at.select_slider(key="startw_select").value == preset["start_w"]
        assert at.select_slider(key="rounds_select").value == preset["rounds"]
        assert at.number_input(key="seed_input").value == preset["seed"]


def test_the_standard_instance_shows_a_warning_verdict_and_four_experiments():
    at = _fresh_app()
    assert any("das Ziel zeichnet die Wahrheit nicht aus" in w.value for w in at.warning)
    headers = [s.value for s in at.subheader]
    assert sum("🔬" in h for h in headers) == 4


def test_the_l1_spikes_preset_shows_the_success_verdict():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    at = _fresh_app()
    at.button[labels["Spitzen mit L1"]].click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Der Kern ist gefunden" in s.value for s in at.success)


def test_the_history_tab_explains_itself_for_profile_and_evidence():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    at = _fresh_app()
    at.button[labels["Evidenz"]].click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("keine Runden" in i.value for i in at.info)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_pdf_of_the_evidence_and_the_l1_instance_is_created():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name in ("Evidenz", "Spitzen mit L1", "Profil über die Breite"):
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        at.button(key="pdf_make").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.session_state["pdf_made"][1][:5] == b"%PDF-"


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["obj"] = "kreis"
    at.query_params["r"] = "0.5"
    at.query_params["w"] = "-3"
    at.query_params["kern"] = "box"
    at.query_params["meth"] = "evidence"
    at.query_params["prior"] = "l2"
    at.query_params["start"] = "delta"
    at.query_params["sw"] = "0.5"
    at.query_params["rounds"] = "60"
    at.query_params["seed"] = "99999999"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="obj_select").value == "phantom"
    assert at.select_slider(key="noise_select").value == 1e-2
    assert abs(at.slider(key="w_slider").value - 0.02) < 1e-9
    assert at.selectbox(key="kernel_select").value == "nonpar"
    assert at.selectbox(key="method_select").value == "evidence"
    assert at.selectbox(key="prior_select").value == "tikhonov"
    assert at.selectbox(key="start_select").value == "delta"
    assert at.select_slider(key="startw_select").value == 0.04
    assert at.select_slider(key="rounds_select").value == 60
    assert at.number_input(key="seed_input").value == 999999


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.selectbox(key="start_select").set_value("delta").run(timeout=TIMEOUT)
    assert at.query_params["start"] == ["delta"] or at.query_params["start"] == "delta"
