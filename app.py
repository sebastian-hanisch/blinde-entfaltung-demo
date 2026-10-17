"""Blinde Entfaltung: Kern und Signal gemeinsam schätzen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwölftes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 1 bis 8 (Phantom, Gauß-Unschärfe) ist der Kern jetzt unbekannt: Verfahren sind die Alternierung
(x und Kern abwechselnd), das Profil über die Breite (x wegminimiert) und die Evidenz (x wegmarginalisiert); gefragt wird, ob das Ziel den wahren Kern auszeichnet.
Kern: bld_model.py, Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import bld_constants as C
from bld_evaluation import (
    Settings,
    alternation_experiment,
    ambiguity_experiment,
    analyse,
    de,
    err_pct,
    evidence_experiment,
    noise_label,
    objective_experiment,
    sci,
    verdict_alternation,
    verdict_ambiguity,
    verdict_evidence,
    verdict_instance,
    verdict_objective,
    width_label,
)
from bld_pdf_export import build_pdf
from bld_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from bld_visualization import (
    build_ambiguity_chart,
    build_cost_chart,
    build_evidence_chart,
    build_evidence_exp_chart,
    build_kernel_chart,
    build_kernel_step_chart,
    build_objective_chart,
    build_profile_chart,
    build_progress_chart,
    build_recon_chart,
    build_start_chart,
    build_table,
    style_table,
)

st.set_page_config(page_title="Blinde Entfaltung – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _ambiguity():
    return ambiguity_experiment()


@st.cache_data(show_spinner=SPINNER)
def _objective():
    return objective_experiment()


@st.cache_data(show_spinner=SPINNER)
def _alternation():
    return alternation_experiment()


@st.cache_data(show_spinner=SPINNER)
def _evidence():
    return evidence_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def _gap(v):
    return "< 10⁻¹²" if v < 1e-12 else sci(v)


st.title("🔍 Blinde Entfaltung: Kern und Signal gemeinsam schätzen")
st.markdown(
    """
**Bei unbekanntem Kern ist die Zielfunktion der falsche Richter, und das Alternieren bleibt, wo es startet.** Dies ist das zwölfte Stück der Linie **Inverse Probleme und Regularisierung** und baut auf
**Tikhonov**, **Bayes-Sicht: MAP** und **Auflösung und Messorte** auf: dieselbe Welt (Phantom, Gauß-Unschärfe), aber der **Kern ist unbekannt**. Gefragt wird, ob man ihn **zusammen mit dem Signal** aus den
Daten schätzen kann (Alternieren; das Profil über die Breite; die Evidenz), **warum nicht** (das Paar „kein Blur“ hat das kleinere Ziel; ein Teil der Aufgabe ist exakt mehrdeutig) und **was hilft**
(marginalisieren statt minimieren; ein dünn besetztes Signal). Die Welt ist simuliert; die Wahrheit ist bekannt und dient nur als Maßstab.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Wie blinde Entfaltung funktioniert", expanded=True):
    st.markdown(
        r"""
1. **Modell.** Daten $b = \kappa * x + e$: ein Kern $\kappa$ (hier 13 Werte, $\kappa \ge 0$, Summe 1) faltet das Signal $x$. Die Daten sind **linear in $x$ und linear in $\kappa$**, aber nicht in beiden zusammen.
2. **Alternieren.** Abwechselnd $x$ bei festem Kern (Tikhonov oder L1, Stück 2 und 6) und den Kern bei festem $x$ (nicht negative kleinste Quadrate mit Summe 1) bestimmen; die Zielfunktion $J$ fällt in jeder Runde.
3. **Das Trivialpaar.** $\kappa = \delta$ („kein Blur“) und $x$ = die geglättete Messung erklärt die Daten fast fehlerfrei, ohne dass $x$ stark bestraft wird: sein $J$ ist **kleiner** als das des wahren Paares.
4. **Mehrdeutigkeit.** Skala und Versatz ($c\,x$ mit $\kappa/c$; $x$ verschoben, $\kappa$ zurück) sind exakte Symmetrien. Für Gauß-Hügel erklärt jedes $(b', w')$ mit $b'^2 + w'^2 = b^2 + w^2$ die Daten gleich gut.
5. **Marginalisieren.** Die Evidenz $p(b \mid w) = \int p(b \mid x, w)\,p(x)\,dx$ mittelt über $x$, statt es wegzuminimieren; sie findet die Breite, wenn der Prior stimmt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt**")
    obj = st.selectbox("Welches Signal?", C.OBJECTS, key="obj_select", format_func=lambda v: C.OBJECT_LABELS[v],
                       help="Das Signal entscheidet, ob der Kern aus x bestimmbar ist: Spitzen sind die Antwort auf einen Impuls, die Hügel allein sind es kaum.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens relativ zum Effektivwert der exakten Daten.")
    w = st.slider("Wahre Breite der Unschärfe w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", format="%.3f", help="Breite des wahren Gauß-Kerns als Anteil der Länge 1.")
    st.markdown("**Verfahren**")
    method = st.selectbox("Welches Verfahren?", C.METHODS, key="method_select", format_func=lambda v: C.METHOD_LABELS[v],
                          help="Alternieren schätzt den Kern frei oder als Gauß; Profil und Evidenz schätzen die Breite eines Gauß-Kerns auf einem Gitter von 0,010 bis 0,060.")
    kernel = st.selectbox("Welches Kernmodell (Alternieren)?", C.KERNELS, key="kernel_select", format_func=lambda v: C.KERNEL_LABELS[v],
                          help="Gilt nur beim Alternieren. Beim Gauß-Kern wird nur die Breite geschätzt (Gitter 0,005 bis 0,080), Start ist dann die Startbreite.")
    prior = st.selectbox("Welcher x-Prior?", C.PRIORS, key="prior_select", format_func=lambda v: C.PRIOR_LABELS[v],
                         help="Tikhonov mit λ = σ/τ oder L1 mit dem exakten Aktivmengen-Löser aus Stück 6. Die Evidenz benutzt immer den exponentiellen Gauß-Prior aus Stück 8.")
    st.markdown("**Start**")
    start = st.selectbox("Welcher Startkern?", C.STARTS, key="start_select", format_func=lambda v: C.START_LABELS[v], help="Womit die Alternierung beginnt (nur das freie Kernmodell).")
    start_w = st.select_slider("Startbreite des Gauß-Kerns", options=C.START_W_CHOICES, key="startw_select", format_func=lambda v: de(v, 2),
                               help="Breite des Gauß-Starts; beim Gauß-Kernmodell der Startwert der Breite.")
    rounds = st.select_slider("Runden", options=C.ROUNDS_CHOICES, key="rounds_select", help="Zahl der Alternierungs-Runden (x-Schritt und Kernschritt).")
    seed = st.number_input("Zufalls-Seed des Rauschens", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"obj_select": obj, "noise_select": float(noise), "w_slider": round(float(w), 3), "kernel_select": kernel, "method_select": method, "prior_select": prior, "start_select": start,
                   "startw_select": float(start_w), "rounds_select": int(rounds), "seed_input": int(seed)})

settings = Settings(obj, float(noise), round(float(w), 3), kernel, method, prior, start, float(start_w), int(rounds), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kernbreite", width_label(a.w_hat), delta=f"wahr {width_label(a.w_true)}", delta_color="off",
          help="Standardabweichung des geschätzten Kerns (Anteil der Länge 1); beim Profil und bei der Evidenz die gewählte Breite des Gauß-Kerns.")
c2.metric("Kern-Fehler", err_pct(a.kerr), delta=f"Start {err_pct(a.kerr_start) if a.kerr_start >= C.NULL_ERR else 'null'}", delta_color="off",
          help="Abstand des geschätzten zum wahren Kern (2-Norm, relativ zur Norm des wahren Kerns); darunter der Fehler des Startkerns.")
c3.metric("Fehler der Rekonstruktion", err_pct(a.xerr), delta=f"Kern bekannt {err_pct(a.xerr_known)}", delta_color="off",
          help="Abstand der blinden Rekonstruktion zur Wahrheit relativ zur Signalnorm; darunter derselbe Fehler, wenn der wahre Kern bekannt wäre.")
if a.score_kind == "evidence":
    c4.metric("Log-Evidenz", de(a.score_hat, 1), delta=f"wahre Breite {de(a.score_true, 1)}", delta_color="off", help="log p(b | w′) bis auf eine Konstante, bei ŵ und bei der wahren Breite; größer ist besser.")
else:
    c4.metric("Zielfunktion J", sci(a.score_hat), delta=f"wahres Paar {sci(a.score_true)}", delta_color="off",
              help="J = |A x − b|² + Strafe auf x beim Ergebnis des Verfahrens und beim wahren Paar (wahrer Kern, bestes x); kleiner ist besser.")
st.caption(f"Trivialpaar (Kern = Delta, x = geglättete Messung): Fehler {err_pct(a.xerr_triv)}" + ("" if a.score_kind == "evidence" else f", J = {sci(a.score_triv)}")
           + f"; Profil wählt die Breite {width_label(a.w_prof)}, Evidenz {width_label(a.w_evid)} (Gitter 0,010 bis 0,060); wahres Rauschen σ = {sci(a.sigma)}.")

tab_ker, tab_rec, tab_hist, tab_prof, tab_tab = st.tabs(["Kern", "Rekonstruktion", "Verlauf", "Profil und Evidenz", "Tabelle"])
with tab_ker:
    st.plotly_chart(build_kernel_chart(a), width="stretch", key="kernel_chart")
    st.caption("Grün der wahre Kern, grau gestrichelt der Startkern, blau die Schätzung (13 Werte, Summe 1).")
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, orange gestrichelt die Rekonstruktion bei bekanntem Kern, blau das Ergebnis des Verfahrens, rot gepunktet das Trivialpaar. Die Achse ist fest.")
with tab_hist:
    if a.js:
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(build_cost_chart(a), width="stretch", key="cost_chart")
        with ch2:
            st.plotly_chart(build_progress_chart(a), width="stretch", key="progress_chart")
        st.caption("Links die Zielfunktion über die Runden (sie fällt nie), mit den Werten des wahren Paares und des Trivialpaares; rechts Kernbreite (in Vielfachen der wahren), Kern-Fehler und Fehler der Rekonstruktion.")
    else:
        st.info("Profil und Evidenz haben keine Runden: sie werten die Breiten des Gitters einzeln aus. Der Verlauf gehört zum Alternieren.")
with tab_prof:
    cp1, cp2 = st.columns(2)
    with cp1:
        st.plotly_chart(build_profile_chart(a), width="stretch", key="profile_chart")
    with cp2:
        st.plotly_chart(build_evidence_chart(a), width="stretch", key="evidence_chart")
    st.caption("Links die Kosten des Profils (x wegminimiert, mit dem gewählten x-Prior), rechts die Evidenz (x wegmarginalisiert, exponentieller Prior). Das Profil fällt zum schmalen Rand; die Evidenz hat ein Maximum.")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=560)
    st.caption("Die ersten Punkte der Instanz.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="blinde-entfaltung.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Findet das Verfahren den Kern?")
verdict(verdict_instance(a))
st.caption("Das Urteil vergleicht den Kern-Fehler mit der Schwelle von 20 %, den Startkern mit der Wahrheit (ein richtiger Start zählt nicht als Fund) und die Zielfunktion mit der des wahren Paares und des Trivialpaares.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Was die Daten gar nicht festlegen")
res1 = _ambiguity()
st.caption(f"Links: Datenunterschied des zweiten Paares (b′² = b² − δ, w′² = w² + δ, h′ = h b / b′) zur Wahrheit mit w = {de(res1['w'], 2)}, rein analytisch. Rechts: Kern-Fehler des Kernschritts bei "
           f"bekanntem x, Median über {C.EXP_REPS} Ströme.")
ce1, ce2 = st.columns(2)
with ce1:
    st.plotly_chart(build_ambiguity_chart(res1), width="stretch", key="ambiguity_chart")
with ce2:
    st.plotly_chart(build_kernel_step_chart(res1), width="stretch", key="kernel_step_chart")
verdict(verdict_ambiguity(res1))
st.dataframe([{"δ": de(r["delta"], 4), "neue Breite w′": de(r["w_new"], 4), "Datenunterschied nur Hügel": _gap(r["gap_hills"]), "Datenunterschied mit Kasten": f"{de(100.0 * r['gap_box'], 1)} %"}
              for r in res1["rows"]], hide_index=True, width="stretch")
st.dataframe([{"Signal": C.OBJECT_SHORT[r["obj"]], "Skala (3 x, κ/3)": _gap(r["scale"]), "Versatz im Innern": sci(r["shift"])} for r in res1["sym"]], hide_index=True, width="stretch")
st.dataframe([{"Signal": C.OBJECT_SHORT[r["obj"]], "Rauschen": noise_label(r["noise"]), "Kern-Fehler (Median)": err_pct(r["median"]), "Kern-Fehler (Maximum)": err_pct(r["max"])}
              for r in res1["steps"]], hide_index=True, width="stretch", height=360)
st.caption("Hügel allein (ohne scharfe Kanten) legen nur die Summe b² + w² fest; der Kasten bricht das. Der Kernschritt ist nur bei informativem Signal gut gestellt: Spitzen sind die Antwort auf einen Impuls, "
           "die breiten Hügel sagen über den Kern fast nichts. Skala und Versatz sind exakte Symmetrien; die Normierung Summe κ = 1 legt die Skala fest.")

st.subheader("🔬 Experiment 2: Das Ziel als Richter")
res2 = _objective()
st.caption(f"Wahres Paar (wahrer Kern, bestes x) gegen das Trivialpaar (Kern = Delta), Tikhonov, w = {de(res2['w'], 2)}, Median über {res2['reps']} Ströme; unten das Profil über die Breite.")
st.plotly_chart(build_objective_chart(res2), width="stretch", key="objective_chart")
verdict(verdict_objective(res2))
st.dataframe([{"Signal": C.OBJECT_SHORT[r["obj"]], "Rauschen": noise_label(r["noise"]), "J wahres Paar": sci(r["j_true"]), "J Trivialpaar": sci(r["j_triv"]),
               "Trivialpaar kleiner": f"{de(100.0 * r['share_triv_smaller'], 0)} %", "Fehler wahres Paar": err_pct(r["xerr_true"]), "Fehler Trivialpaar": err_pct(r["xerr_triv"]),
               "Profil wählt": de(r["w_prof"], 3), "am Rand": f"{de(100.0 * r['share_edge'], 0)} %"} for r in res2["rows"]], hide_index=True, width="stretch", height=360)
st.caption("Das Trivialpaar ist kein Fehler des Verfahrens, sondern ein gutes Ergebnis für das Ziel: bei verrauschten Daten ist „nicht entfalten“ die vernünftige Antwort (Stück 4). Das Ziel kann deshalb den wahren Kern "
           "nicht auszeichnen; bei den Spitzen stimmt aber auch das Trivialpaar nicht (große Fehler), nur das Ziel sieht den Unterschied nicht.")

st.subheader("🔬 Experiment 3: Alternieren bleibt beim Start")
res3 = _alternation()
st.caption(f"Alternierung (Tikhonov, {C.DEFAULT_ROUNDS} Runden, Rauschen 1 %), Median über {C.ALT_REPS} Ströme, w = {de(res3['w'], 2)}: Breite des Ergebnisses über die Startbreite; darunter die parametrische "
           "Alternierung (eine Breite) und Spitzen mit L1-Prior.")
st.plotly_chart(build_start_chart(res3), width="stretch", key="start_chart")
verdict(verdict_alternation(res3))
st.dataframe([{"Signal": C.OBJECT_SHORT[r["obj"]], "Start": C.START_SHORT[r["start"]] + ("" if r["start_w"] is None else f" {de(r['start_w'], 2)}"), "Breite des Ergebnisses": width_label(r["width"]),
               "Kern-Fehler": err_pct(r["kerr"]), "Fehler der Rekonstruktion": err_pct(r["xerr"]), "bei bekanntem Kern": err_pct(r["xerr_known"]), "J / J wahr": de(r["j_ratio"], 2),
               "Kern gefunden": f"{de(100.0 * r['share_ok'], 0)} %"} for r in res3["rows"]], hide_index=True, width="stretch", height=420)
st.dataframe([{"Signal": C.OBJECT_SHORT[r["obj"]], "Startbreite": de(r["w0"], 2), "Breite nach 50 Runden": width_label(r["width"]), "Anteil der Ströme, in denen sie sich bewegt": f"{de(100.0 * r['moved'], 0)} %",
               "Fehler der Rekonstruktion": err_pct(r["xerr"])} for r in res3["param"]], hide_index=True, width="stretch", height=360)
st.dataframe([{"Rauschen": noise_label(r["noise"]), "Start": C.START_SHORT[r["start"]] + ("" if r["start_w"] is None else f" {de(r['start_w'], 2)}"), "Breite des Ergebnisses": width_label(r["width"]),
               "Kern-Fehler": err_pct(r["kerr"]), "Fehler der Rekonstruktion": err_pct(r["xerr"]), "bei bekanntem Kern": err_pct(r["xerr_known"]), "Kern gefunden": f"{de(100.0 * r['share_ok'], 0)} %"}
              for r in res3["l1"]], hide_index=True, width="stretch")
st.caption("Die Alternierung folgt dem Ziel nur langsam und endet nahe dem Start; ein schmalerer Start gibt ein kleineres J. Bei Spitzen mit L1-Prior gelingt der Kern aus einem breiten Start und mittlerem "
           "Rauschen; bei 0,1 % Rauschen nicht. Nicht gemessen: Mehrfach-Starts, eine Strafe auf die Kernglätte und Variationsverfahren.")

st.subheader("🔬 Experiment 4: Marginalisieren statt Minimieren")
res4 = _evidence()
st.caption(f"Die Breite des Gauß-Kerns schätzen, Median und Quartile über {res4['reps']} Ströme je Rauschstufe: Evidenz (x wegmarginalisiert) und Profil (x wegminimiert), wenn die Wahrheit eine Prior-Ziehung ist "
           "(das Modell stimmt) und wenn sie das Phantom ist.")
st.plotly_chart(build_evidence_exp_chart(res4), width="stretch", key="evidence_exp_chart")
verdict(verdict_evidence(res4))
st.dataframe([{"Wahrheit": "Prior-Ziehung" if r["truth"] == "prior" else "Phantom", "Rauschen": noise_label(r["noise"]), "Evidenz: Breite (Median)": width_label(r["w_evid"]),
               "Quartile": f"{width_label(r['q25'])} bis {width_label(r['q75'])}", "Profil: Breite (Median)": width_label(r["w_prof"]), "Fehler mit ŵ": err_pct(r["err_evid"]),
               "Fehler mit wahrer Breite": err_pct(r["err_known"])} for r in res4["rows"]], hide_index=True, width="stretch")
st.caption("Die Evidenz bestraft zu schmale Kerne selbst: sie enthält die Determinante von A P Aᵀ + σ² I. Sie ist aber nur so gut wie das Modell: das Phantom (Kasten, Kanten) ist keine Ziehung aus dem exponentiellen "
           "Prior, und die Breite ist verzerrt. Gelernte Priors (Stück 13 bis 15) liefern genau die Annahme über x, die hier fehlt.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
ph2 = next(r for r in res2["rows"] if r["obj"] == "phantom" and r["noise"] == 1e-2)
ph4 = next(r for r in res4["rows"] if r["truth"] == "phantom" and r["noise"] == 1e-2)
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Ziel zeichnet das wahre Paar aus** | Beim Phantom und Rauschen 1 % ist J des Trivialpaares {sci(ph2['j_triv'])} gegen {sci(ph2['j_true'])} beim wahren Paar. | Stück 16 (Verfahrensvergleich) |
| **Die Alternierung findet den Kern** | Sie bleibt bei der Startbreite; nur ein dünn besetztes Signal mit L1-Prior und breitem Start gelingt. | gelernte Priors (Stück 13 bis 15) |
| **Die Evidenz ist verlässlich** | Beim Phantom ist die Breite {width_label(ph4['w_evid'])} statt {width_label(res4['w'])}: sie ist so gut wie der Prior. | – |
| **Ein Kern, ein Operator** | Ein Gauß-Kern mit 13 Werten, Faltung ohne Randeffekte (Nullränder), eine Dimension; die Zahlen gelten für dieses Beispiel (n = {C.N}). | – |
| **Rauschen weiß und bekannt** | σ ist bekannt und wird nicht mitgeschätzt; farbiges Rauschen ist nicht modelliert. | – |
| **Ein Start, eine Strafe** | Keine Mehrfach-Starts, keine Strafe auf die Kernglätte, keine Variationsverfahren (Levin und andere). | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $b = A(\kappa)\,x + e$, $e \sim \mathcal N(0, \sigma^2 I)$, $A(\kappa)_{ij} = \kappa_{i-j}$ für $|i-j| \le m$ (Nullränder, $m = 6$); $A(\kappa)\,x = \sum_d \kappa_d S_d x$ mit der Verschiebung $S_d$.

**Zielfunktion.** $J(\kappa, x) = \|A(\kappa)x - b\|^2 + \lambda^2\|x\|^2$ (Tikhonov, $\lambda = \sigma/\tau$, $\tau = 0{,}5$) oder $\|A(\kappa)x - b\|^2 + 2\mu\|x\|_1$ (L1).

**Alternieren.** $x_r = \arg\min_x J(\kappa_r, x)$, $\kappa_{r+1} = \arg\min_{\kappa \ge 0,\ \sum\kappa = 1} \|\sum_d \kappa_d S_d x_r - b\|^2$ (nicht negative kleinste Quadrate, Lawson–Hanson, die Summe über eine schwere Zeile
erzwungen). $J(\kappa_r, x_r)$ fällt in jeder Runde.

**Trivialpaar.** $\kappa = \delta$, $x = b/(1+\lambda^2)$ (Tikhonov): $J = \lambda^2\|b\|^2/(1+\lambda^2)$. Das wahre Paar hat $J = \|A x_\lambda - b\|^2 + \lambda^2\|x_\lambda\|^2$ mit Datenfehler und Strafe; im Beispiel der
Mini-Instanz (Kern $(\tfrac14,\tfrac12,\tfrac14)$, $x = (1,2,1)$) liegt das Trivialpaar bei $\lambda = 0{,}1$ mit $0{,}0421$ unter $0{,}0587$.

**Mehrdeutigkeit.** Skala: $(c\,x,\ \kappa/c)$; Versatz: $(S_{-1}x,\ S_{1}\kappa)$. Gauß-Hügel (Mitte $c$, Breite $b$, Höhe $h$) werden zu Hügeln der Breite $\sqrt{b^2 + w^2}$ und der Höhe $h\,b/\sqrt{b^2+w^2}$; das Paar
$(b', w', h') = (\sqrt{b^2 - \delta},\ \sqrt{w^2 + \delta},\ h\,b/b')$ gibt exakt dieselben Daten. Ein Kasten (scharfe Kanten) bricht das.

**Evidenz.** $\log p(b \mid w) = -\tfrac12 b^\top (K + \sigma^2 I)^{-1} b - \tfrac12\log\det(K + \sigma^2 I) + \text{const}$ mit $K = A(w)\,P\,A(w)^\top$ und dem exponentiellen Prior $P$ aus Stück 8. Für ein Skalar
($n = 1$) liegt das Maximum bei $a^2 p = b^2 - \sigma^2$.

**Literatur.** Das Beispiel ist erfunden. Blinde Entfaltung und das Trivialpaar: A. Levin, Y. Weiss, F. Durand, W. T. Freeman (CVPR 2009 und 2011); alternierende Minimierung mit Sparsity-Priors: T. F. Chan und C. K. Wong
(1998), D. Krishnan und R. Fergus (2009); Mehrdeutigkeit der Faltung: Ayers und Dainty (1988); Nicht negative kleinste Quadrate: C. L. Lawson und R. J. Hanson (1974); Evidenz: D. MacKay (alles aus dem Gedächtnis,
nicht geprüft; die Zahlen hier sind gemessen, nicht der Literatur entnommen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
