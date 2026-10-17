"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import bld_constants as C
from bld_evaluation import Analysis, de, noise_label, sci, width_label


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _ref_line(fig: go.Figure, value: float, line: dict, text: str, position: str, axis: str) -> None:
    """Referenzlinie mit Beschriftung auf einer logarithmischen Achse: die Linie liegt auf dem Datenwert, die Beschriftung braucht log10 des Werts
    (add_hline und add_vline setzen beide auf denselben Wert, die Beschriftung sitzt dann falsch und dehnt die Achse)."""
    if not value > 0.0:
        return
    pos = math.log10(value)
    top, left = "top" in position, "left" in position
    if axis == "y":
        fig.add_hline(y=value, line=line)
        fig.add_annotation(xref="paper", yref="y", x=0 if left else 1, y=pos, text=text, showarrow=False, xanchor="left" if left else "right", yanchor="bottom" if top else "top")
    else:
        fig.add_vline(x=value, line=line)
        fig.add_annotation(xref="x", yref="paper", x=pos, y=1 if top else 0, text=text, showarrow=False, xanchor="right" if left else "left", yanchor="top" if top else "bottom")


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, C.N)
    return pd.DataFrame({"Punkt": [f"{i + 1}" for i in range(k)], "Ort s": a.s[:k], "Daten b": a.b[:k], "Wahrheit x": a.x[:k], "bekannter Kern": a.x_known[:k], "blind": a.x_hat[:k],
                         "Trivialpaar": a.x_triv[:k]})


def style_table(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 4)) for c in df.columns if c != "Punkt"})


def build_kernel_chart(a: Analysis) -> go.Figure:
    """Der wahre Kern, der Startkern und der geschätzte Kern über den Versatz (in Zellen)."""
    m = C.KERNEL_HALF
    d = list(range(-m, m + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=a.kappa_start, mode="lines+markers", name="Start", line=dict(color=C.COLOR_START, width=2, dash="dash"), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=d, y=a.kappa_true, mode="lines+markers", name="Wahrheit", line=dict(color=C.COLOR_TRUTH, width=3), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=d, y=a.kappa_hat, mode="lines+markers", name=f"Schätzung (Breite {width_label(a.w_hat)})", line=dict(color=C.COLOR_BLIND, width=3), marker=dict(size=6),
                             hovertemplate="Versatz %{x}<br>κ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Versatz (Zellen)", tickvals=d)
    fig.update_yaxes(title="Kern κ (Summe 1)", range=[-0.02, 1.02])
    return _fixed(fig, 340)


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, Rekonstruktion bei bekanntem Kern, blinde Rekonstruktion und Trivialpaar; die Achse ist fest (je Objekt)."""
    lo, hi = C.SHOW_RANGE[a.settings.obj]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_known, mode="lines", name="Kern bekannt", line=dict(color=C.COLOR_KNOWN, width=2, dash="dash"), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_hat, mode="lines", name="blind (Verfahren)", line=dict(color=C.COLOR_BLIND, width=2.5), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_triv, mode="lines", name="Trivialpaar (Kern = Delta)", line=dict(color=C.COLOR_TRIVIAL, width=1.5, dash="dot"), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    return _fixed(fig, 400)


def build_cost_chart(a: Analysis) -> go.Figure:
    """Zielfunktion J über die Runden (logarithmisch), mit J des wahren Paares und des Trivialpaares als Linien."""
    r = list(range(len(a.js)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=a.js, mode="lines+markers", name="J des Verfahrens", line=dict(color=C.COLOR_BLIND, width=3), marker=dict(size=4)))
    _ref_line(fig, a.score_true, dict(color=C.COLOR_TRUTH, width=2, dash="dash"), f"wahres Paar {sci(a.score_true)}", "top left", "y")
    _ref_line(fig, a.score_triv, dict(color=C.COLOR_TRIVIAL, width=2, dash="dot"), f"Trivialpaar {sci(a.score_triv)}", "bottom left", "y")
    fig.update_xaxes(title="Runde")
    fig.update_yaxes(title="Zielfunktion J", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_progress_chart(a: Analysis) -> go.Figure:
    """Kernbreite, Kern-Fehler und Fehler der Rekonstruktion über die Runden."""
    r = list(range(len(a.widths_hist)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=[v / a.w_true for v in a.widths_hist], mode="lines+markers", name="Kernbreite / wahre Breite", line=dict(color=C.COLOR_BLIND, width=3), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=r, y=a.kerrs_hist, mode="lines", name="Kern-Fehler", line=dict(color=C.COLOR_KNOWN, width=2)))
    fig.add_trace(go.Scatter(x=r, y=a.xerrs_hist, mode="lines", name="Fehler der Rekonstruktion", line=dict(color=C.COLOR_TRIVIAL, width=2)))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_TRUTH, width=1.5, dash="dash"))
    fig.update_xaxes(title="Runde")
    fig.update_yaxes(title="relativ", range=[0, 2.5])
    return _fixed(fig, 340)


def build_profile_chart(a: Analysis) -> go.Figure:
    """Kosten des Profils (x wegminimiert) über die Breite w′, mit der wahren Breite und dem Minimum."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.grid_w, y=a.prof, mode="lines+markers", name="Profil J(w′)", line=dict(color=C.COLOR_BLIND, width=3), marker=dict(size=5), hovertemplate="w′ %{x:.3f}<br>J %{y:.2e}<extra></extra>"))
    fig.add_vline(x=a.w_true, line=dict(color=C.COLOR_TRUTH, width=2, dash="dash"), annotation_text="wahre Breite", annotation_position="top right")
    fig.add_trace(go.Scatter(x=[a.w_prof], y=[float(a.prof.min())], mode="markers", name=f"Minimum bei {width_label(a.w_prof)}", marker=dict(symbol="x", size=12, color=C.COLOR_TRIVIAL, line=dict(width=2))))
    fig.update_xaxes(title="Breite w′ des Kerns")
    fig.update_yaxes(title="min über x von J", type="log", exponentformat="power")
    return _fixed(fig, 320)


def build_evidence_chart(a: Analysis) -> go.Figure:
    """Log-Evidenz log p(b | w′) (x wegmarginalisiert) über die Breite w′."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.grid_w, y=a.evid, mode="lines+markers", name="Evidenz log p(b | w′)", line=dict(color=C.COLOR_BLIND, width=3), marker=dict(size=5), hovertemplate="w′ %{x:.3f}<br>log p %{y:.1f}<extra></extra>"))
    fig.add_vline(x=a.w_true, line=dict(color=C.COLOR_TRUTH, width=2, dash="dash"), annotation_text="wahre Breite", annotation_position="top right")
    fig.add_trace(go.Scatter(x=[a.w_evid], y=[float(a.evid.max())], mode="markers", name=f"Maximum bei {width_label(a.w_evid)}", marker=dict(symbol="x", size=12, color=C.COLOR_TRIVIAL, line=dict(width=2))))
    fig.update_xaxes(title="Breite w′ des Kerns")
    fig.update_yaxes(title="log p(b | w′) bis auf eine Konstante")
    return _fixed(fig, 320)


# ----------------------------------------------------------------------------- Experimente

def build_ambiguity_chart(res: dict) -> go.Figure:
    """Datenunterschied des mehrdeutigen Paares (Prozent) mit Kasten, über die neue Breite w′; ohne Kasten liegt er bei null."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[width_label(r["w_new"]) for r in res["rows"]], y=[100.0 * r["gap_box"] for r in res["rows"]], name="mit Kasten", marker_color=C.COLOR_TRIVIAL,
                         text=[f"{de(100.0 * r['gap_box'], 1)} %" for r in res["rows"]], textposition="outside"))
    fig.add_trace(go.Bar(x=[width_label(r["w_new"]) for r in res["rows"]], y=[100.0 * r["gap_hills"] for r in res["rows"]], name="nur Hügel", marker_color=C.COLOR_TRUTH,
                         text=["< 10⁻¹⁰ %" for _ in res["rows"]], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title=f"Breite w′ des zweiten Paares (wahr: {width_label(res['w'])})", type="category")
    fig.update_yaxes(title="Datenunterschied (%)", range=[0, 7])
    return _fixed(fig, 340)


def build_kernel_step_chart(res: dict) -> go.Figure:
    """Kern-Fehler des Kernschritts bei bekanntem x nach Objekt und Rauschen (Median; logarithmisch)."""
    fig = go.Figure()
    for obj, color in zip(C.OBJECTS, (C.COLOR_BLIND, C.COLOR_KNOWN, C.COLOR_TRUTH)):
        rows = [r for r in res["steps"] if r["obj"] == obj]
        fig.add_trace(go.Bar(x=[noise_label(r["noise"]) for r in rows], y=[r["median"] for r in rows], name=C.OBJECT_SHORT[obj], marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Rauschen", type="category")
    fig.update_yaxes(title="Kern-Fehler (Median)", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_objective_chart(res: dict) -> go.Figure:
    """J des wahren Paares gegen J des Trivialpaares (Median) bei Rauschen 1 % nach Objekt."""
    rows = [r for r in res["rows"] if r["noise"] == 1e-2]
    labels = [C.OBJECT_SHORT[r["obj"]] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["j_true"] for r in rows], name="wahres Paar", marker_color=C.COLOR_TRUTH))
    fig.add_trace(go.Bar(x=labels, y=[r["j_triv"] for r in rows], name="Trivialpaar (Kern = Delta)", marker_color=C.COLOR_TRIVIAL))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Objekt (Rauschen 1 %)", type="category")
    fig.update_yaxes(title="Zielfunktion J (Median)", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_start_chart(res: dict) -> go.Figure:
    """Breite des Ergebnisses der Alternierung über die Startbreite (Gauß-Starts, Median), nach Objekt; die Diagonale ist „bleibt beim Start“."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.0, 0.07], y=[0.0, 0.07], mode="lines", name="bleibt beim Start", line=dict(color=C.COLOR_GREY, dash="dash")))
    for obj, color in zip(C.OBJECTS, (C.COLOR_BLIND, C.COLOR_KNOWN, C.COLOR_TRUTH)):
        rows = [r for r in res["rows"] if r["obj"] == obj and r["start"] == "gauss"]
        fig.add_trace(go.Scatter(x=[r["start_w"] for r in rows], y=[r["width"] for r in rows], mode="lines+markers", name=C.OBJECT_SHORT[obj], line=dict(color=color, width=3), marker=dict(size=7)))
    fig.add_hline(y=res["w"], line=dict(color=C.COLOR_REF, width=1.5, dash="dot"), annotation_text="wahre Breite", annotation_position="top left")
    fig.update_xaxes(title="Startbreite", tickvals=[0.01, 0.02, 0.03, 0.04, 0.06])
    fig.update_yaxes(title="Breite des Ergebnisses (Median)", range=[0, 0.07])
    return _fixed(fig, 340)


def build_evidence_exp_chart(res: dict) -> go.Figure:
    """Geschätzte Breite (Median, Quartile) von Evidenz und Profil nach Rauschen, für Prior-Ziehungen und das Phantom als Wahrheit."""
    fig = go.Figure()
    for truth, color, dash in (("prior", C.COLOR_TRUTH, "solid"), ("phantom", C.COLOR_KNOWN, "solid")):
        rows = [r for r in res["rows"] if r["truth"] == truth]
        fig.add_trace(go.Scatter(x=[noise_label(r["noise"]) for r in rows], y=[r["w_evid"] for r in rows], mode="lines+markers", line=dict(color=color, width=3, dash=dash), marker=dict(size=8),
                                 name="Evidenz, " + ("Prior-Ziehung" if truth == "prior" else "Phantom"),
                                 error_y=dict(type="data", symmetric=False, array=[r["q75"] - r["w_evid"] for r in rows], arrayminus=[r["w_evid"] - r["q25"] for r in rows])))
    rows = [r for r in res["rows"] if r["truth"] == "prior"]
    fig.add_trace(go.Scatter(x=[noise_label(r["noise"]) for r in rows], y=[r["w_prof"] for r in rows], mode="lines+markers", name="Profil (beide Wahrheiten)", line=dict(color=C.COLOR_TRIVIAL, width=2, dash="dot"),
                             marker=dict(size=6)))
    fig.add_hline(y=res["w"], line=dict(color=C.COLOR_REF, width=1.5, dash="dot"), annotation_text="wahre Breite", annotation_position="top left")
    fig.update_xaxes(title="Rauschen", type="category")
    fig.update_yaxes(title="geschätzte Breite (Median, Quartile)", range=[0, 0.05])
    return _fixed(fig, 340)
