"""PDF-Export der aktuellen Instanz (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import bld_constants as C
from bld_evaluation import Analysis, de, err_pct, noise_label, sci, width_label

_REPLACE = {"→": "->", "–": "-", "—": "-", "σ": "sigma", "λ": "lambda", "κ": "kappa", "′": "'", "ŵ": "w-hat", "Γ": "Gamma", "×": "x", "τ": "tau", "−": "-", "€": "EUR", "≥": ">=", "·": "*", "⁻": "-",
            "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Blinde Entfaltung"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Objekt: {C.OBJECT_SHORT[s.obj]}, Rauschen {noise_label(s.noise)} des Effektivwerts, wahre Breite w = {de(s.w, 3)}, Verfahren: {C.METHOD_SHORT[s.method]} "
                            f"(Kernmodell {C.KERNEL_SHORT[s.kernel]}, x-Prior {C.PRIOR_SHORT[s.prior]}), Start: {C.START_SHORT[s.start]} (Startbreite {de(s.start_w, 2)}), {s.rounds} Runden, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    score = (f"log p(b | w') = {de(a.score_hat, 1)} bei w-hat, {de(a.score_true, 1)} bei der wahren Breite" if a.score_kind == "evidence"
             else f"Zielfunktion J = {sci(a.score_hat)} (wahres Paar {sci(a.score_true)}, Trivialpaar {sci(a.score_triv)})")
    lines = [
        f"Geschätzte Kernbreite {width_label(a.w_hat)} (wahr {width_label(a.w_true)}); Kern-Fehler {err_pct(a.kerr)}, Kern-Fehler des Starts {err_pct(a.kerr_start)}.",
        f"Fehler der Rekonstruktion: blind {err_pct(a.xerr)}, bei bekanntem Kern {err_pct(a.xerr_known)}, Trivialpaar (Kern = Delta) {err_pct(a.xerr_triv)}.",
        f"{score}; wahres Rauschen sigma = {sci(a.sigma)}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Versatz", "Start", "Wahrheit", "Schätzung"], [24, 40, 40, 40]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    m = C.KERNEL_HALF
    for i, d in enumerate(range(-m, m + 1)):
        for v, w in zip([f"{d}", de(a.kappa_start[i], 4), de(a.kappa_true[i], 4), de(a.kappa_hat[i], 4)], widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    head, widths = ["Punkt", "Ort s", "Wahrheit x", "Kern bekannt", "blind", "Trivialpaar"], [16, 26, 32, 36, 32, 36]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(C.N, MAX_ROWS)):
        row = [f"{i + 1}", de(a.s[i], 3), de(a.x[i], 3), de(a.x_known[i], 3), de(a.x_hat[i], 3), de(a.x_triv[i], 3)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if C.N > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {C.N - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Blinde Entfaltung: Kern und Signal gemeinsam schätzen. Quelle: blinde-entfaltung-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
