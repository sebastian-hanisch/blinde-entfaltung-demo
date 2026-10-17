"""Auswertung: Formatierung, Welt und Instanz, die Analyse einer Einstellung, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import bld_constants as C
import bld_model as model
from inverse_generator import BOX, BUMPS, SplitMix64, grid, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 100 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


def width_label(w: float) -> str:
    """Breite (Anteil der Länge 1) mit vier Stellen, z. B. 0,0297."""
    return de(w, 4)


def rel_err(x: np.ndarray, xt: np.ndarray) -> float:
    """Fehler in der 2-Norm relativ zur Norm der Wahrheit."""
    return float(np.linalg.norm(x - xt) / np.linalg.norm(xt))


# ----------------------------------------------------------------------------- Welt und Instanz

@lru_cache(maxsize=256)
def truth_signal(obj: str, n: int = C.N) -> np.ndarray:
    """Die Wahrheit x: `phantom` (zwei Hügel und ein Kasten), `hills` (das Phantom ohne Kasten), `spikes` (drei Impulse; Masse mal n in der Zelle floor(Ort mal n))."""
    s = grid(n)
    if obj == "phantom":
        x = phantom(s)
    elif obj == "hills":
        x = phantom(s) - BOX[2] * ((s >= BOX[0]) & (s < BOX[1]))
    else:
        x = np.zeros(n)
        for pos, mass in C.SPIKES:
            x[int(math.floor(pos * n))] += mass * n
    x.flags.writeable = False
    return x


def noise_sigma(clean: np.ndarray, noise: float) -> float:
    """Absolute Standardabweichung des Rauschens: `noise` mal Effektivwert der exakten Daten."""
    return float(noise) * float(np.sqrt(np.mean(np.asarray(clean) ** 2)))


@lru_cache(maxsize=512)
def instance(obj: str, noise: float, w: float, seed: int):
    """(x, b, sigma, clean): Wahrheit, Daten b = A x + e (das Modell stimmt, kein Gitterfehler), absolutes Rauschen, rauschfreie Daten."""
    x = truth_signal(obj)
    clean = model.op(C.N, w) @ x
    sigma = noise_sigma(clean, noise)
    return x, clean + SplitMix64(seed).normal(C.N) * sigma, sigma, clean


def start_kernel(start: str, start_w: float) -> np.ndarray:
    """Startkern der Alternierung: Delta, Gauß-Kern der Startbreite oder Kasten."""
    if start == "delta":
        return model.delta_kernel(C.KERNEL_HALF)
    if start == "box":
        return model.box_kernel(C.KERNEL_HALF)
    return model.gauss_kernel(start_w, C.N, C.KERNEL_HALF)


@dataclass(frozen=True)
class Settings:
    obj: str = "phantom"
    noise: float = C.DEFAULT_NOISE
    w: float = C.DEFAULT_W
    kernel: str = "nonpar"
    method: str = "alternate"
    prior: str = "tikhonov"
    start: str = "gauss"
    start_w: float = C.DEFAULT_START_W
    rounds: int = C.DEFAULT_ROUNDS
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    x: np.ndarray
    b: np.ndarray
    sigma: float
    kappa_true: np.ndarray
    kappa_start: np.ndarray
    kappa_hat: np.ndarray
    x_known: np.ndarray       # Rekonstruktion bei bekanntem Kern (gleiches x-Verfahren)
    x_hat: np.ndarray         # Rekonstruktion des Verfahrens
    x_triv: np.ndarray        # Rekonstruktion des Trivialpaares (Kern = Delta)
    w_hat: float              # Breite des geschätzten Kerns
    w_true: float
    kerr: float               # Kern-Fehler (relative 2-Norm)
    kerr_start: float
    xerr: float
    xerr_known: float
    xerr_triv: float
    score_kind: str           # "J" (Zielfunktion, kleiner ist besser) oder "evidence" (log p(b), größer ist besser)
    score_hat: float
    score_true: float
    score_triv: float         # nan bei der Evidenz
    js: list                  # Verlauf der Zielfunktion je Runde (leer bei Profil und Evidenz)
    widths_hist: list         # Kernbreite je Runde
    xerrs_hist: list
    kerrs_hist: list
    grid_w: np.ndarray        # Breitengitter von Profil und Evidenz
    prof: np.ndarray          # Kosten J(w') je Breite
    evid: np.ndarray          # log p(b | w') je Breite
    w_prof: float
    w_evid: float
    at_edge: bool             # ŵ liegt am Rand des Gitters (Profil und Evidenz)


def _kernel_from_width(w: float) -> np.ndarray:
    return model.gauss_kernel(w, C.N, C.KERNEL_HALF)


def analyse(settings: Settings) -> Analysis:
    st = settings
    n, m = C.N, C.KERNEL_HALF
    x, b, sigma, _ = instance(st.obj, st.noise, st.w, st.seed)
    a_true = model.op(n, st.w)
    kt = model.true_kernel(st.w, n, m)
    k0 = start_kernel(st.start, st.start_w)
    kerr_start = rel_err(_kernel_from_width(model.snap(st.start_w, C.PARAM_GRID)) if st.kernel == "gauss" else k0, kt)
    p = model.prior_cov(grid(n))
    gw = np.array(C.PROFILE_GRID)
    prof = model.profile_costs(b, sigma, st.prior, gw)
    evid = model.evidence_curve(b, sigma, gw)
    w_prof, w_evid = float(gw[int(np.argmin(prof))]), float(gw[int(np.argmax(evid))])
    kappa_triv, x_triv, j_triv = model.trivial_pair(b, sigma, st.prior, m)
    js, widths_hist, xerrs_hist, kerrs_hist = [], [], [], []
    at_edge = False
    if st.method == "alternate" and st.kernel == "nonpar":
        run = model.alternate(b, sigma, st.prior, k0, st.rounds)
        kappa_hat, x_hat = run["kappas"][-1], run["xs"][-1]
        js = run["js"]
        widths_hist = [model.kernel_width(k, n) for k in run["kappas"]]
        kerrs_hist = [rel_err(k, kt) for k in run["kappas"]]
        xerrs_hist = [rel_err(xx, x) for xx in run["xs"]]
        x_known = model.x_step(a_true, b, st.prior, sigma)
        score_kind, score_hat, score_true, score_triv = "J", js[-1], model.objective(a_true, x_known, b, st.prior, sigma), j_triv
        w_hat = widths_hist[-1]
    elif st.method == "alternate":
        run = model.alternate_width(b, sigma, st.prior, st.start_w, C.PARAM_GRID, st.rounds)
        w_hat = run["widths"][-1]
        kappa_hat, x_hat = _kernel_from_width(w_hat), run["xs"][-1]
        js = run["js"]
        widths_hist = list(run["widths"])
        kerrs_hist = [rel_err(_kernel_from_width(v), kt) for v in run["widths"]]
        xerrs_hist = [rel_err(xx, x) for xx in run["xs"]]
        x_known = model.x_step(a_true, b, st.prior, sigma)
        score_kind, score_hat, score_true, score_triv = "J", js[-1], model.objective(a_true, x_known, b, st.prior, sigma), j_triv
    elif st.method == "profile":
        w_hat = w_prof
        at_edge = w_hat in (float(gw[0]), float(gw[-1]))
        kappa_hat = _kernel_from_width(w_hat)
        x_hat = model.x_step(model.op(n, w_hat), b, st.prior, sigma)
        x_known = model.x_step(a_true, b, st.prior, sigma)
        score_kind, score_hat, score_true, score_triv = "J", float(prof.min()), model.objective(a_true, x_known, b, st.prior, sigma), j_triv
    else:
        w_hat = w_evid
        at_edge = w_hat in (float(gw[0]), float(gw[-1]))
        kappa_hat = _kernel_from_width(w_hat)
        x_hat = model.posterior_mean(model.op(n, w_hat), b, sigma, p)
        x_known = model.posterior_mean(a_true, b, sigma, p)
        score_kind, score_hat, score_true, score_triv = "evidence", float(evid.max()), model.log_evidence(a_true, b, sigma, p), math.nan
    return Analysis(
        settings=st, s=grid(n), x=x, b=b, sigma=sigma, kappa_true=kt, kappa_start=k0, kappa_hat=kappa_hat, x_known=x_known, x_hat=x_hat, x_triv=x_triv, w_hat=w_hat, w_true=st.w,
        kerr=rel_err(kappa_hat, kt), kerr_start=kerr_start, xerr=rel_err(x_hat, x), xerr_known=rel_err(x_known, x), xerr_triv=rel_err(x_triv, x), score_kind=score_kind, score_hat=score_hat,
        score_true=score_true, score_triv=score_triv, js=js, widths_hist=widths_hist, xerrs_hist=xerrs_hist, kerrs_hist=kerrs_hist, grid_w=gw, prof=prof, evid=evid, w_prof=w_prof, w_evid=w_evid,
        at_edge=at_edge,
    )


# ----------------------------------------------------------------------------- Experiment 1: Mehrdeutigkeit

def ambiguity_experiment() -> dict:
    """Teil 1: die exakte Mehrdeutigkeit der Gauß-Hügel (b'^2 = b^2 - delta, w'^2 = w^2 + delta) und der Kasten, der sie bricht (analytisch, vom Gitter unabhängig). Teil 2: Skala und Versatz.
    Teil 3: der Kernschritt bei bekanntem x, Kern-Fehler nach Objekt und Rauschen (Median und Maximum über EXP_REPS Ströme)."""
    w, n, m = C.DEFAULT_W, C.N, C.KERNEL_HALF
    rows = [{"delta": d, "w_new": math.sqrt(w * w + d), "gap_hills": model.ambiguity_gap(d, w, False), "gap_box": model.ambiguity_gap(d, w, True)} for d in C.AMBIGUITY_DELTAS]
    kt = model.true_kernel(w, n, m)
    sym = []
    for obj in C.OBJECTS:
        x = truth_signal(obj)
        data = model.toeplitz(kt, n) @ x
        scale = float(np.abs(model.toeplitz(kt / 3.0, n) @ (3.0 * x) - data).max() / np.abs(data).max())
        shifted = model.toeplitz(model.shift_vec(kt, 1), n) @ model.shift_vec(x, -1)
        sym.append({"obj": obj, "scale": scale, "shift": float(np.abs(shifted - data)[3:-3].max() / np.abs(data).max())})
    steps = []
    for obj in C.OBJECTS:
        for noise in C.NOISE_CHOICES:
            errs = []
            for rep in range(C.EXP_REPS):
                x, b, _, _ = instance(obj, noise, w, C.SEED_BASE + rep)
                errs.append(rel_err(model.kernel_step(x, b, m), kt))
            steps.append({"obj": obj, "noise": noise, "median": float(np.median(errs)), "max": float(np.max(errs))})
    return {"rows": rows, "sym": sym, "steps": steps, "w": w}


# ----------------------------------------------------------------------------- Experiment 2: das Ziel als Richter

def objective_experiment() -> dict:
    """Wahres Paar gegen Trivialpaar (Kern = Delta, x = Daten geglättet): Zielfunktion und x-Fehler, und das Profil über die Breite (Tikhonov), nach Objekt und Rauschen, über EXP_REPS Ströme."""
    w, n = C.DEFAULT_W, C.N
    gw = np.array(C.PROFILE_GRID)
    rows = []
    for obj in C.OBJECTS:
        for noise in C.NOISE_CHOICES:
            jt, jv, et, ev_, hat, edge = [], [], [], [], [], 0
            for rep in range(C.EXP_REPS):
                x, b, sigma, _ = instance(obj, noise, w, C.SEED_BASE + rep)
                a = model.op(n, w)
                xo = model.x_step(a, b, "tikhonov", sigma)
                jt.append(model.objective(a, xo, b, "tikhonov", sigma))
                _, xv, j_triv = model.trivial_pair(b, sigma, "tikhonov", C.KERNEL_HALF)
                jv.append(j_triv)
                et.append(rel_err(xo, x))
                ev_.append(rel_err(xv, x))
                costs = model.profile_costs(b, sigma, "tikhonov", gw)
                hat.append(float(gw[int(np.argmin(costs))]))
                edge += int(hat[-1] == float(gw[0]))
            rows.append({"obj": obj, "noise": noise, "j_true": float(np.median(jt)), "j_triv": float(np.median(jv)), "share_triv_smaller": float(np.mean(np.array(jv) < np.array(jt))),
                         "xerr_true": float(np.median(et)), "xerr_triv": float(np.median(ev_)), "w_prof": float(np.median(hat)), "share_edge": edge / C.EXP_REPS})
    return {"rows": rows, "w": w, "reps": C.EXP_REPS}


# ----------------------------------------------------------------------------- Experiment 3: Alternierung

def _alt_cell(obj: str, noise: float, prior: str, start: str, start_w, rounds: int, reps: int) -> dict:
    n, m, w = C.N, C.KERNEL_HALF, C.DEFAULT_W
    kt = model.true_kernel(w, n, m)
    k0 = start_kernel(start, start_w if start_w else 0.03)
    widths, kerrs, xerrs, known, ratio = [], [], [], [], []
    for rep in range(reps):
        x, b, sigma, _ = instance(obj, noise, w, C.SEED_BASE + rep)
        run = model.alternate(b, sigma, prior, k0, rounds)
        a_true = model.op(n, w)
        xk = model.x_step(a_true, b, prior, sigma)
        widths.append(model.kernel_width(run["kappas"][-1], n))
        kerrs.append(rel_err(run["kappas"][-1], kt))
        xerrs.append(rel_err(run["xs"][-1], x))
        known.append(rel_err(xk, x))
        ratio.append(run["js"][-1] / model.objective(a_true, xk, b, prior, sigma))
    return {"obj": obj, "noise": noise, "prior": prior, "start": start, "start_w": start_w, "width": float(np.median(widths)), "kerr": float(np.median(kerrs)), "xerr": float(np.median(xerrs)),
            "xerr_known": float(np.median(known)), "j_ratio": float(np.median(ratio)), "share_ok": float(np.mean(np.array(kerrs) <= C.KERNEL_OK))}


def alternation_experiment() -> dict:
    """Teil 1: Alternierung (Tikhonov, 60 Runden, Rauschen 1 %) nach Objekt und Startkern. Teil 2: parametrische Alternierung der Breite nach Objekt und Startbreite. Teil 3: Spitzen mit L1 nach
    Rauschen und Startkern. Jede Zelle ist der Median über ALT_REPS Ströme."""
    w, n = C.DEFAULT_W, C.N
    rows = [_alt_cell(obj, 1e-2, "tikhonov", st, sw, C.DEFAULT_ROUNDS, C.ALT_REPS) for obj in C.OBJECTS for st, sw in C.ALT_STARTS]
    param = []
    for obj in C.OBJECTS:
        for w0 in (0.01, 0.03, 0.06):
            fin, xerr = [], []
            for rep in range(C.ALT_REPS):
                x, b, sigma, _ = instance(obj, 1e-2, w, C.SEED_BASE + rep)
                run = model.alternate_width(b, sigma, "tikhonov", w0, C.PARAM_GRID, 50)
                fin.append(run["widths"][-1])
                xerr.append(rel_err(run["xs"][-1], x))
            param.append({"obj": obj, "w0": w0, "width": float(np.median(fin)), "moved": float(np.mean([abs(f - model.snap(w0, C.PARAM_GRID)) > 1e-9 for f in fin])), "xerr": float(np.median(xerr))})
    l1 = [_alt_cell("spikes", noise, "l1", st, sw, C.L1_ROUNDS, C.ALT_REPS) for noise in (1e-3, 1e-2) for st, sw in C.L1_STARTS]
    return {"rows": rows, "param": param, "l1": l1, "w": w, "n": n}


# ----------------------------------------------------------------------------- Experiment 4: Marginalisieren statt Minimieren

def evidence_experiment() -> dict:
    """Die Breite schätzen, EVID_REPS Ströme je Zelle: Evidenz (x wegmarginalisiert) gegen Profil (x wegminimiert, Tikhonov), wenn die Wahrheit eine Prior-Ziehung ist (das Modell stimmt) und wenn sie
    das Phantom ist. Fehler der Rekonstruktion (Posterior-Mittel) mit der geschätzten gegen die bekannte Breite."""
    w, n = C.DEFAULT_W, C.N
    s = grid(n)
    p = model.prior_cov(s)
    chol = np.linalg.cholesky(p)
    gw = np.array(C.PROFILE_GRID)
    ref = model.op(n, w)
    rows = []
    for truth in ("prior", "phantom"):
        for noise in C.NOISE_CHOICES:
            sigma = noise_sigma(ref @ phantom(s), noise)
            he, hp, ee, ek = [], [], [], []
            for rep in range(C.EVID_REPS):
                rng = SplitMix64(C.SEED_BASE + 100 + rep)
                x = chol @ rng.normal(n) if truth == "prior" else phantom(s)
                b = ref @ x + rng.normal(n) * sigma
                evid = model.evidence_curve(b, sigma, gw)
                prof = model.profile_costs(b, sigma, "tikhonov", gw)
                we, wp = float(gw[int(np.argmax(evid))]), float(gw[int(np.argmin(prof))])
                he.append(we)
                hp.append(wp)
                ee.append(rel_err(model.posterior_mean(model.op(n, we), b, sigma, p), x))
                ek.append(rel_err(model.posterior_mean(ref, b, sigma, p), x))
            rows.append({"truth": truth, "noise": noise, "w_evid": float(np.median(he)), "q25": float(np.percentile(he, 25)), "q75": float(np.percentile(he, 75)), "w_prof": float(np.median(hp)),
                         "err_evid": float(np.median(ee)), "err_known": float(np.median(ek))})
    return {"rows": rows, "w": w, "reps": C.EVID_REPS}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def _scores_text(a: Analysis) -> str:
    if a.score_kind == "evidence":
        return f"log p(b | w′) = {de(a.score_hat, 1)} bei ŵ, {de(a.score_true, 1)} bei der wahren Breite (größer ist besser)"
    return f"Zielfunktion J = {sci(a.score_hat)} (wahres Paar {sci(a.score_true)}, Trivialpaar {sci(a.score_triv)}; kleiner ist besser)"


def verdict_instance(a: Analysis) -> tuple[str, str]:
    """Findet das Verfahren den Kern? Alternierung: Kern-Fehler gegen KERNEL_OK, mit Start und Zielfunktion als Zeugen; Profil und Evidenz: die geschätzte Breite gegen die wahre."""
    st = a.settings
    lead = (f"{C.METHOD_SHORT[st.method]}, {C.OBJECT_SHORT[st.obj]}, Rauschen {noise_label(st.noise)}: geschätzte Kernbreite {width_label(a.w_hat)} (wahr {width_label(a.w_true)}), "
            f"Kern-Fehler {err_pct(a.kerr)}, Fehler der Rekonstruktion {err_pct(a.xerr)} (bei bekanntem Kern {err_pct(a.xerr_known)}, Trivialpaar {err_pct(a.xerr_triv)}); {_scores_text(a)}.")
    if st.method != "alternate":
        close = abs(a.w_hat - a.w_true) <= C.WIDTH_OK * a.w_true
        if close:
            return "success", f"{lead} Die Breite liegt innerhalb von 10 % der wahren."
        if a.at_edge:
            return "warning", f"{lead} Die Breite liegt am Rand des Gitters ({width_label(a.w_hat)}): das Ziel fällt, je schmaler der Kern, und wählt nicht die Wahrheit."
        return "warning", f"{lead} Die Breite verfehlt die wahre um mehr als 10 %: die Evidenz ist so gut wie das Modell, und das Phantom ist keine Prior-Ziehung."
    if a.kerr <= C.KERNEL_OK:
        if a.kerr_start > C.KERNEL_OK:
            return "success", f"{lead} Der Kern ist gefunden: der Kern-Fehler fällt vom Start ({err_pct(a.kerr_start)}) auf {err_pct(a.kerr)}."
        start_err = "null" if a.kerr_start < C.NULL_ERR else err_pct(a.kerr_start)
        return "warning", f"{lead} Der Kern stimmt, aber der Startkern war schon nahe an der Wahrheit (Kern-Fehler {start_err}): das Verfahren hat den Start bestätigt, nicht den Kern gefunden."
    stays = st.start == "gauss" and st.kernel == "nonpar" and abs(a.w_hat - st.start_w) <= C.STAYS_TOL * st.start_w
    tail = f" Die Breite ist bei der Startbreite {width_label(st.start_w)} geblieben." if stays else ""
    if a.w_hat < C.TRIVIAL_WIDTH:
        return "warning", f"{lead} Das Ergebnis ist das Trivialpaar (Kern fast ein Delta, x ist die geglättete Messung): sein Ziel ist kleiner als das des wahren Paares.{tail}"
    if a.score_hat <= a.score_true:
        return "warning", f"{lead} Der Kern ist falsch, und das Ziel ist nicht größer als beim wahren Paar: das Ziel zeichnet die Wahrheit nicht aus.{tail}"
    return "warning", f"{lead} Der Kern ist falsch, und das Ziel ist größer als beim wahren Paar: das Verfahren ist im Start steckengeblieben.{tail}"


def _by(rows: list, **kw) -> dict:
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


def verdict_ambiguity(res: dict) -> tuple[str, str]:
    rows = res["rows"]
    shift = max(r["shift"] for r in res["sym"])
    steps = res["steps"]
    return "info", (f"Für Gauß-Hügel erklären (b′, w′) mit b′² = b² − δ und w′² = w² + δ die Daten exakt: bei δ = {de(rows[0]['delta'], 4)} bis {de(rows[-1]['delta'], 4)} (Breite {de(rows[0]['w_new'], 4)} bis "
                    f"{de(rows[-1]['w_new'], 4)} statt {de(res['w'], 4)}) ist der Datenunterschied unter 10⁻¹²; der Kasten bricht das, mit ihm sind es {de(100 * rows[0]['gap_box'], 1)} bis "
                    f"{de(100 * rows[-1]['gap_box'], 1)} %. Die Skala (3 x, κ/3) ist exakt (unter 10⁻¹²), der Versatz (x um eine Zelle, κ um eine Zelle zurück) im Innern bis {sci(shift)}. Bei bekanntem x trifft der Kernschritt den Kern bei Spitzen auf {err_pct(_by(steps, obj='spikes', noise=1e-2)['median'])}, "
                    f"beim Phantom auf {err_pct(_by(steps, obj='phantom', noise=1e-2)['median'])}, bei den Hügeln allein nur auf {err_pct(_by(steps, obj='hills', noise=1e-2)['median'])} (Rauschen 1 %).")


def verdict_objective(res: dict) -> tuple[str, str]:
    rows = res["rows"]
    all_smaller = sum(r["share_triv_smaller"] == 1.0 for r in rows)
    ph, sp = _by(rows, obj="phantom", noise=1e-2), _by(rows, obj="spikes", noise=1e-2)
    return "warning", (f"Das Trivialpaar (Kern = Delta) hat in {all_smaller} von {len(rows)} Zellen in allen {res['reps']} Strömen die kleinere Zielfunktion als das wahre Paar; bei Rauschen 1 % beim Phantom "
                       f"{sci(ph['j_triv'])} gegen {sci(ph['j_true'])}, bei den Spitzen {sci(sp['j_triv'])} gegen {sci(sp['j_true'])}. Das Profil über die Breite wählt in "
                       f"{sum(r['share_edge'] == 1.0 for r in rows)} von {len(rows)} Zellen in allen Strömen den Rand des Gitters ({width_label(rows[0]['w_prof'])}). Beim Phantom ist das Trivialpaar bei Rauschen 1 % sogar "
                       f"näher an der Wahrheit ({err_pct(ph['xerr_triv'])} gegen {err_pct(ph['xerr_true'])}), bei den Spitzen nicht ({err_pct(sp['xerr_triv'])} gegen {err_pct(sp['xerr_true'])}).")


def verdict_alternation(res: dict) -> tuple[str, str]:
    ph = [_by(res["rows"], obj="phantom", start="gauss", start_w=w0) for w0 in (0.01, 0.02, 0.03, 0.04, 0.06)]
    par = [_by(res["param"], obj="phantom", w0=w0) for w0 in (0.01, 0.03, 0.06)]
    l1 = _by(res["l1"], noise=1e-2, start="gauss", start_w=0.05)
    l1_low = _by(res["l1"], noise=1e-3, start="gauss", start_w=0.05)
    l1_delta = _by(res["l1"], noise=1e-2, start="delta")
    return "warning", (f"Beim Phantom (Rauschen 1 %) enden die Kerne aus Gauß-Starts der Breite 0,01 / 0,02 / 0,03 / 0,04 / 0,06 bei den Breiten {' / '.join(width_label(r['width']) for r in ph)}: das Verfahren bleibt "
                       f"beim Start, und nur der Start bei der wahren Breite führt zum richtigen Kern (er war es schon). Parametrisch (eine Breite) bleibt die Breite bei {' / '.join(width_label(r['width']) for r in par)}, jeder Start ist ein Fixpunkt. "
                       f"Bei Spitzen mit L1-Prior gelingt es aus dem breiten Start 0,05 (Breite {width_label(l1['width'])}, Kern-Fehler {err_pct(l1['kerr'])}, Fehler {err_pct(l1['xerr'])} gegen "
                       f"{err_pct(l1['xerr_known'])} bei bekanntem Kern), aus Delta nicht (Fehler {err_pct(l1_delta['xerr'])}); bei Rauschen 0,1 % bleibt die Breite aus dem Start 0,05 bei {width_label(l1_low['width'])}.")


def verdict_evidence(res: dict) -> tuple[str, str]:
    pr = [_by(res["rows"], truth="prior", noise=nz) for nz in C.NOISE_CHOICES]
    ph = [_by(res["rows"], truth="phantom", noise=nz) for nz in C.NOISE_CHOICES]
    return "info", (f"Bei Prior-Ziehungen als Wahrheit findet die Evidenz die Breite: Median {' / '.join(width_label(r['w_evid']) for r in pr)} bei Rauschen 0,1 / 1 / 10 %, Fehler "
                    f"{' / '.join(err_pct(r['err_evid']) for r in pr)} gegen {' / '.join(err_pct(r['err_known']) for r in pr)} bei bekannter Breite; das Profil wählt immer {width_label(pr[0]['w_prof'])}. "
                    f"Beim Phantom ist die Evidenz verzerrt (Median {' / '.join(width_label(r['w_evid']) for r in ph)}), der Fehler {' / '.join(err_pct(r['err_evid']) for r in ph)} gegen "
                    f"{' / '.join(err_pct(r['err_known']) for r in ph)}: sie ist so gut wie das Modell.")
