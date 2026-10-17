"""Kern: Faltungsmatrix zu einem Kern, Kernschritt (nicht negative kleinste Quadrate mit Summe 1, eigene Aktivmenge), x-Schritte (Tikhonov und L1), Alternierung, Profil und Evidenz über die Breite,
parametrische Alternierung und die exakten Mehrdeutigkeiten. Nur numpy, kein Streamlit, kein scipy.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand, `tests/test_oracle_bld.py` gegen scipy (nnls, Gaußdichte),
np.convolve, die Mini-Instanz und die analytischen Mehrdeutigkeiten.

Modell: Daten b = A(kappa) x + e mit A(kappa)_ij = kappa[i - j + m] für |i - j| <= m (Nullränder), also linear in x UND linear im Kern kappa (b = sum_d kappa_d S_d x, S_d = Verschiebung um d).
Zielfunktion der Alternierung: J(kappa, x) = |A(kappa) x - b|^2 + Strafe(x) mit der Strafe lambda^2 |x|^2 (Tikhonov, lambda = sigma / tau) oder 2 mu |x|_1 (L1).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import bld_constants as C
from inverse_generator import BOX, BUMPS, grid, normal_cdf, operator

_TINY = 1e-300


# ----------------------------------------------------------------------------- Kerne und Faltungsmatrix

def toeplitz(kappa: np.ndarray, n: int) -> np.ndarray:
    """Faltungsmatrix A_ij = kappa[i - j + m] für |i - j| <= m (kappa hat die Länge 2m + 1, Mitte = Versatz 0), Nullränder."""
    kappa = np.asarray(kappa, dtype=float)
    m = (len(kappa) - 1) // 2
    a = np.zeros((n, n))
    for d in range(-m, m + 1):
        idx = np.arange(max(d, 0), min(n, n + d))
        a[idx, idx - d] = kappa[d + m]
    return a


def shift_columns(x: np.ndarray, m: int) -> np.ndarray:
    """Matrix (n, 2m + 1), deren Spalte d + m die um d verschobene Funktion S_d x ist (Nullränder): b = sum_d kappa_d S_d x = shift_columns(x, m) @ kappa."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.zeros((n, 2 * m + 1))
    for d in range(-m, m + 1):
        idx = np.arange(max(d, 0), n + min(d, 0))
        out[idx, d + m] = x[idx - d]
    return out


def delta_kernel(m: int) -> np.ndarray:
    """Der Kern „kein Blur“: eine Eins im Versatz 0."""
    k = np.zeros(2 * m + 1)
    k[m] = 1.0
    return k


def box_kernel(m: int) -> np.ndarray:
    """Gleichverteilter Kern."""
    return np.full(2 * m + 1, 1.0 / (2 * m + 1))


def gauss_kernel(w: float, n: int, m: int) -> np.ndarray:
    """Gauß-Kern der Breite w (Anteil der Länge 1) auf den Versätzen -m bis m, auf Summe 1 normiert."""
    d = np.arange(-m, m + 1) / n
    k = np.exp(-(d ** 2) / (2.0 * w * w))
    return k / k.sum()


def true_kernel(w: float, n: int, m: int) -> np.ndarray:
    """Der Kern des Operators aus dem Generator: die mittlere Zeile von operator(n, w) auf den Versätzen -m bis m, auf Summe 1 normiert."""
    a = op(n, w)
    row = np.array([a[n // 2, n // 2 + d] for d in range(-m, m + 1)])
    return row / row.sum()


def kernel_width(kappa: np.ndarray, n: int) -> float:
    """Breite des Kerns als Standardabweichung seiner Verteilung über die Versätze (Anteil der Länge 1)."""
    kappa = np.asarray(kappa, dtype=float)
    m = (len(kappa) - 1) // 2
    return math.sqrt(float(np.sum(kappa * (np.arange(-m, m + 1) / n) ** 2)))


@lru_cache(maxsize=256)
def _op(n: int, w: float) -> np.ndarray:
    a = operator(n, w)
    a.flags.writeable = False
    return a


def op(n: int, w: float) -> np.ndarray:
    """Der Operator des Generators (Gauß-Unschärfe der Breite w), zwischengespeichert; auf sechs Stellen gerundete Breite als Schlüssel."""
    return _op(n, round(float(w), 6))


# ----------------------------------------------------------------------------- Kernschritt: NNLS

def nnls(a: np.ndarray, b: np.ndarray, tol: float = C.NNLS_TOL) -> np.ndarray:
    """min |a x - b|^2 unter x >= 0 nach Lawson und Hanson (Aktivmenge): jeweils die Koordinate mit dem größten positiven Gradienten freigeben, auf der freien Menge die kleinsten Quadrate lösen und
    bei einer negativen Komponente bis zur Grenze zurückgehen."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = a.shape[1]
    x = np.zeros(p)
    passive = np.zeros(p, dtype=bool)
    scale = max(float(np.abs(a.T @ b).max()), _TINY)
    for _ in range(3 * p):
        w = a.T @ (b - a @ x)
        free = np.where(passive, -np.inf, w)
        j = int(np.argmax(free))
        if free[j] <= tol * scale:
            break
        passive[j] = True
        while True:
            s = np.zeros(p)
            s[passive] = np.linalg.lstsq(a[:, passive], b, rcond=None)[0]
            if s[passive].min() > 0.0:
                x = s
                break
            neg = passive & (s <= 0.0)
            alpha = float(np.min(x[neg] / (x[neg] - s[neg])))
            x = x + alpha * (s - x)
            passive = passive & (x > tol)
            x = np.where(passive, x, 0.0)
    return x


def kernel_step(x: np.ndarray, b: np.ndarray, m: int) -> np.ndarray:
    """Kern bei festem x: min |sum_d kappa_d S_d x - b|^2 unter kappa >= 0 und Summe kappa = 1 (NNLS mit einer schweren Zeile für die Summe, danach auf Summe 1 normiert)."""
    cols = shift_columns(x, m)
    heavy = C.NNLS_HEAVY
    sol = nnls(np.vstack([cols, heavy * np.ones((1, 2 * m + 1))]), np.append(np.asarray(b, dtype=float), heavy))
    return sol / sol.sum()


# ----------------------------------------------------------------------------- x-Schritte und Zielfunktion

def lam_of(sigma: float) -> float:
    """Tikhonov-Parameter lambda = sigma / tau (wie in Stück 7)."""
    return float(sigma) / C.PRIOR_TAU


def mu_of(sigma: float, n: int) -> float:
    """L1-Gewicht mu = L1_FACTOR sigma sqrt(2 ln n) (universelle Schwelle, halbiert)."""
    return C.L1_FACTOR * float(sigma) * math.sqrt(2.0 * math.log(n))


def tikhonov(a: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """x = (A^T A + lam^2 I)^-1 A^T b (Stück 2)."""
    return np.linalg.solve(a.T @ a + lam ** 2 * np.eye(a.shape[1]), a.T @ b)


def lasso(a: np.ndarray, y: np.ndarray, mu: float, maxit: int = C.LASSO_MAXIT, tol: float = 1e-10) -> np.ndarray:
    """Exakte Lösung von min 1/2 |a w - y|^2 + mu |w|_1 per Aktivmenge (wie in Stück 6): auf der aktiven Menge mit festen Vorzeichen das lineare System lösen, bei Vorzeichenwechsel die Koordinate
    mit dem frühesten Wechsel entfernen, sonst die am stärksten verletzte Bedingung |g_j| <= mu aufnehmen."""
    p = a.shape[1]
    mtm, mty = a.T @ a, a.T @ y
    w = np.zeros(p)
    active = np.zeros(p, dtype=bool)
    sgn = np.zeros(p)
    for _ in range(maxit):
        idx = np.flatnonzero(active)
        if idx.size:
            s = sgn[idx]
            sol = np.linalg.solve(mtm[idx][:, idx], mty[idx] - mu * s)
            bad = sol * s < 0.0
            if bad.any():
                w_old = w[idx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(bad, w_old / (w_old - sol), np.inf)
                j = int(np.argmin(t))
                w[idx[j]] = 0.0
                active[idx[j]] = False
                continue
            w[idx] = sol
        g = mtm @ w - mty
        viol = np.where(active, 0.0, np.abs(g) - mu)
        j = int(np.argmax(viol))
        if viol[j] <= tol * max(mu, _TINY):
            return w
        active[j] = True
        sgn[j] = -np.sign(g[j])
        w[j] = 0.0
    return w


def x_step(a: np.ndarray, b: np.ndarray, prior: str, sigma: float) -> np.ndarray:
    """x bei festem Operator: Tikhonov mit lambda = sigma / tau oder L1 mit mu."""
    if prior == "l1":
        return lasso(a, b, mu_of(sigma, a.shape[1]))
    return tikhonov(a, b, lam_of(sigma))


def objective(a: np.ndarray, x: np.ndarray, b: np.ndarray, prior: str, sigma: float) -> float:
    """J = |A x - b|^2 + Strafe: lambda^2 |x|^2 (Tikhonov) oder 2 mu |x|_1 (L1)."""
    data = float(np.sum((a @ x - b) ** 2))
    if prior == "l1":
        return data + 2.0 * mu_of(sigma, a.shape[1]) * float(np.abs(x).sum())
    return data + lam_of(sigma) ** 2 * float(x @ x)


# ----------------------------------------------------------------------------- Alternierung

def alternate(b: np.ndarray, sigma: float, prior: str, kappa0: np.ndarray, rounds: int) -> dict:
    """Abwechselnd x (bei festem Kern) und Kern (bei festem x). Gibt Kerne, x und J zurück, je rounds + 1 Einträge: Eintrag r ist der Kern vor der Runde r, das dazu optimale x und
    J(kappa_r, x_r). Der letzte Kern ist das Ergebnis der Runde rounds; J fällt nie."""
    b = np.asarray(b, dtype=float)
    n = len(b)
    m = (len(kappa0) - 1) // 2
    kap = np.asarray(kappa0, dtype=float)
    kappas, xs, js = [], [], []
    for _ in range(rounds):
        a = toeplitz(kap, n)
        x = x_step(a, b, prior, sigma)
        kappas.append(kap)
        xs.append(x)
        js.append(objective(a, x, b, prior, sigma))
        kap = kernel_step(x, b, m)
    a = toeplitz(kap, n)
    x = x_step(a, b, prior, sigma)
    kappas.append(kap)
    xs.append(x)
    js.append(objective(a, x, b, prior, sigma))
    return {"kappas": kappas, "xs": xs, "js": js}


def trivial_pair(b: np.ndarray, sigma: float, prior: str, m: int):
    """Das Paar „kein Blur“: Kern = Delta, x aus dem x-Schritt mit A = I. Gibt (Kern, x, J) zurück."""
    b = np.asarray(b, dtype=float)
    kap = delta_kernel(m)
    a = toeplitz(kap, len(b))
    x = x_step(a, b, prior, sigma)
    return kap, x, objective(a, x, b, prior, sigma)


# ----------------------------------------------------------------------------- Profil, parametrische Alternierung, Evidenz

def profile_costs(b: np.ndarray, sigma: float, prior: str, widths) -> np.ndarray:
    """J(w') = min_x J(A(w'), x) für jede Breite w' des Gitters (der Operator ist der des Generators)."""
    b = np.asarray(b, dtype=float)
    n = len(b)
    out = []
    for w in widths:
        a = op(n, w)
        out.append(objective(a, x_step(a, b, prior, sigma), b, prior, sigma))
    return np.array(out)


def snap(w: float, widths) -> float:
    """Der Gitterwert, der w am nächsten liegt (bei Gleichstand der kleinere)."""
    return float(min(widths, key=lambda v: abs(v - w)))


def alternate_width(b: np.ndarray, sigma: float, prior: str, w0: float, widths, rounds: int) -> dict:
    """Parametrische Alternierung: x-Schritt bei der aktuellen Breite, dann die Breite des Gitters, die den Datenfehler |A(w) x - b|^2 bei festem x minimiert. Gibt Breiten, x und J zurück, je rounds + 1
    Einträge (Eintrag r: Breite vor der Runde r, das dazu optimale x, J)."""
    b = np.asarray(b, dtype=float)
    n = len(b)
    w = snap(w0, widths)
    ws, xs, js = [], [], []
    for _ in range(rounds):
        a = op(n, w)
        x = x_step(a, b, prior, sigma)
        ws.append(w)
        xs.append(x)
        js.append(objective(a, x, b, prior, sigma))
        w = float(widths[int(np.argmin([np.sum((op(n, v) @ x - b) ** 2) for v in widths]))])
    a = op(n, w)
    x = x_step(a, b, prior, sigma)
    ws.append(w)
    xs.append(x)
    js.append(objective(a, x, b, prior, sigma))
    return {"widths": ws, "xs": xs, "js": js}


def exp_kernel(s: np.ndarray, ell: float) -> np.ndarray:
    """Korrelationsmatrix exp(-|s_i - s_j| / ell) (Stück 8)."""
    s = np.asarray(s, dtype=float)
    return np.exp(-np.abs(s[:, None] - s[None, :]) / ell)


def prior_cov(s: np.ndarray) -> np.ndarray:
    """Prior-Kovarianz tau^2 C + 1e-10 I (der Zuschlag hält die Zerlegung stabil)."""
    return C.PRIOR_TAU ** 2 * exp_kernel(s, C.PRIOR_ELL) + 1e-10 * np.eye(len(s))


def log_evidence(a: np.ndarray, b: np.ndarray, sigma: float, p: np.ndarray) -> float:
    """log p(b | A) = -1/2 b^T (K + sigma^2 I)^-1 b - 1/2 log det(K + sigma^2 I) mit K = A P A^T (Konstante weggelassen): x ist wegmarginalisiert."""
    k = a @ p @ a.T + sigma ** 2 * np.eye(len(b))
    logdet = float(np.linalg.slogdet(k)[1])
    return -0.5 * float(b @ np.linalg.solve(k, b)) - 0.5 * logdet


def posterior_mean(a: np.ndarray, b: np.ndarray, sigma: float, p: np.ndarray) -> np.ndarray:
    """P A^T (A P A^T + sigma^2 I)^-1 b."""
    return p @ a.T @ np.linalg.solve(a @ p @ a.T + sigma ** 2 * np.eye(len(b)), b)


def evidence_curve(b: np.ndarray, sigma: float, widths) -> np.ndarray:
    """log p(b | w') für jede Breite des Gitters (exponentieller Prior aus Stück 8)."""
    b = np.asarray(b, dtype=float)
    n = len(b)
    p = prior_cov(grid(n))
    return np.array([log_evidence(op(n, w), b, sigma, p) for w in widths])


# ----------------------------------------------------------------------------- Exakte Mehrdeutigkeiten

def hills_data(t: np.ndarray, w: float, bumps) -> np.ndarray:
    """Analytisch unscharfe Gauß-Hügel (Mitte c, Breite b, Höhe h): Hügel bleiben Hügel mit der Breite sqrt(b^2 + w^2)."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for c, width, height in bumps:
        v = width * width + w * w
        out = out + height * (width / math.sqrt(v)) * np.exp(-((t - c) ** 2) / (2.0 * v))
    return out


def box_data(t: np.ndarray, w: float, box=BOX) -> np.ndarray:
    """Analytisch unscharfer Kasten: Differenz zweier Normalverteilungsfunktionen."""
    left, right, height = box
    t = np.asarray(t, dtype=float)
    return height * (normal_cdf((t - left) / w) - normal_cdf((t - right) / w))


def ambiguous_pair(bumps, w: float, delta: float):
    """Das zweite Paar (Hügel, Breite) mit b'^2 = b^2 - delta, w'^2 = w^2 + delta und h' = h b / b': für Gauß-Hügel dieselben Daten wie (bumps, w)."""
    new = []
    for c, width, height in bumps:
        bp = math.sqrt(width * width - delta)
        new.append((c, bp, height * width / bp))
    return tuple(new), math.sqrt(w * w + delta)


def ambiguity_gap(delta: float, w: float, with_box: bool, bumps=BUMPS, box=BOX, points: int = 2001) -> float:
    """Relativer Unterschied (2-Norm) der Daten des Paares (bumps, w) und des mehrdeutigen Paares; mit_box: der Kasten bleibt gleich, nur der Kern wird breiter."""
    t = np.linspace(0.0, 1.0, points)
    new, wp = ambiguous_pair(bumps, w, delta)
    g0 = hills_data(t, w, bumps)
    g1 = hills_data(t, wp, new)
    if with_box:
        g0 = g0 + box_data(t, w, box)
        g1 = g1 + box_data(t, wp, box)
    return float(np.linalg.norm(g1 - g0) / np.linalg.norm(g0))


def shift_vec(v: np.ndarray, d: int) -> np.ndarray:
    """Um d Stellen nach rechts verschoben, mit Nullen aufgefüllt (negatives d: nach links)."""
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v)
    if d >= 0:
        out[d:] = v[: len(v) - d]
    else:
        out[: len(v) + d] = v[-d:]
    return out
