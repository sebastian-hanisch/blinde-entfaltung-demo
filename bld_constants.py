"""Konstanten: Regler, feste Annahmen, Experimente, Farben und Presets."""

# Welt: die lineare Entfaltung aus Stück 1 bis 8 (n = 48 Gitterpunkte, Phantom, Gauß-Unschärfe), aber der Kern ist unbekannt.
N = 48
KERNEL_HALF = 6                                                  # der Kern hat 2 * 6 + 1 = 13 Werte (Versätze -6 bis 6)
SPIKES = ((0.25, 0.04), (0.5, 0.03), (0.7, 0.05))                # Spitzen: (Ort als Anteil der Länge, Masse); die Höhe in der Zelle ist Masse mal n
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.02, 0.04, 0.005, 0.03       # wahre Breite der Unschärfe (Anteil der Länge 1)
NOISE_CHOICES = (1e-3, 1e-2, 1e-1)                               # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-2
OBJECTS = ("phantom", "hills", "spikes")
OBJECT_LABELS = {"phantom": "Phantom (zwei Hügel und ein Kasten)", "hills": "nur die Hügel (ohne Kasten)", "spikes": "drei Spitzen"}
OBJECT_SHORT = {"phantom": "Phantom", "hills": "nur Hügel", "spikes": "Spitzen"}
KERNELS = ("nonpar", "gauss")
KERNEL_LABELS = {"nonpar": "frei: 13 Werte, nicht negativ, Summe 1", "gauss": "Gauß-Kern mit einer Breite"}
KERNEL_SHORT = {"nonpar": "frei", "gauss": "Gauß"}
METHODS = ("alternate", "profile", "evidence")
METHOD_LABELS = {"alternate": "Alternieren (x und Kern abwechselnd)", "profile": "Profil über die Breite (x wegminimiert)", "evidence": "Evidenz über die Breite (x wegmarginalisiert)"}
METHOD_SHORT = {"alternate": "Alternierung", "profile": "Profil", "evidence": "Evidenz"}
PRIORS = ("tikhonov", "l1")
PRIOR_LABELS = {"tikhonov": "Tikhonov (glatt)", "l1": "L1 (dünn besetzt)"}
PRIOR_SHORT = {"tikhonov": "Tikhonov", "l1": "L1"}
STARTS = ("delta", "gauss", "box")
START_LABELS = {"delta": "Delta (kein Blur)", "gauss": "Gauß-Kern der Startbreite", "box": "Kasten (gleichverteilt)"}
START_SHORT = {"delta": "Delta", "gauss": "Gauß", "box": "Kasten"}
START_W_CHOICES = (0.01, 0.02, 0.03, 0.04, 0.06)
DEFAULT_START_W = 0.04
ROUNDS_CHOICES = (1, 5, 20, 60)
DEFAULT_ROUNDS = 60
SEED_MAX, DEFAULT_SEED = 999999, 7

# feste Modellannahmen
PRIOR_TAU, PRIOR_ELL = 0.5, 0.12                                 # x-Prior: Tikhonov mit lambda = sigma / tau, exponentieller Prior der Evidenz wie in Stück 7 und 8
L1_FACTOR = 0.5                                                  # L1-Gewicht mu = L1_FACTOR * sigma * sqrt(2 ln n)
NNLS_HEAVY = 1e3                                                 # Gewicht der Zeile "Summe des Kerns = 1" im Kernschritt
NNLS_TOL = 1e-12                                                 # Abbruchschwelle der Aktivmenge (relativ zur Skala der Aufgabe)
LASSO_MAXIT = 5000
PROFILE_GRID = tuple(0.01 + 0.002 * k for k in range(26))        # Gitter der Breite für Profil und Evidenz (0,010 bis 0,060)
PARAM_GRID = tuple(0.005 + 0.001 * k for k in range(76))         # Gitter der Breite für die parametrische Alternierung (0,005 bis 0,080)
GRID_EDGE_TOL = 1e-9                                             # eine Breite am Rand des Gitters
TRIVIAL_WIDTH = 0.005                                            # Kernbreite (Anteil der Länge 1) unter der der Kern als „kein Blur“ gilt
SHOW_RANGE = {"phantom": (-0.4, 1.9), "hills": (-0.4, 1.9), "spikes": (-0.6, 3.0)}   # feste Achse der Rekonstruktion je Objekt

# Verdikt-Schwellen
KERNEL_OK = 0.2                                                  # Kern-Fehler (relative 2-Norm), bis zu dem der Kern als gefunden gilt
XERR_FACTOR = 1.5                                                # x-Fehler höchstens das 1,5-Fache des x-Fehlers bei bekanntem Kern
WIDTH_OK = 0.1                                                   # Breite ŵ innerhalb von 10 % der wahren Breite
STAYS_TOL = 0.1                                                  # Ergebnisbreite innerhalb von 10 % der Startbreite: „bei der Startbreite geblieben“
NULL_ERR = 1e-9                                                  # Kern-Fehler unter dieser Schwelle gilt als null (Start bei der Wahrheit)

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 1000
EXP_REPS = 20                                                    # Wiederholungen je Zelle (Experimente 1 und 2)
ALT_REPS = 8                                                     # Experiment 3 (Alternierung)
EVID_REPS = 30                                                   # Experiment 4
AMBIGUITY_DELTAS = (0.0002, 0.0004, 0.0006, 0.0008)              # Experiment 1: b'^2 = b^2 - delta, w'^2 = w^2 + delta
ALT_STARTS = (("delta", None), ("gauss", 0.01), ("gauss", 0.02), ("gauss", 0.03), ("gauss", 0.04), ("gauss", 0.06), ("box", None))
L1_STARTS = (("delta", None), ("gauss", 0.02), ("gauss", 0.05))
L1_ROUNDS = 40

COLOR_TRUTH = "#2F6B65"
COLOR_BLIND = "#4c78a8"
COLOR_KNOWN = "#D68A2E"
COLOR_TRIVIAL = "#c0392b"
COLOR_START = "#8a8a8a"
COLOR_GREY = "#8a8a8a"
COLOR_REF = "#222222"

_BASE = {"obj": "phantom", "noise": DEFAULT_NOISE, "w": DEFAULT_W, "kernel": "nonpar", "method": "alternate", "prior": "tikhonov", "start": "gauss", "start_w": DEFAULT_START_W,
         "rounds": DEFAULT_ROUNDS, "seed": DEFAULT_SEED}
PRESETS = {
    "Standard": dict(_BASE),
    "Start bei Delta": dict(_BASE, start="delta"),
    "Start zu breit": dict(_BASE, start_w=0.06),
    "Start passt": dict(_BASE, start_w=0.03),
    "Spitzen mit L1": dict(_BASE, obj="spikes", prior="l1", start_w=0.06),
    "Profil über die Breite": dict(_BASE, method="profile"),
    "Evidenz": dict(_BASE, method="evidence"),
}
PRESET_HELP = {
    "Standard": "Phantom, Rauschen 1 %, Alternierung aus einem Gauß-Kern der Breite 0,04 (wahr 0,03): die Breite bleibt bei 0,040, und das Ziel ist nicht größer als beim wahren Paar.",
    "Start bei Delta": "Start ohne Blur: der Kern bleibt ein Delta, x ist die geglättete Messung (Fehler 21,8 % statt 29,3 % bei bekanntem Kern), und das Ziel ist kleiner als beim wahren Paar.",
    "Start zu breit": "Start bei 0,06: der Kern bleibt mit Breite 0,056 viel zu breit, die Rekonstruktion ist mit 119 % Fehler unbrauchbar, und das Ziel bleibt rund 2,8-mal so groß wie beim wahren Paar.",
    "Start passt": "Start bei der wahren Breite: das Ergebnis sieht gut aus (Fehler 12,7 %), aber der Start war schon die Wahrheit; das Verfahren hat ihn nur bestätigt.",
    "Spitzen mit L1": "Drei Spitzen mit L1-Prior aus dem breiten Start 0,06: der Kern wird gefunden (Kern-Fehler 1,7 %). Die Ausnahme: dünn besetztes x und ein breiter Start.",
    "Profil über die Breite": "Das Profil (x wegminimiert) fällt bis zum Rand des Gitters bei 0,010: das Ziel wählt den schmalsten Kern.",
    "Evidenz": "Die Evidenz (x wegmarginalisiert) wählt 0,036 statt 0,030: besser als das Profil, aber verzerrt, weil das Phantom keine Ziehung aus dem Prior ist.",
}
