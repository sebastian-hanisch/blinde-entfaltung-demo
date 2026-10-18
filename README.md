# Blinde Entfaltung: Kern und Signal gemeinsam schätzen – Streamlit-Demo

Zwölftes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [aufloesung-messorte-demo](https://github.com/sebastian-hanisch/aufloesung-messorte-demo), [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo), [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo) und [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo) auf und nutzt dieselbe Welt: ein Phantom aus zwei Hügeln und einem Kasten, gemessen mit einer Gauß-Unschärfe und weißem Rauschen. Neu ist, dass der **Kern unbekannt** ist. Die Wahrheit ist bekannt, weil das Phantom selbst die Wahrheit ist; sie dient hier nur als Maßstab.

## Kernfrage

Kann man den Kern **zusammen mit dem Signal** aus den Daten schätzen (**Alternieren**, **Profil über die Breite**, **Evidenz**)? Warum geht das schlecht (das Paar „kein Blur“ hat das kleinere Ziel; ein Teil der Aufgabe ist exakt mehrdeutig), und was hilft (**marginalisieren statt minimieren**; ein dünn besetztes Signal)?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, Gitter, Operator, Rauschen):

- **Daten:** b = A(κ) x + e mit der Faltungsmatrix A(κ)ᵢⱼ = κ[i − j + m] (13 Werte, m = 6, Nullränder), e ~ N(0, σ² I). Die Daten sind **linear in x und linear in κ**, aber nicht in beiden zusammen. Standardinstanz n = 48, wahre Breite w = 0,03, Rauschen 1 % des Effektivwerts, Daten ohne Gitterfehler (b = A x + e).
- **Signale:** das Phantom, **nur die Hügel** (ohne Kasten) und **drei Spitzen**. Das Signal entscheidet, ob der Kern aus x bestimmbar ist.
- **Zielfunktion der Alternierung:** J(κ, x) = |A(κ) x − b|² + λ² |x|² (Tikhonov, λ = σ/τ, τ = 0,5) oder + 2μ |x|₁ (L1, μ = 0,5 σ √(2 ln n), exakter Aktivmengen-Löser wie in Stück 6).
- **Kernschritt:** min |Σ_d κ_d S_d x − b|² unter κ ≥ 0 und Σ κ = 1: nicht negative kleinste Quadrate (eigene Aktivmenge nach Lawson und Hanson, die Summe über eine schwere Zeile), im Test gegen scipy geprüft. Die Zielfunktion fällt in jeder Runde.
- **Profil über die Breite:** J(w′) = min_x J(A(w′), x) auf einem Gitter w′ = 0,010 bis 0,060.
- **Evidenz:** log p(b | w′) = −½ bᵀ(K + σ²I)⁻¹b − ½ log det(K + σ²I), K = A(w′) P A(w′)ᵀ, exponentieller Prior (ℓ = 0,12, τ = 0,5) wie in Stück 8.
- **Trivialpaar:** κ = Delta („kein Blur“), x = die geglättete Messung.

Alles ist erfunden; die Wahrheit ist das simulierte Phantom, die Hügel oder die Spitzen.

## Methodik

- **Verdikt** (Alternierung): Kern-Fehler höchstens 20 % **und** der Startkern war nicht schon so nah (sonst „bestätigt, nicht gefunden“); sonst: Trivialpaar, „das Ziel ist nicht größer als beim wahren Paar“ oder „im Start steckengeblieben“. Profil und Evidenz: die Breite innerhalb von 10 % der wahren.
- **Experimente** laufen über feste Zufallsströme (SplitMix64, Seeds ab 1000); nie ein einzelner Seed: 20 Ströme (Experimente 1 und 2), 8 (Experiment 3), 30 (Experiment 4), jeweils der Median.
- **Gleichstände und letzte Stellen** werden nicht festgelegt; die Tests prüfen Schranken und Mediane.

## Befunde (gemessen, keine Behauptungen)

Standardinstanz (n = 48, w = 0,03), Rauschen 1 %, wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py` oder `tests/test_presets.py`.

| Frage | Befund | Test |
|---|---|---|
| Ist die Aufgabe eindeutig (Experiment 1)? | **Nein, teilweise exakt mehrdeutig:** für Gauß-Hügel erklärt (b′² = b² − δ, w′² = w² + δ, h′ = h·b/b′) dieselben Daten (Unterschied unter 10⁻¹²) bei w′ = 0,0332 bis 0,0412 statt 0,0300; der **Kasten** bricht das (**1,5 bis 5,2 %** Datenunterschied). Skala (3 x, κ/3) exakt, Versatz im Innern bis 1,7·10⁻⁴. | `test_ambiguity_numbers` |
| Ist der Kern bei bekanntem x bestimmbar? | Der Kernschritt trifft ihn bei **Spitzen** auf 0,05 / 0,47 / 4,4 %, beim **Phantom** auf 0,5 / 4,2 / 32 % und bei den **Hügeln allein** nur auf 8 / 30 / 83 % (Rauschen 0,1 / 1 / 10 %). | `test_kernel_step_numbers` |
| Zeichnet das Ziel das wahre Paar aus (Experiment 2)? | **Nein:** das Trivialpaar hat in **9 von 9** Zellen (3 Signale × 3 Rauschstufen) in allen 20 Strömen das **kleinere** J; Phantom, Rauschen 1 %: **2,4·10⁻⁴ gegen 4,7·10⁻⁴**, Spitzen 4,2·10⁻⁵ gegen 2,2·10⁻⁴. Das Profil über die Breite wählt immer den Rand des Gitters (0,010). Beim Phantom ist das Trivialpaar bei Rauschen 1 % sogar näher an der Wahrheit (**21,9 % gegen 27,5 %**), bei den Spitzen nicht (80,1 % gegen 52,9 %). | `test_the_trivial_pair_wins_the_objective_in_every_cell`, `test_objective_numbers` |
| Findet die Alternierung den Kern (Experiment 3)? | **Nein, sie bleibt beim Start:** aus Gauß-Starts 0,01 / 0,02 / 0,03 / 0,04 / 0,06 enden die Kerne beim Phantom bei den Breiten **0,0085 / 0,0197 / 0,0297 / 0,0396 / 0,0561**; nur der Start bei der wahren Breite gibt einen richtigen Kern (er war es schon). Aus Delta bleibt Delta (Trivialpaar), aus dem Kasten wird ein Kern der Breite 0,075 mit 16-fachem J. | `test_alternation_stays_at_the_start` |
| Hilft ein parametrischer Kern? | **Nein:** jeder Start ist ein Fixpunkt (Breite nach 50 Runden 0,010 / 0,030 / 0,060, in einem von neun Fällen bewegt sich ein Strom). | `test_parametric_alternation_is_a_fixed_point` |
| Hilft ein dünn besetztes x? | **Ausnahme:** Spitzen mit L1-Prior aus dem breiten Start 0,05, Rauschen 1 %: Breite **0,0292**, Kern-Fehler 1,5 %, Fehler 3,0 % gegen 1,0 % bei bekanntem Kern; aus Delta (80,1 %) und aus 0,02 nicht; bei Rauschen 0,1 % bleibt die Breite aus 0,05 bei 0,0485. | `test_l1_on_spikes_finds_the_kernel_only_from_a_wide_start_at_medium_noise` |
| Hilft Marginalisieren (Experiment 4)? | **Ja, wenn das Modell stimmt:** bei Prior-Ziehungen als Wahrheit findet die Evidenz die Breite 0,030 (Median bei Rauschen 0,1 / 1 / 10 %), Fehler 14,2 / 18,5 / 25,6 % gegen 13,9 / 17,7 / 25,6 % bei bekannter Breite; das Profil wählt immer 0,010. **Beim Phantom verzerrt:** 0,034 / 0,036 / 0,038, Fehler 19,8 / 20,3 / 23,8 % gegen 9,4 / 12,1 / 19,1 %. | `test_evidence_numbers` |
| Einzelstichproben (Presets) | **Standard** (Phantom, Alternierung, Start 0,04): Breite 0,0396, Kern-Fehler 23,3 %, J 3,86·10⁻⁴ gegen 3,89·10⁻⁴ beim wahren Paar. **Start bei Delta:** Trivialpaar, Fehler 21,8 % statt 29,3 % bei bekanntem Kern. **Start 0,06:** Breite 0,0559, Fehler 119 %, J das 2,8-Fache. **Start 0,03:** Kern-Fehler 11,7 %, aber der Start war die Wahrheit. **Spitzen mit L1:** Kern-Fehler 1,7 %. **Profil:** Rand 0,010. **Evidenz:** 0,036. | `tests/test_presets.py` |
| Mini-Instanz von Hand | n = 3, κ* = (¼, ½, ¼), x = (1, 2, 1): b = (1, 1,5, 1); der Kernschritt trifft κ* exakt; Trivialpaar J = λ²·4,25/(1 + λ²) = 0,042079 (λ = 0,1) gegen 0,058665 beim wahren Paar; Evidenz n = 1: Maximum bei a²p = b² − σ². | `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Zielfunktion ist der falsche Richter.** Die Skizze hieß „Alternieren, Mehrdeutigkeit“ und erwartete, dass die Alternierung den Kern findet, wenn man lange genug rechnet. Die Messung zeigt eine Kette: das Ziel hat die falsche Lösung als Minimum (Experiment 2), die Alternierung folgt dem Ziel nur langsam und bleibt beim Start (Experiment 3), und ein Teil der Mehrdeutigkeit ist exakt (Experiment 1).
- **Das Trivialpaar ist kein Fehler, sondern ein gutes Ergebnis für das Ziel.** Sein J ist kleiner, und beim Phantom hat es bei Rauschen ≥ 1 % sogar den kleineren x-Fehler als die echte Entfaltung. „Kein Blur“ ist die vernünftige Antwort auf Daten, die man nicht entfalten kann (Stück 4).
- **Dünn besetztes x macht die Alternierung zu einem echten Verfahren, aber nur aus breitem Start und mittlerem Rauschen.**
- **Marginalisieren statt Minimieren ist modellabhängig:** gut, wenn das Modell stimmt (Prior-Ziehung), mit 13 bis 27 % Bias der Breite, wenn die Wahrheit ein Phantom ist (Stück 7 und 11).

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Ziel zeichnet das wahre Paar aus** | Beim Phantom und Rauschen 1 % ist J des Trivialpaares 2,4·10⁻⁴ gegen 4,7·10⁻⁴ beim wahren Paar. | Stück 16 (Verfahrensvergleich) |
| **Die Alternierung findet den Kern** | Sie bleibt bei der Startbreite; nur ein dünn besetztes Signal mit L1-Prior und breitem Start gelingt. | gelernte Priors (Stück 13 bis 15) |
| **Die Evidenz ist verlässlich** | Beim Phantom ist die Breite 0,0360 statt 0,0300: sie ist so gut wie der Prior. | – |
| **Ein Kern, ein Operator** | Ein Gauß-Kern mit 13 Werten, Faltung mit Nullrändern, eine Dimension, n = 48. | – |
| **Rauschen weiß und bekannt** | σ ist bekannt und wird nicht mitgeschätzt. | – |
| **Ein Start, eine Strafe** | Keine Mehrfach-Starts, keine Strafe auf die Kernglätte, keine Variationsverfahren (Levin und andere). | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [ica-demo](https://github.com/sebastian-hanisch/ica-demo) und [nmf-demo](https://github.com/sebastian-hanisch/nmf-demo): trennen mehrere Quellen blind; die Mehrdeutigkeiten (Skala, Versatz) sind verwandt, hier gibt es aber einen Kern und ein Signal, und es wird keine Unabhängigkeit benutzt.
- [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo), [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo), [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo): Stück 2, 6 und 7; der x-Schritt (Tikhonov, L1) und die Evidenz werden hier benutzt, nicht neu erklärt.
- [aufloesung-messorte-demo](https://github.com/sebastian-hanisch/aufloesung-messorte-demo): Stück 11; Nullraum und Auflösung sind das Vorbild der Mehrdeutigkeit.

## Tests

Mini-Instanz von Hand (Faltungsmatrix, verschobene Funktionen, Kernschritt, NNLS, Trivialpaar gegen wahres Paar, Tikhonov und L1, Alternierung, Profil, Evidenz für n = 1, Hügel- und Kastendaten, mehrdeutiges Paar), Orakel (Faltungsmatrix gegen `np.convolve` und den Operator des Generators, NNLS gegen scipy, Kernschritt auf rauschfreien Daten, Skala und Versatz, Hügelmehrdeutigkeit auf 10⁻¹², Kasten bricht sie, Trivialpaar gegen wahres Paar bei acht Strömen je Signal, Zielfunktion nie steigend, Alternierung gegen eine scipy-Nachrechnung, Evidenz gegen die Gaußdichte, Evidenzmaximum bei Prior-Ziehungen, Posterior-Mittel gegen die Informationsform, KKT-Bedingungen von L1, parametrische Fixpunkte), Generator, Auswertung (jede Schwelle des Urteils auf der Kante), jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler-Grenzen, jedes Signal, Verfahren, Kernmodell, x-Prior und jeder Start, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, Gitter, Operator) |
| `bld_model.py` | Faltungsmatrix, Kernschritt (NNLS), x-Schritte, Alternierung, Profil, Evidenz, exakte Mehrdeutigkeiten |
| `bld_evaluation.py` | Welt, Instanz, Analyse, Experimente, Verdikte |
| `bld_visualization.py`, `bld_pdf_export.py` | Figuren, PDF |
| `bld_constants.py`, `bld_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Mehrfach-Starts und globale Suche:** ein Start, eine Alternierung.
- **Strafe auf die Glätte des Kerns, Variationsverfahren (Levin und andere), Kerne mit mehr Freiheitsgraden:** nur Verweis.
- **Mitschätzen der Rauschstärke:** σ ist bekannt.
- **Zweidimensionale Bilder:** eine Dimension.
- **Gelernte Priors auf x:** Stück 13 bis 15.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html).
