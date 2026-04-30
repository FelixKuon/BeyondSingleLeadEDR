# EDR Analysis Notebook
## Electrocardiogram-Derived Respiration — Methodenvergleich & Statistische Auswertung

> **Autor:** Felix Kuon  
> **Institution:** HAW Hamburg — Biomedizinische Signalverarbeitung  
> **Datenbasis:** Kontrolliertes Atemprotokoll (paced breathing), N=19 Probanden (nach QC)  
> **Ziel:** Vergleich von drei EDR-Methoden hinsichtlich Schätzgenauigkeit und SNR unter Einfluss von Atemrate und kardialer Periode

---

## Inhaltsverzeichnis

1. [Überblick](#überblick)
2. [Datenstruktur](#datenstruktur)
3. [Notebook-Sektionen](#notebook-sektionen)
   - [Sektion 1 — Setup & Konfiguration](#sektion-1--setup--konfiguration)
   - [Sektion 2 — Datenladen & Preprocessing](#sektion-2--datenladen--preprocessing)
   - [Sektion 3 — SNR-Qualitätsprüfung](#sektion-3--snr-qualitätsprüfung)
   - [Sektion 4 — Explorative Datenanalyse](#sektion-4--explorative-datenanalyse)
   - [Sektion 5 — SNR-Visualisierung](#sektion-5--snr-visualisierung)
   - [Sektion 6 — Long-Format Export](#sektion-6--long-format-export)
   - [Sektion 7–10 — LMM-Analyse](#sektion-710--lmm-analyse)
   - [Sektion 11–13 — Paper Figures](#sektion-1113--paper-figures)
4. [EDR-Methoden](#edr-methoden)
5. [Statistische Methodik](#statistische-methodik)
   - [Warum LMM?](#warum-lmm)
   - [Modellstruktur](#modellstruktur)
   - [Ergebnisinterpretation](#ergebnisinterpretation)
   - [ConvergenceWarning: Group Var ≈ 0](#convergencewarning-group-var--0)
6. [SPSS-Äquivalenz](#spss-äquivalenz)
7. [Ausgabedateien](#ausgabedateien)
8. [Abhängigkeiten](#abhängigkeiten)

---

## Überblick

Dieses Notebook implementiert eine vollständige Analyse-Pipeline für den Vergleich von drei EDR-Algorithmen (Electrocardiogram-Derived Respiration). EDR-Methoden schätzen die Atemfrequenz direkt aus dem EKG-Signal, ohne separaten Atemsensor.

### Forschungsfragen

1. **Genauigkeit:** Wie gut schätzen die drei Methoden die wahre Atemfrequenz, und hängt der Schätzfehler von der Atemrate ab?
2. **Herzperiodeneffekt:** Beeinflusst die kardiale Periode (Mean RR-Intervall) die Schätzgenauigkeit, und unterscheidet sich dieser Effekt zwischen Methoden?
3. **SNR:** Welche Methode liefert das beste Signal-zu-Rausch-Verhältnis, und ist dieses herzratenabhängig?

### Studiendesign

- **Protokoll:** Paced Breathing — Probanden atmen auf Kommando mit 7 vorgegebenen Raten (6–18 breaths/min, entspricht 0.10–0.30 Hz)
- **Messung:** 12-Kanal EKG, Vektorkardiogramm (VCG), Referenz-Atemsignal
- **Segmentierung:** Gleitendes Fenster über die Zeitserie, pro Segment wird eine Frequenzschätzung berechnet
- **Outcome:** Frequenzfehler in Hz (Schätzung − Referenz), SNR in dB

---

## Datenstruktur

### Rohdaten (`df_all`)

| Spalte | Typ | Beschreibung |
|---|---|---|
| `subject_id` | str/cat | Probandenkennung (z.B. `"Subject38"`) |
| `session_id` | str | Methodenkennung (`RR_Intervals`, `HeartMovement`, `VectorLength_R`) |
| `bpm_target` | float | Vorgegebene Atemrate in breaths/min |
| `freq_median_hz` | float | Geschätzte Atemfrequenz (Median über Segment) in Hz |
| `snr_median_db` | float | Signal-zu-Rausch-Verhältnis des dominanten Spektralpeak (dB) |
| `hrv_mean_rr_ms` | float | Mittleres RR-Intervall im Segment (ms) — HRV-Kovariate |
| `segment_id` | int | Segment-Index innerhalb der Session |
| `t_start_seg_s` | float | Segmentstart (Sekunden) |
| *weitere HRV-Spalten* | float | SDNN, RMSSD, pNN50, LF/HF, etc. |

### Long-Format für LMM (`LMM_data_long_format.csv`)

Eine Zeile pro **Subject × Atemrate × Methode**. Enthält:
- Demographische Kovariaten (`gender`, `age_group`, `BMI`)
- Outcome-Metriken (`SNR_dB`, `error_hz`, `error_hz_abs`)
- Methodenspezifische Metriken (`coverage_within_tol`, `freq_mae_hz`, `ridge_power_mean`)
- HRV-Kovariaten (methodenunabhängig, aggregiert über HeartMovement-Sessions)

---

## Notebook-Sektionen

### Sektion 1 — Setup & Konfiguration

Globale Konstanten, Pfade, Plotly-Theme-Einstellungen.

```python
SNR_THRESHOLD_DB = 1.0      # Schwellwert für Subject-Einschluss
METHODS_ORDER    = ["HeartMovement", "RR_Intervals", "R_Amplitude"]
METHOD_COLORS    = {"HeartMovement": "#E8871A",
                    "RR_Intervals":  "#4E9BD1",
                    "R_Amplitude":   "#4DB878"}
METHOD_LABELS    = {"HeartMovement": "Heartmovement EDR",
                    "RR_Intervals":  "RR-Interval EDR",
                    "R_Amplitude":   "R-Amplitude EDR"}
FONT, FS         = "Arial", 13
```

Hilfsfunktionen:
- `sig_star(p)` — gibt `"***"`, `"**"`, `"*"` oder `"ns"` zurück
- `paper_layout(title, width, height)` — einheitliches Plotly-Layout-Dict
- `ax_style(title)` — einheitliche Achsen-Konfiguration
- `fit_lmm(formula, df, label)` — wrapper für `statsmodels MixedLM`
- `extract_fe(model, label_map)` — extrahiert Fixed Effects mit CIs für Forest Plots

---

### Sektion 2 — Datenladen & Preprocessing

Lädt alle Subject-CSVs aus `data_dir`, vereint sie in `df_all`.

```
Preprocessing-Schritte:
1. session_id → edr_method mapping (paper_titles Dict)
2. Fehler berechnen: error_hz = freq_median_hz − (bpm_target / 60)
3. Standardisierung für LMM: bpm_target_z, hrv_mean_rr_ms_z (z-Score)
4. bpm_vals, hz_vals, hz_labels für konsistente Tick-Beschriftungen
```

---

### Sektion 3 — SNR-Qualitätsprüfung

**Ziel:** Probanden mit systematisch schlechtem Signal-zu-Rausch-Verhältnis ausschließen.

**Vorgehen:**
1. Für jeden Probanden: medianer SNR über alle Segmente und Methoden
2. Schwellwert: `SNR_THRESHOLD_DB = 1.0 dB`
3. Probanden unterhalb des Schwellwerts → `excluded_subjects`
4. Verbleibende → `good_subjects`, Datensatz → `df_all_good`

**Rationale:** Ein SNR < 1.0 dB bedeutet, dass das respiratorische Signal im Spektrum nicht substanziell über dem Rauschen liegt. Frequenzschätzungen aus solchen Segmenten wären nicht interpretierbar.

**Ausgabe:** `snr_by_subject` (Series: subject_id → median SNR)

---

### Sektion 4 — Explorative Datenanalyse

Deskriptive Statistiken pro Methode und Atemrate:

| Metrik | Beschreibung |
|---|---|
| Mean ± SD Error | Systematische Unterschätzung (negative Werte = Underestimation) |
| MAE | Mittlerer absoluter Fehler in Hz |
| RMSE | Root Mean Square Error |
| Coverage | Anteil Segmente innerhalb Toleranzband (±0.02 Hz) |
| Median SNR | Signal-zu-Rausch-Verhältnis |

---

### Sektion 5 — SNR-Visualisierung

Balkenplot des medianen SNR pro Subject, farblich nach Einschluss-Status:

- 🟢 **Grün** = eingeschlossen (SNR ≥ 1.0 dB)
- 🟣 **Lila** = ausgeschlossen (SNR < 1.0 dB)
- Gestrichelte Linie = SNR-Schwellwert

Plot in Plotly für konsistente Darstellung mit allen anderen Figures.

---

### Sektion 6 — Long-Format Export

Erstellt `LMM_data_long_format.csv` — kompatibel mit Python (statsmodels), R (lme4) und SPSS.

**Wichtig: HRV-Kovariaten sind methodenunabhängig**, d.h. sie werden einmalig über die HeartMovement-Sessions aggregiert (Median pro Subject × bpm_target) und dann an alle Methoden gejoint. So wird verhindert, dass methodenspezifische Signalverarbeitungsartefakte die HRV-Werte kontaminieren.

**Ausgeschlossene Spalten (`EXCLUDE_COLS`):**  
Methodenspezifische Metriken wie `ridge_power_mean`, `snr_main_over_band` etc. sind aus den HRV-Kovariaten ausgeschlossen — sie würden beim Merge NaN für andere Methoden erzeugen.

---

### Sektion 7–10 — LMM-Analyse

Siehe [Statistische Methodik](#statistische-methodik) für Details.

**Implementierte Modelle:**

| Modell-ID | Formel | Zweck |
|---|---|---|
| `lmm_E1` | `error_hz ~ C(edr_method)` | Baseline: nur Methodeneffekt |
| `lmm_E2` | `error_hz ~ C(edr_method) + bpm_target_z` | + Atemrate als Kovariate |
| `lmm_E3` | `error_hz ~ C(edr_method) + bpm_target_z + hrv_mean_rr_ms_z` | + Herzperiode |
| `lmm_E4` | `error_hz ~ C(edr_method) * bpm_target_z + hrv_mean_rr_ms_z` | Interaktion Methode × Atemrate (**bestes AIC**) |
| `lmm_Res1` | `Residualvarianz ~ C(edr_method) + hrv_mean_rr_ms_z` | Residualfehler ohne Atemrateneffekt |
| `lmm_Res2` | `Residualvarianz ~ C(edr_method) * hrv_mean_rr_ms_z` | + Interaktion Methode × Herzperiode |

**Modellvergleich:** AIC/BIC-basierte Selektion. Niedrigerer AIC = bessere Modellpassung unter Berücksichtigung der Parameterzahl.

---

### Sektion 11–13 — Paper Figures

Alle Figuren im einheitlichen Paper-Layout (`paper_layout()`, `ax_style()`), gespeichert als interaktive HTML-Dateien.

| Figure | Datei | Inhalt |
|---|---|---|
| `fig_combined` | `Fig_Estimation_Error_Combined.html` | Scatter+Reg, Mean±SD, Violin — Error in bpm |
| `fig_p2` | `Fig_SNR_vs_BreathingRate.html` | SNR vs. Atemrate, 3 Panels (1 pro Methode) |
| `fig_p3` | `Fig_ResidualError_vs_RR.html` | Residualfehler vs. Mean RR |
| `fig_p4` | `Fig_Overall_Comparison.html` | Overall Violin: Error & SNR |
| `fig_rr` | `Fig_Error_SNR_vs_RR.html` | Error & SNR vs. Mean RR (kombiniert) |
| `fig_p5` | `Fig_LMM_ForestPlot.html` | LMM Fixed Effects Forest Plot |

---

## EDR-Methoden

### 1. HeartMovement EDR (Referenzmethode)

Nutzt die mechanische Herzbeweg̈ung, die im VCG als niederfrequente Modulation sichtbar ist. Die Atemfrequenz wird aus dem Spektrum der VCG-Amplitude extrahiert.

**Stärken:** Robust gegenüber Atemratenvariation; mechanistisch unabhängig von RR-Intervall-Länge.

### 2. RR-Interval EDR

Klassische EDR-Methode: Die respiratorische Sinusarrhythmie (RSA) moduliert die RR-Intervall-Länge. Die Atemfrequenz wird aus dem Spektrum der RR-Zeitreihe extrahiert.

**Stärken:** Konzeptionell einfach, gut validiert in der Literatur.  
**Schwäche:** Stark atemratenabhängig (Interaktion mit bpm_target_z signifikant, β = −0.034, z = −6.2, p < 0.001). Bei hohen Atemraten (> 15 bpm) nimmt die Schätzgenauigkeit stark ab, da die RSA-Amplitude bei höheren Frequenzen abnimmt.

### 3. R-Amplitude EDR (VectorLength_R)

Nutzt die respiratorische Modulation der R-Zacken-Amplitude im VCG. Die Thoraxbewegung beim Atmen verändert die elektrische Achse und damit die Amplitude.

**Stärken:** Positiver Herzperioden-Effekt (höhere HR → stärkere R-Amplitude → besseres SNR).  
**Schwäche:** Moderate Atemratenabhängigkeit.

---

## Statistische Methodik

### Warum LMM?

Das Studiendesign ist **hierarchisch**: Jeder Proband liefert mehrere Messungen (7 Atemraten × 3 Methoden × mehrere Segmente). Klassische ANOVA oder OLS würden die Abhängigkeit innerhalb von Probanden ignorieren und die Standardfehler unterschätzen → zu viele falsch-positive Befunde.

**Linear Mixed Models (LMM)** berücksichtigen:
- **Fixed Effects:** Systematische Einflüsse (Methode, Atemrate, Herzperiode) — das eigentliche Ergebnis
- **Random Effects (Random Intercept):** Individuelle Baseline-Unterschiede zwischen Probanden — Kontrollvariable

Das entspricht konzeptuell einer **Messwiederholungs-ANOVA mit Kovariaten**, ist aber flexibler (erlaubt unbalancierte Designs, kontinuierliche Kovariaten, Interaktionen).

### Modellstruktur

```
Schätzfehler_ijk = β₀ + β₁·Methode_j + β₂·Atemrate_k + β₃·Methode_j×Atemrate_k
                  + β₄·MeanRR_i + u_i + ε_ijk

wobei:
  i = Subject (Random Intercept u_i ~ N(0, σ²_u))
  j = EDR-Methode (Referenz: HeartMovement)
  k = Atemratenstufe (z-standardisiert)
  ε_ijk ~ N(0, σ²) = Residualfehler
```

**Referenzkategorie:** HeartMovement — alle Methoden-Koeffizienten sind Kontraste relativ zu HeartMovement.

**Standardisierung:** `bpm_target_z` und `hrv_mean_rr_ms_z` sind z-standardisiert (Mittelwert=0, SD=1). Dadurch sind die Koeffizienten als Effekt **pro Standardabweichung** interpretierbar und numerisch stabiler.

### Ergebnisinterpretation

**LMM-E4 Hauptbefunde:**

| Term | β | Interpretation |
|---|---|---|
| Intercept | −0.012 Hz | HeartMovement unterschätzt bei mittlerer Atemrate um 0.012 Hz (≈ 0.7 bpm) |
| RR_Intervals vs. HM | −0.035 Hz | RR-Intervall-EDR schätzt **zusätzlich** 0.035 Hz zu wenig |
| bpm_target_z | −0.017 Hz/SD | Pro SD Anstieg der Atemrate: +0.017 Hz mehr Unterschätzung (alle Methoden) |
| RR × Atemrate | −0.034 Hz/SD | **Interaktion:** RR-Methode zeigt 3× stärkere Atemratenabhängigkeit als HM |
| hrv_mean_rr_ms_z | −0.008 Hz/SD | Längere RR-Intervalle → kleinerer Fehler (mehr Schläge → bessere Schätzung) |

**LMM-Res1/Res2 Hauptbefunde:**

Nach Herausrechnen des Atemfrequenzeffekts (Residualfehler):
- **Kein signifikanter Methodeneffekt** (p > 0.8) — Methodenunterschiede sind atemraten-induziert
- **Herzperiode signifikant** (β ≈ −0.007 bis −0.008, p < 0.05) — robuster physiologischer Effekt
- **Keine signifikante Interaktion** (Res2) — Herzperioden-Steigung gleich für alle Methoden

### ConvergenceWarning: Group Var ≈ 0

In allen drei Residualmodellen konvergiert die Random-Intercept-Varianz gegen Null. Das bedeutet **nicht**, dass das Modell fehlerhaft ist (`Converged: Yes` in allen Outputs). Es bedeutet:

> Nach Kontrolle der Fixed Effects (Methode, Atemrate, Herzperiode) ist **kein substanzieller systematischer Unterschied zwischen Probanden** mehr vorhanden. Die Prädiktoren erklären die Varianz vollständig auf Messebene.

**Für die Arbeit:** *"The random intercept variance converged to zero, indicating that between-subject variability in EDR estimation error was fully accounted for by the fixed-effects predictors (breathing rate, cardiac period, and method)."*

**Praktische Konsequenz:** Für die Residualmodelle wäre technisch ein einfaches OLS ausreichend. Die LMM-Struktur wird dennoch beibehalten für Konsistenz und weil `Group Var ≈ 0` selbst ein inhaltliches Ergebnis ist.

---

## SPSS-Äquivalenz

Das Long-Format CSV (`LMM_data_long_format.csv`) ist direkt in SPSS importierbar und kann dort mit dem **Linear Mixed Models**-Dialog analysiert werden.

### Import in SPSS

```
File → Import Data → CSV
Trennzeichen: Komma
Erste Zeile als Variablennamen: Ja
```

### SPSS-Äquivalent zu LMM-E4

```
Analyze → Mixed Models → Linear

Dependent: error_hz
Subject: subject_id         ← Random Factor
Repeated: —                 ← (kein Repeated, nur Random Intercept)

Fixed Effects:
  - edr_method (Faktor)
  - bpm_target_z (Kovariate)
  - edr_method * bpm_target_z (Interaktion)
  - hrv_mean_rr_ms_z (Kovariate)

Random Effects:
  - Intercept (Subjekt: subject_id)

Estimation: REML
```

### Unterschiede Python ↔ SPSS

| Aspekt | Python (statsmodels) | SPSS |
|---|---|---|
| Methode | REML | REML (Standard) |
| Referenzkategorie | Alphabetisch erste Kategorie oder `[T.X]`-Notation | Letzte Kategorie (änderbar) |
| Test-Statistik | z-Wert (asymptotisch) | t-Wert (mit Satterthwaite df) |
| Kovarianzstruktur | Nur Variance Components (Random Intercept) | Flexibel wählbar |
| AIC/BIC | Ausgegeben | Ausgegeben |
| Post-hoc Tests | Manuell (z.B. `statsmodels contrast`) | Eingebaut (Bonferroni, Tukey) |

**Wichtig bei SPSS:** Die Referenzkategorie für `edr_method` muss explizit auf `HeartMovement` gesetzt werden (letzter Level in SPSS = Referenz, oder über `Contrast → Simple, Reference = HeartMovement`), damit die Koeffizienten mit den Python-Ergebnissen vergleichbar sind.

### SPSS-Äquivalent zu LMM-Res1/Res2

Identischer Dialog, aber:
- **Dependent:** `Residualvarianz` (vorher berechnen: `error_hz` nach Regression auf `bpm_target_z` partialisieren)
- Für Res2: Interaktion `edr_method * hrv_mean_rr_ms_z` hinzufügen

### Modellvergleich in SPSS

SPSS gibt AIC und BIC in der Tabelle **Information Criteria** aus. Niedrigerer Wert = besseres Modell (bei gleichen Daten und Schätzungsmethode).

---

## Ausgabedateien

### Ergebnisse (`results_dir/`)

| Datei | Inhalt |
|---|---|
| `LMM_data_long_format.csv` | Analysedaten im Long-Format (SPSS-kompatibel) |
| `Table1_Descriptive_Stats.csv` | Deskriptive Statistik pro Methode × Atemrate |
| `Table2_LMM_ModelComparison.csv` | AIC/BIC/LogLik aller LMM-Varianten |
| `Table3_LMM_Coefficients.csv` | Fixed Effects aller Modelle mit CIs und p-Werten |

### Plots (`plot_dir/`)

| Datei | Inhalt |
|---|---|
| `AllSubjects_SNR_thresh_1.0dB.html` | SNR Balkenplot Qualitätsprüfung |
| `Fig_Estimation_Error_Combined.html` | 3-Panel Error (Scatter, Mean±SD, Violin) |
| `Fig_SNR_vs_BreathingRate.html` | SNR vs. Atemrate |
| `Fig_ResidualError_vs_RR.html` | Residualfehler vs. Mean RR |
| `Fig_Overall_Comparison.html` | Overall Violin: Error & SNR |
| `Fig_Error_SNR_vs_RR.html` | Error & SNR vs. Mean RR kombiniert |
| `Fig_LMM_ForestPlot.html` | LMM Fixed Effects Forest Plot |

---

## Abhängigkeiten

```python
# requirements
numpy >= 1.24
pandas >= 2.0
scipy >= 1.11
statsmodels >= 0.14      # MixedLM
plotly >= 5.18           # Alle Visualisierungen
```

Installation:
```bash
pip install numpy pandas scipy statsmodels plotly
```

---

## Hinweise zur Reproduzierbarkeit

- Alle Zufallsprozesse sind deterministisch (keine stochastischen Komponenten in der Pipeline)
- Segmentierung und Spektralschätzung erfolgen upstream (vor diesem Notebook)
- Die `SNR_THRESHOLD_DB`-Konstante in Sektion 1 steuert den Subject-Einschluss — Änderungen wirken sich auf alle nachfolgenden Analysen aus
- LMM-Modelle werden mit REML geschätzt; für Modellvergleiche via Likelihood Ratio Test müsste auf ML umgestellt werden (`reml=False`)

---

*Letztes Update: April 2026*
