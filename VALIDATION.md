# Verifikation und Validierung

Alle Zahlen wurden mit diesem Code auf einem Apple A18 Pro (6 Kerne, 8 GB) erzeugt.
Einheiten: Hartree (Ha), Bohr, sofern nicht anders angegeben.

## 1. Bausteintests (`build/test_unit`)

| Test | Abweichung |
|---|---|
| sphärische Bessel-Funktionen j_l, l ≤ 6, gegen geschlossene Formeln/Rekursion | 1e-15 |
| Orthonormalität der reellen Y_lm (l ≤ 6), Gaunt-Koeffizienten, Rotationsmatrizen | 1e-14 |
| Gradienten der Y_lm und Ableitung der Formfaktor-Tabellen (finite Differenzen) | < 1e-9 |
| Simpson-Quadratur, radiale Hartree-Potentiale (l = 0, 2) gegen analytische Ergebnisse | 1e-9 |
| Bessel-Transformation + Tabelleninterpolation (Gauß-Funktionen) | 3e-12 |
| PW92-Korrelationsenergie (ζ = 0: r_s = 1, 2, 5) | 4e-7 |
| v_xc = ∂e_xc/∂n_σ (finite Differenzen) | < 1e-7 |
| FFT-Konvention, verallgemeinertes Eigenwertproblem | 1e-13 |
| Symmetrie: 48 Operationen für B2/fcc, irreduzible k-Punkte (4³ → 10 bzw. 4; fcc 8³ → 29) | exakt |
| Besetzungsfunktionen: s′(x) = x f′(x) für alle vier Verschmierungen | 6e-11 |
| PAW-Datensätze: Rumpfladung, Dualität ⟨p̃ᵢ\|φ̃ⱼ⟩ = δᵢⱼ | 1e-13 (Al), 1.5e-5 (Fe/Ni, Eigenschaft der Datensätze) |
| On-site-Hamiltonoperator ΔH = ∂ΔE/∂D (finite Differenzen), Spin-Konsistenz | < 2e-7 |

| PPCG (Block 1 und 4) und LOBPCG gegen dichte Diagonalisierung der vollen PAW-Matrix (trikline Zelle, Ni + Al) | 1e-15 (Eigenwerte) |

Regressionstests (`tests/run_regression.py`, ~30 s): gespeicherte Energien/Kräfte für fcc-Al, B2-NiAl
(PPCG und LOBPCG) und verzerrtes ferromagnetisches FeAl; Relaxation einer verzerrten NiAl-Zelle findet
die Idealstruktur (Energie auf 2e-8 Ha, Position auf 5e-5) wieder; Neustart aus dem Checkpoint
reproduziert die Energie (2e-12 Ha) in 3 statt ~13 SCF-Schritten; zwei Spannungskomponenten gegen
finite Differenzen; Zellrelaxation von fcc-Al.

## 2. Isoliertes Atom gegen die All-Elektronen-Referenz

Al-Atom (sphärisch, nicht spinpolarisiert, 3s² 3p¹) in einer kubischen Box, Γ-Punkt. Die
Gesamtenergie muss die All-Elektronen-Energie des Referenzatoms im Datensatz reproduzieren
(`ae_energy total` = −241.766039 Ha); das prüft das komplette PAW-Energiefunktional
(Hartree, XC, Null-Potential, Kompensationsladungen, alle On-site-Terme).

| Box L (Bohr) | ecut (Ha) | E (Ha) | E − E_AE (mHa) |
|---|---|---|---|
| 14 | 15 | −241.763818 | +2.221 |
| 18 | 15 | −241.765573 | +0.466 |
| 24 | 15 | −241.765995 | +0.045 |
| 30 | 15 | −241.766023 | +0.017 |
| 24 | 25 | −241.766007 | +0.032 |

Der Restfehler (17 µHa bei L = 30) fällt weiter mit Boxgröße und Cutoff.

PBE (Datensatz `Al.GGA_PBE-JTH.xml`, Referenz −242.684686 Ha), ecut 15 Ha:

| Box L (Bohr) | E (Ha) | E − E_AE (mHa) |
|---|---|---|
| 16 | −242.683575 | +1.111 |
| 22 | −242.684592 | +0.094 |

Gleiche Boxkonvergenz wie bei LDA; damit sind Funktional, Gitter-Gradiententerme und die
gradientenkorrigierte On-site-Korrektur gemeinsam geprüft.

## 3. Kräfte gegen finite Differenzen

Verzerrtes B2-FeAl (Al um (0.04, 0.03, −0.02)·a verschoben), spinpolarisiert (M = 1.05 µB),
ohne Symmetrie, 3×3×3 k-Punkte. Zentrale Differenzen der freien Energie mit h = 0.01 Bohr:

| Komponente | analytisch (Ha/Bohr) | finite Differenz | Differenz |
|---|---|---|---|
| F_x(Al) | −0.01842093 | −0.01842038 | 5.5e-7 |
| F_z(Al) | +0.00961736 | +0.00961705 | 3.1e-7 |

Summe aller Kräfte: 1e-7 Ha/Bohr.

## 3a. Spannungstensor gegen finite Differenzen

Trikline NiAl-Zelle (Al verschoben), ecut 15 Ha, 2×2×2 k-Punkte, ohne Symmetrie; σ = (1/Ω) ∂F/∂ε.

*Feste Basis* (`basis_lattice`), zentrale Differenzen:

| Komponente | analytisch (Ha/Bohr³) | finite Differenz (h = 1e-3) | Differenz |
|---|---|---|---|
| σ_xx | 1.233583154e-02 | 1.233589895e-02 | 6.7e-8 (2.5e-8 bei h = 5e-4: Differenzenfehler) |
| σ_yy | 1.217169559e-02 | 1.217176649e-02 | 7.1e-8 |
| σ_zz | 1.301421636e-02 | 1.301427185e-02 | 5.6e-8 |
| σ_yz | 1.919538197e-04 | 1.919538864e-04 | 6.7e-11 |
| σ_xz | 5.601088496e-05 | 5.601096118e-05 | 7.6e-11 |
| σ_xy | 2.173427148e-05 | 2.173420079e-05 | 7.1e-11 |

Spinpolarisiertes FeAl (M = 1.35 µB): σ_xx 1.6e-8, σ_yz 1.7e-10 Abweichung. Mit/ohne Symmetrie
(NiAl kubisch): 17.30422 / 17.30437 GPa.

*Konstanter Cutoff* (Basis ändert sich mit der Zelle). Ohne Glättung weicht die Spannung der festen
Basis um die Pulay-Spannung von der Ableitung der Energie ab; mit `ecut_smooth` stimmt sie:

| System | `ecut_smooth` (Ha) | \|σ_xx(analytisch) − finite Differenz\| (Ha/Bohr³) |
|---|---|---|
| NiAl, 15 Ha (stark unterkonvergiert) | 0 | 1.6e-2 |
| NiAl, 15 Ha | 0.5 | 2.1e-5 (h = 1e-3) |
| NiAl, 15 Ha | 1.0 | 4.1e-6 (h = 1e-3); skaliert mit h², Grenzwert ≈ 1.4e-6 |
| Al, 15 Ha | 0.5 | 5.7e-8 |

*Zellrelaxation* (`task = cellrelax`, `ecut_smooth = 0.5`): fcc-Al von a = 4.08 Å → 3.990 Å in 3 Schritten
(6×6×6 k); NiAl von einer tetragonal verzerrten Zelle (2.90 × 2.929 × 2.90 Å) → kubisch, a = 2.840 Å
in 7 Schritten (4×4×4 k, 20 Ha). Bei 20 Ha ist P(V) für NiAl noch auf etwa 1 GPa rau (Ni-Semicore);
für Spannungen an Ni/Fe-Systemen daher ≥ 25–30 Ha.

## 3b. PBE

| Test | Abweichung |
|---|---|
| PBE bei verschwindendem Gradienten = LSDA; Austausch-Faktor F_x(s); Grenzfälle der Korrelation | ≤ 2e-6 |
| alle fünf partiellen Ableitungen des Funktionals (finite Differenzen) | 1e-6 |
| On-site-Hamiltonoperator = ∂ΔE/∂D mit PBE (Fe, Ni) | 5–7e-8 |
| Kräfte, verzerrtes FeAl spinpolarisiert (M = 1.03 µB): F_x, F_z gegen finite Differenzen | 9e-7 Ha/Bohr |
| Spannung NiAl (feste Basis): σ_xx / σ_yz | 4e-8 / 8e-11 Ha/Bohr³ |
| Spannung FeAl spinpolarisiert (M = 1.77 µB): σ_xx / σ_yz | 4e-8 / 1.5e-10 Ha/Bohr³ |
| Zellrelaxation fcc-Al (6×6×6 k): a = 4.048 Å (LDA mit gleichen Einstellungen: 3.990 Å) | – |

Gitterkonstanten und Momente von NiAl, FeAl, Fe und Ni mit PBE sind **nicht** gerechnet.

## 4. Symmetrie und Löser

| Vergleich (NiAl, ecut 20, 4×4×4) | freie Energie (Ha) |
|---|---|
| mit Symmetrie (10 k-Punkte) | −1760.3107867541 |
| ohne Symmetrie (36 k-Punkte) | −1760.3107867541 |
| PPCG | −1760.3107867541 |
| LOBPCG | −1760.3107867541 |

15-Atom-Superzelle (NiAl mit Ni-Leerstelle): PPCG und LOBPCG stimmen auf 5e-10 Ha überein.

## 5. Zustandsgleichungen (Birch-Murnaghan, 5 Volumina, cold smearing 0.01 Ha)

| System | a₀ (Å) | B₀ (GPa) | Moment (µB) | typische LDA-Literaturwerte¹ |
|---|---|---|---|---|
| Al (fcc), ecut 15, 14³ | 3.980 | 82.5 | – | a ≈ 3.98–3.99, B ≈ 82–84 |
| NiAl (B2), ecut 20, 12³ | 2.834 | 185 | – | a ≈ 2.83–2.84, B ≈ 185–195 |
| FeAl (B2, unmagnetisch), ecut 20, 12³ | 2.808 | 208 | – | a ≈ 2.81–2.82, B ≈ 200–210 |
| FeAl (B2, ferromagnetisch), ecut 20, 12³ | 2.813 | 194² | 0.64 | Moment ≈ 0.6–0.7 |
| Fe (bcc), ecut 20, 14³ | 2.752 | 251 | 2.04 | a ≈ 2.75–2.76, B ≈ 245–255, M ≈ 2.0 |
| Ni (fcc), ecut 20, 14³ | 3.425 | 270³ | 0.61 | a ≈ 3.42–3.43, B ≈ 250–260, M ≈ 0.6 |

¹ Größenordnungen aus All-Elektronen-LDA-Rechnungen der Literatur, aus dem Gedächtnis zitiert und
nicht einzeln nachgeschlagen; für eine Veröffentlichung bitte gegen eine Referenzrechnung
(z. B. ABINIT/GPAW mit denselben JTH-Datensätzen) prüfen. Experimentelle Gitterkonstanten liegen
wegen der bekannten LDA-Überbindung 1–3 % höher.
² Der Fit überdeckt den Übergang unmagnetisch → ferromagnetisch (bei a ≤ 2.73 Å verschwindet das Moment).
³ Bei ecut = 20 Ha ist E(V) für Ni um einige µHa verrauscht (B₀′ = 9 unphysikalisch); für Ni-haltige
Systeme ecut ≥ 25 Ha verwenden (siehe Abschnitt 6).

Kohäsionsenergie von Al relativ zum (unpolarisierten, sphärischen) Referenzatom: 4.18 eV.

## 6. Konvergenz der Abschneideparameter (B2-Zelle, 8×8×8)

Energiedifferenz ΔE = E(a₂) − E(a₁) zwischen zwei Gitterkonstanten (mHa):

| ecut (Ha) | NiAl (2.77 / 2.89 Å) | FeAl (2.75 / 2.87 Å) |
|---|---|---|
| 15 | −3.553 | −0.323 |
| 20 | −1.037 | −0.233 |
| 25 | −0.895 | −0.219 |
| 30 | −0.964 | −0.215 |
| 40 | −0.956 | −0.209 |

Die absolute Energie von NiAl konvergiert wegen der Ni-3s/3p-Semicore-Zustände langsam
(noch −8 mHa von 30 auf 40 Ha); dieser atomare Beitrag hebt sich in Reaktionen mit erhaltener
Teilchenzahl heraus. Empfehlung: **ecut = 25–30 Ha**.

Dichte G-Kugel (`dense_factor`, bei ecut = 25 Ha): Faktor 4 → 16 ändert die Energie um
3e-7 Ha (NiAl) bzw. 1.4e-6 Ha (FeAl); ab Faktor 6 < 1e-7 Ha.

## 7. SCF-Verhalten

* Mischfaktor 0.3 für Ladung und Magnetisierung konvergiert alle Testfälle in 10–25 Schritten.
  Größere Faktoren (≥ 0.4–0.5) können bei Fe-haltigen, schwach magnetischen Zellen stagnieren.
* Alle konvergierten Läufe mit unterschiedlichen Mischparametern und Lösern landen auf derselben
  Energie (Streuung < 1e-9 Ha).
* B2-FeAl hat in der LSDA eine sehr flache Energie als Funktion des Moments (ferromagnetisch nur
  ~0.25 mHa unter unmagnetisch bei a = 2.81 Å). Bei grober k-Punkt-Abtastung/breiter Verschmierung
  kann die Rechnung in die unmagnetische Lösung laufen; für FeAl feinere Gitter und
  `width ≤ 0.005` verwenden und das Moment kontrollieren (`fixed_moment` für Scans).

## 7a. Unterbrechen und Fortsetzen

* Relaxation von verzerrtem, spinpolarisiertem FeAl in drei Zeitscheiben (`max_seconds = 9`) gegen den
  ununterbrochenen Lauf: Energie 3e-11 Ha, Positionen 7e-7 Abweichung.
* SCF nach 5 Iterationen abgebrochen und fortgesetzt: gleiche Energie (< 1e-12 Ha), insgesamt 14 statt
  17 Iterationen (Pulay-Historie und Wellenfunktionen liegen im Checkpoint). Ohne gespeicherte Historie
  waren es 25.

## 7b. Zustandsdichte, NEB, Bildungsvolumina

* **Zustandsdichte** (FeAl, 4×4×4): Integral der Gesamt-DOS = Zahl der Bänder (16.9999 von 17 je Spin),
  Zustände unterhalb E_F = 18.993 (19 Elektronen); mit und ohne Symmetrie stimmen Gesamt- und projizierte
  DOS im Bereich −8 … +3 eV auf 6e-5 Zustände/eV überein, die integrierten Kugelanteile auf 2e-5.
* **NEB**: Der Band-Optimierer findet auf der Müller-Brown-Fläche den Sattelpunkt zwischen zwei Minima
  (Energie auf 2e-6, Position auf 1e-5, 148 Schritte, 7 Bilder + Climbing Image). Die DFT-Anbindung ist nur
  als Durchlauf getestet (gleiche Endpunktenergien, sinkende Pfadenergie, Neustart setzt an den
  gespeicherten Bildpositionen fort); eine echte Wanderungsbarriere wurde **nicht** gerechnet.
* **Bildungsvolumina** (`defect_thermo.py`, synthetische Daten): effektives Bildungsvolumen des
  Tripeldefekts = V_triple/3; relaxierte Zellen und der Weg über Restdruck + Kompressionsmodul liefern
  identische Konzentrationen und Volumina.

## 7c. Phononen und Schwingungsentropie

* **Symmetriereduktion**: tetragonale Zelle mit zwei äquivalenten Al-Atomen – 6 Auslenkungen mit Symmetrie
  gegen 12 ohne: optische Frequenzen 7.137 / 7.367 / 7.367 THz gegen 7.137 / 7.364 / 7.365 THz. Die Abweichung
  (3e-3 THz) ist das Kraftrauschen der Rechnung ohne Symmetrie (dort Asymmetrie der Kraftkonstanten 2e-5,
  mit Symmetrie 2e-19 Ha/Bohr²).
* **Kubisches NiAl** (Regressionstest): drei akustische Moden bei null, optische Mode (8.71 THz bei 15 Ha,
  3×3×3 k) dreifach entartet bis auf das Kraftrauschen der drei unabhängigen Auslenkungsrichtungen (3e-3 THz).
* **Thermodynamik-Werkzeug** mit `--vibrations` (synthetische Einstein-Spektren): Schwingungsbeitrag und
  Bildungsentropie von Tripeldefekt und Schottky-Paar stimmen mit der analytischen Rechnung überein
  (−1.3817 eV / 15.544 k_B bzw. −0.6819 eV / 7.613 k_B bei 1000 K).
* Der Workflow mit `PHONONS=1` ist nur als Durchlauf mit Minimalzellen getestet; Phononen echter
  Defektzellen wurden **nicht** gerechnet.

Dabei behobene Schwäche der SCF-Schleife: Residuenänderungen bei unveränderter Eingangsdichte (z. B. direkt
nach einem Neustart, wenn der Eigenlöser nur die Bänder verfeinert) gehen nicht mehr in die Pulay-Historie
ein, und die schärfste Eigenlöser-Toleranz richtet sich nach `scf_tol`.

## 8. Laufzeit (Laptop, nur zur Orientierung)

NiAl-Superzelle mit Ni-Leerstelle: 15 Atome, 150 Elektronen, 94 Bänder, ecut 20 Ha (5250 ebene Wellen),
10 irreduzible k-Punkte, 18 SCF-Schritte bis |Residuum|² < 1e-9:

| Löser | Wandzeit |
|---|---|
| LOBPCG, 4 Schritte/SCF-Schritt, Zufallsstart (erste Version) | 142 s |
| LOBPCG, 2 Schritte/SCF-Schritt | 100 s |
| PPCG + LCAO-Start (Standard) | 65–80 s |

Im PPCG-Lauf entfallen etwa 40 % auf die FFTs und 45 % auf dichte lineare Algebra; bei größeren Zellen
wächst der Anteil der linearen Algebra (O(N³)). Diese Zeiten stammen von vor der ausgedünnten FFT.

Ausgedünnte FFT (nur die 1D-Transformationen, die zur G-Kugel beitragen), Einzeltransformation inklusive
Füllen des Gitters, ein Thread:

| Gitter | volle 3D-FFT | ausgedünnt | Faktor |
|---|---|---|---|
| 45³ | 0.75 ms | 0.55 ms | 1.36 |
| 49³ | 0.99 ms | 0.55 ms | 1.79 |
| 70³ | 2.98 ms | 1.78 ms | 1.68 |
| 72³ | 2.86 ms | 1.77 ms | 1.62 |

Der PPCG-Löser fasst seither Orthonormalisierung und Rayleigh-Ritz zu einem verallgemeinerten
Eigenproblem zusammen, verwendet die Projektionen ⟨p̃|ψ⟩ des vorigen SCF-Schritts wieder und spart die
zweite Residuenberechnung; nach Zählung der Matrixprodukte sind das etwa 25 % weniger dichte lineare
Algebra pro SCF-Schritt. Die Ergebnisse sind unverändert (Regressionstests), die Laufzeit ist **nicht**
neu gemessen.

Im Programm sank die FFT-Zeit pro Hamilton-Anwendung auf der 15-Atom-Zelle von 32 auf 24 ms; der Wert
ist wegen thermischer Drosselung des Laptops und unterschiedlicher Bandzahlen pro Aufruf nur grob vergleichbar.

## 9. Noch nicht gerechnet

Die Produktionsrechnungen für die Defektenergien (relaxierte 16-/54-/128-Atom-Superzellen für NiAl und
FeAl) wurden auf dem Entwicklungsrechner **nicht** durchgeführt. Getestet ist die Kette bis dorthin:
SCF und Kräfte für eine 15-Atom-Leerstellenzelle, die Relaxation (kleine Zelle), der Workflow
`run_defects.sh` als Durchlauftest mit einer Minimalzelle (physikalisch ohne Aussage) und das
Thermodynamik-Modell an synthetischen Daten mit bekannten analytischen Grenzfällen
(Tripeldefekt: E_eff = E_triple/3; konstitutionelle Leerstellen c_V = (1 − 2x)/(1 − x)).
Ebenfalls offen: Konvergenz der Defektenergien mit der Superzellengröße und ein Vergleich mit einem
Referenzcode (z. B. ABINIT mit denselben JTH-Datensätzen).
