# DFT-LSDA

Eigenständiger Dichtefunktional-Code in C++20: ebene Wellen + PAW (Projector Augmented Waves),
lokale Spindichte-Näherung (LSDA). Entwickelt für rigorose Defektbildungsenergien in den
B2-Aluminiden NiAl und FeAl, aber allgemein für periodische Kristalle einsetzbar.

## Methode

| Baustein | Umsetzung |
|---|---|
| Funktional | LSDA: Slater-Austausch + Perdew-Wang-92-Korrelation, kollinearer Spin |
| Rumpf-Valenz | PAW nach Blöchl in der Formulierung von Mortensen, Hansen, Jacobsen (PRB 71, 035109): neutrale glatte Ladung mit Kompensationsladungen, keine Ewald-Summe |
| Datensätze | PAW-XML (JTH-Tabelle v1.1, LDA-PW, skalarrelativistisch; Ni/Fe mit 3s3p-Semicore) in `pseudos/` |
| Basis | ebene Wellen; grobes FFT-Gitter für Wellenfunktionen, dichte G-Kugel für Dichte, Kompensationsladung, Hartree und XC |
| Brillouin-Zone | Monkhorst-Pack-Gitter, Reduktion mit der vollen Raumgruppe + Zeitumkehr; Dichte, PAW-Dichtematrizen und Kräfte werden symmetrisiert |
| Besetzungen | Fermi-Dirac, Gauß, Methfessel-Paxton (1. Ordnung), Marzari-Vanderbilt ("cold"); optional festes Gesamtmoment |
| Eigenlöser | PPCG (Projected Preconditioned CG, Vecharynski/Yang/Pask 2015) für H ψ = ε S ψ: bandweise 3×3-Rayleigh-Ritz-Schritte, Cholesky-QR, alle Matrixelemente über PAW-Projektionen ⟨p̃ᵢ\|ψ⟩; alternativ All-Band-LOBPCG. Teter-Payne-Allan-Vorkonditionierung, Toleranz an das Dichteresiduum gekoppelt, LCAO-Startwellenfunktionen |
| SCF | Pulay/Anderson-Mischung von Dichte, Magnetisierung und PAW-Dichtematrizen, Kerker-Vorkonditionierung, Hartree-Metrik |
| Kräfte / Relaxation | analytische Hellmann-Feynman-Kräfte (inkl. PAW-Terme), BFGS |
| Spannungstensor / Zellrelaxation | analytischer PAW-Spannungstensor; geglätteter kinetischer Cutoff (`ecut_smooth`), damit die Spannung die exakte Ableitung der Energie bei konstantem Cutoff ist; Quasi-Newton-Relaxation von Zelle und Atomen auf Zieldruck |
| Beschleunigung | FFTW (ein Plan, threadsichere Ausführung pro Band), BLAS-3 für alle Blockoperationen, OpenMP |

Energien sind absolute Frozen-Core-Gesamtenergien (inklusive Rumpfbeiträgen der Datensätze),
alle internen Größen in Hartree-Atomeinheiten.

## Bauen

Voraussetzungen: C++20-Compiler mit OpenMP, CMake ≥ 3.20, FFTW3, LAPACK/BLAS (Accelerate, OpenBLAS oder MKL).

```sh
# macOS (Homebrew): brew install gcc fftw cmake ninja
cmake --preset macos-gcc        # bzw. --preset linux
cmake --build build
ctest --test-dir build          # Unit- und Regressionstests
```

## Benutzung

```sh
cd examples/NiAl_B2
../../build/lsda NiAl.in        # Log auf stdout, Ergebnis in NiAl.json
```

Die Zahl der Threads wird über `OMP_NUM_THREADS` gesteuert.

### Eingabedatei

`schlüssel = wert`, Kommentare mit `#`, drei Blöcke `begin lattice|species|atoms … end`
(optional `begin basis_lattice … end`: Referenzgitter, das die Ebene-Wellen-Basis festlegt – für
Rechnungen bei fester Basis, z. B. Dehnungstests).

```
task       = relax            # scf | forces | relax
prefix     = NiAl_vac
pseudo_dir = ../../pseudos
ecut       = 20               # Ha
kpoints    = 4 4 4
units      = angstrom         # Einheit der Gittervektoren (bohr | angstrom)
begin lattice
  5.668 0 0
  0 5.668 0
  0 0 5.668
end
begin species
  Ni Ni.LDA_PW-JTH.xml
  Al Al.LDA_PW-JTH.xml
end
begin atoms                   # Label, fraktionale Koordinaten, optional Startmoment (µB)
  Ni 0.0 0.0 0.0
  ...
end
```

| Schlüssel | Standard | Bedeutung |
|---|---|---|
| `task` | `scf` | `scf`, `forces` (SCF + Kräfte), `stress` (+ Spannungstensor), `relax` (Atome bei fester Zelle), `cellrelax` (Zelle + Atome) |
| `stress` | `false` | Spannungstensor nach dem letzten SCF zusätzlich berechnen |
| `ecut_smooth` | 0 (bei `cellrelax`: 0.5) | Breite (Ha) der Glättung der kinetischen Energie am Cutoff; nötig für Spannungen/Zellrelaxation. Verschiebt die Energie leicht: in allen zu vergleichenden Rechnungen gleich wählen |
| `pressure`, `stress_tol` | 0, 0.05 GPa | Zieldruck und Konvergenzschwelle der Zellrelaxation |
| `cell_steps`, `cell_modulus`, `max_strain` | 30, 300 GPa, 0.03 | Schritte, Start-Steifigkeit und maximale Dehnung pro Schritt |
| `prefix` | `lsda` | Name der Ergebnisdatei `<prefix>.json` und des Checkpoints `<prefix>.chk` |
| `restart` | `false` | Positionen, Dichte und PAW-Dichtematrizen aus `<prefix>.chk` übernehmen (falls vorhanden und kompatibel) |
| `pseudo_dir` | Verzeichnis der Eingabe | Ort der PAW-Datensätze |
| `units`, `scale` | `bohr`, 1 | Einheit und Skalierungsfaktor der Gittervektoren |
| `coordinates` | `fractional` | oder `cartesian` (in `units`·`scale`) |
| `ecut` | Empfehlung der Datensätze | Cutoff der Wellenfunktionen (Ha); für NiAl/FeAl 25–30 Ha, siehe `VALIDATION.md` |
| `ecut_dense` / `dense_factor` | 8·`ecut` | Cutoff der dichten G-Kugel (≥ 4·`ecut`; ab Faktor 6 auf < 1e-6 Ha konvergiert) |
| `kpoints`, `kshift` | `1 1 1`, `0 0 0` | Monkhorst-Pack-Gitter, Verschiebung (0 oder 0.5) |
| `symmetry` | `true` | Raumgruppensymmetrie verwenden |
| `nspin` | 1 | 2 = spinpolarisiert |
| `nbands` | automatisch | Zahl der Bänder |
| `smearing`, `width` | `cold`, 0.005 Ha | `fermi`, `gauss`, `mp1`, `cold` |
| `fixed_moment` | – | festes Gesamtmoment (µB) |
| `max_scf`, `scf_tol` | 100, 1e-9 | SCF-Schritte, Schwelle für das quadrierte Dichteresiduum |
| `mix_alpha`, `mix_alpha_mag` | 0.3, 0.3 | Mischfaktoren Ladung / Magnetisierung |
| `kerker_q0`, `mix_history` | 0.8 bohr⁻¹, 8 | Kerker-Wellenzahl, Pulay-Historie |
| `eigensolver` | `ppcg` | `ppcg` oder `lobpcg` (robuster Referenzlöser, mehr dichte lineare Algebra) |
| `ppcg_block` | 8 | Blockgröße der PPCG-Rayleigh-Ritz-Schritte (1 = billigste Variante, ≥ Bandzahl ≙ LOBPCG) |
| `eig_max_iter`, `eig_max_iter_first` | 2, 6 | Eigenlöser-Schritte pro SCF-Schritt bzw. im ersten SCF-Schritt |
| `eig_tol`, `eig_tol_factor`, `eig_tol_max` | 1e-7, 0.1, 1e-3 | Residuumstoleranz = Faktor·√(Dichteresiduum²/N_el), begrenzt auf [`eig_tol`, `eig_tol_max`] |
| `paw_ntheta` | 7 | Winkelquadratur der On-site-XC-Korrektur (2n² Punkte) |
| `relax_steps`, `force_tol`, `max_step` | 50, 5e-4 Ha/bohr, 0.3 bohr | Relaxation |

## Defektbildungsenergien in B2-Verbindungen

In einer geordneten Verbindung ändert jeder Punktdefekt die Teilchenzahlen; Bildungsenergien sind
daher nur zusammen mit den chemischen Potentialen definiert. Der Workflow:

1. Gleichgewichtsgitterkonstante: `task = cellrelax` an der primitiven Zelle (oder E(V)-Fit mit
   `tools/eos.py`), mit demselben `ecut`, `ecut_smooth` und vergleichbarer k-Punkt-Dichte wie später
   in den Superzellen.
2. Superzellen mit `tools/b2_supercell.py` erzeugen: perfekt, `vac_tm`, `vac_al`, `al_tm`, `tm_al`
   (der Defekt sitzt auf einem Platz voller kubischer Symmetrie → 48 Symmetrieoperationen bleiben
   erhalten), Atompositionen relaxieren (`task = relax`).
3. `tools/defect_thermo.py`: großkanonisches Modell nicht wechselwirkender Defekte
   (Mayer, Elsässer, Fähnle, phys. stat. sol. (b) 191, 283 (1995)) → chemische Potentiale,
   Defektkonzentrationen und effektive Bildungsenergien als Funktion von Zusammensetzung und
   Temperatur, außerdem die zusammensetzungserhaltenden Reaktionen (Tripeldefekt, Schottky-Paar,
   Antistrukturpaar) und die konstitutionellen Defekte.

`examples/defects/run_defects.sh` führt die Schritte 2–3 automatisch aus (Standard: Defektzellen bei
festem Volumen; `DEFECT_TASK=cellrelax` relaxiert zusätzlich die Zelle auf Druck null):

```sh
cd examples/defects
./run_defects.sh Ni 2.834 3 "4 4 4"          # NiAl, 54-Atom-Superzelle
./run_defects.sh Fe 2.813 3 "4 4 4" 2 1.0    # FeAl, spinpolarisiert (Startmoment 1 µB)
```

## Rechnen auf einem Server

```sh
cmake --preset linux && cmake --build build && ctest --test-dir build   # ~1 min, muss fehlerfrei sein
export OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores
cd examples/defects
ECUT=25 ./run_defects.sh Ni 2.834 3 "4 4 4"                             # 54-Atom-Zellen, 5 Rechnungen
```

* BLAS/LAPACK sollte dieselbe Thread-Laufzeit wie der Code nutzen (OpenBLAS mit OpenMP oder MKL mit
  `MKL_THREADING_LAYER=GNU`); die FFTs werden pro Band über OpenMP verteilt.
* Speicher: die Ausgabe nennt den Bedarf der Wellenfunktionen (16 Byte × ebene Wellen × Bänder ×
  k-Punkte × Spins); der Eigenlöser braucht pro k-Punkt zusätzlich etwa das Sechsfache eines
  einzelnen Wellenfunktionsblocks. Richtwert NiAl, 54 Atome, 25 Ha, 10 k-Punkte: ca. 3 GB.
* Abgebrochene Läufe: `restart = true` setzt mit den letzten Positionen und der letzten Dichte fort
  (bei Relaxationen wird nach jedem Schritt ein Checkpoint und `<prefix>.json` geschrieben);
  `run_defects.sh` überspringt fertige Rechnungen und setzt unfertige fort.
* Für belastbare Defektenergien zu prüfen: Superzellengröße (2³ → 3³ → 4³), k-Punkte, `ecut`
  (25 → 30 Ha), Verschmierung; für FeAl der magnetische Zustand (`nspin = 2`, Startmomente).

## Verifikation

Siehe `VALIDATION.md` für die Zahlen. Kurzfassung der Prüfungen, die der Code besteht:

* isoliertes Al-Atom in der Box reproduziert die All-Elektronen-Gesamtenergie des Referenzatoms;
* beide Eigenlöser reproduzieren die dichte Diagonalisierung der vollen PAW-Hamiltonmatrix;
* analytische Kräfte und Spannungen = finite Differenzen der freien Energie (spinpolarisiert, d-Elektronen);
* Rechnung mit und ohne Symmetrie liefert identische Energien;
* On-site-Hamiltonoperator = numerische Ableitung der On-site-Energie; XC-Potential = Ableitung
  der XC-Energie; Spezialfunktionen, Radialintegrale und FFT-Konventionen gegen analytische Ergebnisse;
* Gitterkonstanten und Kompressionsmoduln von Al, NiAl, FeAl, Fe, Ni gegen LDA-Literaturwerte.

## Aufbau des Quelltexts

```
src/base      Typen, Timer, lineare Algebra (BLAS/LAPACK), FFT
src/math      Kugelflächenfunktionen, Gaunt-Koeffizienten, Bessel-Transformation, Radialgitter
src/xc        XC-Funktionale (Schnittstelle XCFunctional, LSDA-PW92)
src/paw       PAW-XML-Leser, On-site-Tensoren und -Energien
src/crystal   Kristall, Raumgruppensymmetrie, k-Punkte
src/dft       Basis, Hamiltonoperator, Eigenlöser (PPCG, LOBPCG), Besetzungen, Mischer, SCF, Kräfte, Relaxation
src/io        Eingabe, JSON-Ausgabe
tools         Superzellen, Zustandsgleichung, Defekt-Thermodynamik
tests         Unit- und Regressionstests
```

### Erweiterungspunkte

* **Weitere Funktionale (GGA):** von `XCFunctional` ableiten; für Gradienten sind zusätzlich die
  Gitter-Ableitungen in `Calculation::evaluate` und die radiale GGA-Korrektur in `PawSetup::onsite` nötig.
* **MPI über k-Punkte:** die Schleife in `Calculation::solve_bands` ist pro k-Punkt unabhängig.
* **Spin-Bahn-Kopplung**, **GPU-FFT**: bisher nicht enthalten.
* **Weitere Elemente:** jeder PAW-XML-Datensatz mit einem Radialgitter und `sinc`-, `gauss`- oder
  `exp`-Formfunktion (z. B. die gesamte JTH-Tabelle).

## Grenzen

* Spannungen sind nur so gut konvergiert wie der Cutoff: Ni/Fe mit Semicore brauchen dafür ≥ 25–30 Ha
  (bei 20 Ha ist P(V) für NiAl noch auf ~1 GPa rau). Ohne `ecut_smooth` gilt die Spannung für die feste
  Basis und enthält nicht die Pulay-Spannung.
* Neustart übernimmt Dichte und Positionen, aber keine Wellenfunktionen (die werden per LCAO neu erzeugt; ~3 SCF-Schritte).
* Parallelisierung nur über Threads (OpenMP + Thread-BLAS).
* Die Defekt-Thermodynamik berücksichtigt nur Konfigurationsentropie (keine Bildungsvolumina,
  keine Schwingungsentropie).

## Lizenz und Datensätze

Code: siehe `LICENSE`. Die PAW-Datensätze stammen aus der JTH-Tabelle
(F. Jollet, M. Torrent, N. Holzwarth, Comput. Phys. Commun. 185, 1246 (2014)), bezogen über
PseudoDojo (`ATOMPAW-LDA-JTHv1.1`).
