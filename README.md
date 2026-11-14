# DFT-LSDA

Eigenständiger Dichtefunktional-Code in C++20: ebene Wellen + PAW (Projector Augmented Waves),
lokale Spindichte-Näherung (LSDA). Entwickelt für rigorose Defektbildungsenergien in den
B2-Aluminiden NiAl und FeAl, aber allgemein für periodische Kristalle einsetzbar.

## Methode

| Baustein | Umsetzung |
|---|---|
| Funktional | LSDA (Slater-Austausch + Perdew-Wang-92-Korrelation) und GGA-PBE, jeweils kollinear spinpolarisiert; Gradiententerme auf dem Gitter (White-Bird) und in den PAW-Kugeln |
| Rumpf-Valenz | PAW nach Blöchl in der Formulierung von Mortensen, Hansen, Jacobsen (PRB 71, 035109): neutrale glatte Ladung mit Kompensationsladungen, keine Ewald-Summe |
| Datensätze | PAW-XML (JTH-Tabelle v1.1, LDA-PW und GGA-PBE, skalarrelativistisch; Ni/Fe mit 3s3p-Semicore) in `pseudos/` |
| Basis | ebene Wellen; grobes FFT-Gitter für Wellenfunktionen, dichte G-Kugel für Dichte, Kompensationsladung, Hartree und XC |
| Brillouin-Zone | Monkhorst-Pack-Gitter, Reduktion mit der vollen Raumgruppe + Zeitumkehr; Dichte, PAW-Dichtematrizen und Kräfte werden symmetrisiert |
| Besetzungen | Fermi-Dirac, Gauß, Methfessel-Paxton (1. Ordnung), Marzari-Vanderbilt ("cold"), Tetraedermethode mit Blöchl-Korrektur; optional festes Gesamtmoment |
| Eigenlöser | PPCG (Projected Preconditioned CG, Vecharynski/Yang/Pask 2015) für H ψ = ε S ψ: bandweise 3×3-Rayleigh-Ritz-Schritte, Cholesky-QR, alle Matrixelemente über PAW-Projektionen ⟨p̃ᵢ\|ψ⟩; alternativ All-Band-LOBPCG. Teter-Payne-Allan-Vorkonditionierung, Toleranz an das Dichteresiduum gekoppelt, LCAO-Startwellenfunktionen |
| SCF | Pulay/Anderson-Mischung von Dichte, Magnetisierung und PAW-Dichtematrizen, Kerker-Vorkonditionierung, Hartree-Metrik |
| Kräfte / Relaxation | analytische Hellmann-Feynman-Kräfte (inkl. PAW-Terme), BFGS |
| Spannungstensor / Zellrelaxation | analytischer PAW-Spannungstensor; geglätteter kinetischer Cutoff (`ecut_smooth`), damit die Spannung die exakte Ableitung der Energie bei konstantem Cutoff ist; Quasi-Newton-Relaxation von Zelle und Atomen auf Zieldruck |
| Beschleunigung | FFTW mit ausgedünnter 3D-Transformation für Wellenfunktionen (nur Spalten/Ebenen der G-Kugel, 1,4–1,8× schneller als die volle FFT), BLAS-3 für alle Blockoperationen, OpenMP |

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

`schlüssel = wert`, Kommentare mit `#`, drei Blöcke `begin lattice|species|atoms … end`,
optional `begin hubbard … end` (DFT+U, Zeilen `<Spezies> <Schale s|p|d|f> <U_eff in eV>`)
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
| `task` | `scf` | `scf`, `forces` (SCF + Kräfte), `stress` (+ Spannungstensor), `relax` (Atome bei fester Zelle), `cellrelax` (Zelle + Atome), `neb` (Wanderungspfad), `phonons` (Γ-Phononen der Zelle), `elastic` (elastische Konstanten), `bands` (SCF + Bandstruktur), `md` (Molekulardynamik) |
| `stress` | `false` | Spannungstensor nach dem letzten SCF zusätzlich berechnen |
| `ecut_smooth` | 0 (bei `cellrelax`: 0.5) | Breite (Ha) der Glättung der kinetischen Energie am Cutoff; nötig für Spannungen/Zellrelaxation. Verschiebt die Energie leicht: in allen zu vergleichenden Rechnungen gleich wählen |
| `pressure`, `stress_tol` | 0, 0.05 GPa | Zieldruck und Konvergenzschwelle der Zellrelaxation |
| `cell_steps`, `cell_modulus`, `max_strain` | 30, 300 GPa, 0.03 | Schritte, Start-Steifigkeit und maximale Dehnung pro Schritt |
| `prefix` | `lsda` | Name der Ergebnisdatei `<prefix>.json` und des Checkpoints `<prefix>.chk` |
| `restart` | `false` | Zustand aus `<prefix>.chk` übernehmen (falls vorhanden und kompatibel): Positionen, bei `cellrelax` auch die Zelle, Dichte, Dichtematrizen, Wellenfunktionen und – bei unterbrochenem SCF – die Pulay-Historie |
| `max_seconds` | 0 (kein Limit) | Wandzeitlimit des Laufs: sauberer Abbruch mit Checkpoint, Exit-Code 4; mit `restart = true` geht es nahtlos weiter |
| `checkpoint_wfc` | `true` | Wellenfunktionen im Checkpoint speichern (groß; ohne sie startet der Neustart per LCAO) |
| `pseudo_dir` | Verzeichnis der Eingabe | Ort der PAW-Datensätze |
| `units`, `scale` | `bohr`, 1 | Einheit und Skalierungsfaktor der Gittervektoren |
| `coordinates` | `fractional` | oder `cartesian` (in `units`·`scale`) |
| `ecut` | Empfehlung der Datensätze | Cutoff der Wellenfunktionen (Ha); für NiAl/FeAl 25–30 Ha, siehe `VALIDATION.md` |
| `ecut_dense` / `dense_factor` | 8·`ecut` | Cutoff der dichten G-Kugel (≥ 4·`ecut`; ab Faktor 6 auf < 1e-6 Ha konvergiert) |
| `kpoints`, `kshift` | `1 1 1`, `0 0 0` | Monkhorst-Pack-Gitter, Verschiebung (0 oder 0.5) |
| `symmetry` | `true` | Raumgruppensymmetrie verwenden |
| `xc` | aus den Datensätzen | `pw92` (LSDA) oder `pbe`; ohne Angabe gilt das Funktional, mit dem die PAW-Datensätze erzeugt wurden (`*.LDA_PW-JTH.xml` bzw. `*.GGA_PBE-JTH.xml`) |
| `nspin` | 1 | 2 = spinpolarisiert (kollinear) |
| `noncollinear` | `false` | zweikomponentige Spinoren, Magnetisierung als Vektorfeld; Atomzeilen dann `<Label> x y z mx my mz` (Startmoment als Vektor in µB) |
| `nbands` | automatisch | Zahl der Bänder |
| `smearing`, `width` | `cold`, 0.005 Ha | `fermi`, `gauss`, `mp1`, `cold`, `tetrahedron` (ohne Breite; Kräfte und Spannungen sind damit nicht exakt konsistent zur Energie – für Relaxationen eine Verschmierung verwenden) |
| `fixed_moment` | – | festes Gesamtmoment (µB) |
| `max_scf`, `scf_tol` | 100, 1e-9 | SCF-Schritte, Schwelle für das quadrierte Dichteresiduum |
| `mix_alpha`, `mix_alpha_mag` | 0.3, 0.3 | Mischfaktoren Ladung / Magnetisierung |
| `kerker_q0`, `mix_history` | 0.8 bohr⁻¹, 8 | Kerker-Wellenzahl, Pulay-Historie |
| `eigensolver` | `ppcg` | `ppcg` oder `lobpcg` (robuster Referenzlöser, mehr dichte lineare Algebra) |
| `ppcg_block` | 8 | Blockgröße der PPCG-Rayleigh-Ritz-Schritte (1 = billigste Variante, ≥ Bandzahl ≙ LOBPCG) |
| `eig_max_iter`, `eig_max_iter_first` | 2, 6 | Eigenlöser-Schritte pro SCF-Schritt bzw. im ersten SCF-Schritt |
| `eig_tol`, `eig_tol_factor`, `eig_tol_max` | 1e-7, 0.1, 1e-3 | Residuumstoleranz = Faktor·√(Dichteresiduum²/N_el), begrenzt auf [`eig_tol`, `eig_tol_max`] |
| `paw_ntheta` | 7 | Winkelquadratur der On-site-XC-Korrektur (2n² Punkte) |
| `relax_steps`, `force_tol`, `max_step` | 50, 5e-4 Ha/bohr, 0.3 bohr | Relaxation (`force_tol`, `max_step` gelten auch für NEB) |
| `dos`, `dos_width`, `dos_points` | `false`, Verschmierungsbreite, 2001 | Zustandsdichte nach dem letzten SCF in `<prefix>.dos`: gesamt und projiziert auf die PAW-Kugeln (Atom, l, Spin); Gauß-Breite in Ha |
| `write_density` | `false` | glatte Elektronendichte (und Magnetisierung) im Cube-Format: `<prefix>.rho.cube`, `<prefix>.mag.cube` |
| `kpath_points` | 20 | Punkte je Segment des k-Pfads bei `task = bands` |
| `md_steps`, `md_dt`, `md_temperature` | 100, 2 fs, 300 K | Molekulardynamik: Schritte, Zeitschritt, Start- bzw. Zieltemperatur |
| `md_thermostat`, `md_tau` | `none`, 100 fs | `none` (NVE) oder `berendsen` mit Relaxationszeit |
| `displacement` | 0.02 bohr | Auslenkung für `task = phonons` (zentrale Differenzen) |
| `phonon_supercell`, `phonon_qgrid` | `1 1 1`, `8 8 8` | Superzelle der (primitiven) Eingabezelle für Kraftkonstanten und Dispersion; q-Gitter für `<prefix>.phmesh.json` |
| `strain` | 0.005 | Dehnungsamplitude für `task = elastic` |
| `neb_images`, `neb_spring`, `neb_climb`, `neb_steps` | 5, 0.1 Ha/bohr², `true`, 200 | NEB: Zwischenbilder, Federkonstante, Climbing Image, maximale Schritte |

## Defektbildungsenergien in B2-Verbindungen

In einer geordneten Verbindung ändert jeder Punktdefekt die Teilchenzahlen; Bildungsenergien sind
daher nur zusammen mit den chemischen Potentialen definiert. Der Workflow:

1. Gleichgewichtsgitterkonstante: `task = cellrelax` an der primitiven Zelle (oder E(V)-Fit mit
   `tools/eos.py`), mit demselben `ecut`, `ecut_smooth` und vergleichbarer k-Punkt-Dichte wie später
   in den Superzellen.
2. Superzellen mit `tools/b2_supercell.py` erzeugen: perfekt, `vac_tm`, `vac_al`, `al_tm`, `tm_al`
   (der Defekt sitzt auf einem Platz voller kubischer Symmetrie → 48 Symmetrieoperationen bleiben
   erhalten), Atompositionen relaxieren (`task = relax`).
3. `tools/defect_thermo.py` (Optionen `--p` für Drücke in GPa, `--bulk-modulus`): großkanonisches Modell nicht wechselwirkender Defekte
   (Mayer, Elsässer, Fähnle, phys. stat. sol. (b) 191, 283 (1995)) → chemische Potentiale,
   Defektkonzentrationen und effektive Bildungsenergien als Funktion von Zusammensetzung und
   Temperatur, außerdem die zusammensetzungserhaltenden Reaktionen (Tripeldefekt, Schottky-Paar,
   Antistrukturpaar) und die konstitutionellen Defekte. Mit Volumeninformation – auf Druck null
   relaxierte Zellen (`DEFECT_TASK=cellrelax`) oder Zellen festen Volumens mit Spannung (`stress = true`)
   plus `--bulk-modulus` – außerdem Bildungsvolumina und Konzentrationen bei endlichem Druck.

Weitere Optionen von `defect_thermo.py`: `--complex NAME DATEI MULTIPLIZITÄT [TEILE]` nimmt gebundene
Defektkomplexe (z. B. den gebundenen Tripeldefekt) mit eigener Superzelle als zusätzliche verdünnte
Defektart auf und gibt ihre Bindungsenergie aus. `tools/size_extrapolation.py VERZ1 VERZ2 …` extrapoliert
die Reaktionsenergien mehrerer Superzellengrößen mit E(V) = E_∞ + a/V.

`examples/defects/run_defects.sh` führt die Schritte 2–3 automatisch aus (Standard: Defektzellen bei
festem Volumen; `DEFECT_TASK=cellrelax` relaxiert zusätzlich die Zelle auf Druck null):

```sh
cd examples/defects
./run_defects.sh Ni 2.834 3 "4 4 4"          # NiAl, 54-Atom-Superzelle
./run_defects.sh Fe 2.813 3 "4 4 4" 2 1.0    # FeAl, spinpolarisiert (Startmoment 1 µB)
```

## Wanderungsbarrieren (NEB)

`task = neb` sucht den Pfad minimaler Energie zwischen zwei relaxierten Endzuständen (Nudged Elastic Band
mit verbesserten Tangenten und Climbing Image, FIRE-Optimierer). Der zweite Endzustand steht im Block
`atoms_final` (gleiche Atome in gleicher Reihenfolge; Koordinaten werden nicht in die Zelle zurückgefaltet,
der Pfad folgt also genau den angegebenen Werten):

```
task       = neb
neb_images = 5
begin atoms          # Anfangszustand, z. B. Leerstelle auf Platz 1
  ...
end
begin atoms_final    # Endzustand, z. B. Nachbaratom ist in die Leerstelle gesprungen
  ...
end
```

Ausgabe: Energien entlang des Pfads, Barriere vorwärts/rückwärts, alle Bilder in `<prefix>.neb.json`.
Es werden nur Symmetrieoperationen benutzt, die beide Endzustände gemeinsam haben. Mit `restart = true`
und `max_seconds` lässt sich auch ein NEB-Lauf in Zeitscheiben rechnen. Sitzt ein Bild per Symmetrie
auf einem Energiemaximum (z. B. das mittlere Bild eines symmetrischen, geraden Pfads), eine gerade Zahl
von Bildern wählen.

## Elastische Konstanten

`task = elastic` bestimmt C_ij (Voigt-Notation) aus der Spannungsantwort auf kleine Dehnungen ±`strain`,
mit Relaxation der Atompositionen in jeder gedehnten Zelle. Dehnungsmuster, die durch eine
Symmetrieoperation auseinander hervorgehen, werden nur einmal gerechnet (kubisch: 4 statt 12 Zellen).
Ausgabe: volle 6×6-Matrix, Kompressionsmodul, C₁₁/C₁₂/C₄₄, C′, Stabilitätsprüfung und
`<prefix>.elastic.json`. Wie bei `cellrelax` wird `ecut_smooth = 0.5` gesetzt. Scherkonstanten von
Metallen brauchen sehr dichte k-Gitter (Al: 6³ liefert Unsinn, 12³ ist brauchbar). Der Kompressionsmodul
ist der Wert für `defect_thermo.py --bulk-modulus`.

## Nichtkollinearer Magnetismus

`noncollinear = true` rechnet mit zweikomponentigen Spinoren; Dichte, Potential und PAW-Dichtematrizen
haben vier Komponenten (n, m_x, m_y, m_z), das XC-Funktional wird lokal im Spinsystem der Magnetisierung
ausgewertet. Ohne Spin-Bahn-Kopplung ist die Energie invariant unter gemeinsamer Drehung aller Momente.
Einschränkungen des jetzigen Stands: nur LSDA (kein PBE), keine Symmetrie und keine Zeitumkehr (volles
k-Gitter), keine Spin-Bahn-Kopplung, keine Zwangsbedingungen an die Momentrichtungen (sie sind frei und
können während des SCF drehen); Spannungstensor, Zustandsdichte, Bandstruktur und Wellenfunktions-Checkpoint
sind in diesem Modus nicht verfügbar. Kräfte, Relaxation, DFT+U und MPI funktionieren.

## DFT+U

```
begin hubbard
  Fe d 2.0        # Spezies, Schale, U_eff = U - J in eV
end
```

Vereinfachte rotationsinvariante Form (Dudarev): E_U = U_eff/2 Σ_σ Sp[n^σ(1 − n^σ)] mit der
Besetzungsmatrix der Schale aus den PAW-Dichtematrizen (All-Elektronen-Partialwellen in der Kugel).
Energie, Hamiltonoperator, Kräfte und Spannungen enthalten den Term konsistent; die Besetzung der Schale
wird am Ende des SCF ausgegeben. Die SCF-Konvergenz kann mit U im magnetischen Fall deutlich zäher sein.

## Phononen und Schwingungsentropie

`task = phonons` berechnet die harmonischen Γ-Punkt-Schwingungen der angegebenen (Super-)Zelle aus finiten
Auslenkungen. Ausgelenkt werden nur symmetrie-inäquivalente Atome (je 6 Kraftrechnungen); die übrigen
Kraftkonstanten folgen aus der Symmetrie. Akustische Summenregel und Symmetrie der Kraftkonstantenmatrix
werden erzwungen, die Verletzung vorher wird ausgegeben. Ergebnis: `<prefix>.phonons.json` mit den
Frequenzen (imaginäre als negative Zahlen). Fertige Auslenkungen stehen in `<prefix>.phonons.forces` und
werden bei `restart = true` wiederverwendet; die Struktur kommt dann aus dem Checkpoint der Relaxation.
Massen: Standard-Atomgewichte, im `species`-Block als dritte Spalte überschreibbar.

**Dispersion und quasiharmonische Näherung.** Mit `phonon_supercell = n n n` ist die Eingabezelle die
primitive Zelle; die Kraftkonstanten kommen aus der Superzelle (das k-Gitter wird entsprechend
verkleinert). Mit einem `kpath`-Block entsteht `<prefix>.phdisp` (Frequenzen entlang des Pfads; exakt an
den mit der Superzelle kommensurablen Wellenvektoren, dazwischen Fourier-interpoliert), außerdem
`<prefix>.phmesh.json` mit den Frequenzen auf einem q-Gitter sowie Energie und Volumen pro primitiver
Zelle. `tools/qha.py vol1.phmesh.json vol2.phmesh.json …` (mindestens vier Volumina) liefert daraus
V₀(T), B₀(T) und den thermischen Ausdehnungskoeffizienten.

Für Defekte: `PHONONS=1 ./run_defects.sh …` rechnet die Phononen aller fünf Superzellen und ruft die
Auswertung mit `--vibrations` auf. Die harmonische freie Energie der Γ-Moden jeder Superzelle geht dann in
alle Energien ein; ausgegeben werden zusätzlich die Schwingungsbeiträge und Bildungsentropien der
zusammensetzungserhaltenden Reaktionen. Voraussetzung sind gut relaxierte Strukturen und scharf
konvergierte Kräfte (`scf_tol = 1e-11`), sonst entstehen imaginäre Moden.

## Molekulardynamik

`task = md`: Born-Oppenheimer-Dynamik mit Velocity-Verlet (NVE) oder Berendsen-Thermostat, ohne Symmetrie.
Pro Schritt eine Zeile (potentielle und kinetische Energie, Erhaltungsgröße, Temperatur), Trajektorie in
`<prefix>.md.xyz` (erweitertes XYZ). Nach jedem Schritt wird `<prefix>.md.restart` geschrieben; mit
`restart = true` (und `max_seconds`) läuft die Dynamik in Zeitscheiben weiter. Massen wie bei den Phononen.

## Bandstruktur und Dichte

`task = bands` rechnet nach dem SCF die Eigenwerte entlang eines k-Pfads im selbstkonsistenten Potential:

```
begin kpath          # Label und fraktionale reziproke Koordinaten
  G 0.0 0.0 0.0
  X 0.5 0.0 0.5
  L 0.5 0.5 0.5
end
```

Ausgabe `<prefix>.bands` (Pfadlänge, Bandenergien relativ zu E_F in eV, bei Spinpolarisation erst alle
Bänder für Spin auf, dann für Spin ab). Pfadpunkte, die auch im SCF-k-Gitter liegen, werden automatisch
gegen die SCF-Eigenwerte geprüft. `write_density = true` schreibt die glatte Dichte (Valenz + Pseudo-Rumpf,
ohne den Anteil in den PAW-Kugeln) als Cube-Datei, z. B. für VESTA.

## Zustandsdichte

`dos = true` schreibt `<prefix>.dos`: Energie relativ zur Fermi-Energie (eV), Gesamtzustandsdichte und die
Anteile in den PAW-Kugeln je Atom, Drehimpuls und Spin (Zustände/eV). Die obersten, unbesetzten Bänder
werden vom Eigenlöser nicht auskonvergiert; für die Zustandsdichte oberhalb von E_F `nbands` erhöhen.

## LDA oder PBE

Das Funktional folgt den Datensätzen im `species`-Block; für PBE also `Ni.GGA_PBE-JTH.xml` usw. eintragen.
Im Defekt-Workflow: `XC=pbe ./run_defects.sh Ni <a> 2 "4 4 4"` (eigene Gitterkonstante für PBE bestimmen,
z. B. mit `task = cellrelax`). PBE kostet auf dem Gitter und in den PAW-Kugeln etwas mehr als LDA.

## Rechnen in Zeitscheiben (ohne Server)

Mit `max_seconds` lässt sich jede Rechnung in kurze Stücke teilen; jeder Aufruf setzt dort fort, wo der
letzte aufgehört hat:

```sh
cd examples/defects
MAX_SECONDS=300 ECUT=20 ./run_defects.sh Ni 2.840 2 "4 4 4"   # so oft aufrufen, bis thermo.txt erscheint
```

Für eine einzelne Eingabe: `max_seconds = 300` und `restart = true` setzen und `lsda` wiederholt starten,
bis der Exit-Code nicht mehr 4 ist.

## MPI über k-Punkte

```sh
cmake -S . -B build-mpi -DLSDA_MPI=ON && cmake --build build-mpi     # benötigt eine MPI-Bibliothek
export OMP_NUM_THREADS=4
mpirun -np 8 build-mpi/lsda input.in                                  # 8 Prozesse × 4 Threads
```

Die k-Punkte werden auf die Prozesse verteilt (k-Punkt i gehört Prozess i mod N); jeder Prozess hält nur
die Wellenfunktionen seiner k-Punkte, alles Übrige (Dichte, Potentiale, Dichtematrizen) ist repliziert.
Sinnvoll sind höchstens so viele Prozesse wie irreduzible k-Punkte; der replizierte Anteil (Potential,
On-site-Terme, Mischung) begrenzt die Skalierung. Ausgabe und Dateien schreibt nur Prozess 0;
Checkpoints sind unabhängig von der Prozesszahl (Neustart mit anderer Prozesszahl möglich). Alle Aufgaben
(SCF, Relaxation, Zellrelaxation, NEB, Phononen, DOS) laufen damit.

**Teststand:** Auf dem Entwicklungsrechner gibt es keine MPI-Bibliothek. Geprüft ist der MPI-Zweig mit einer
Attrappe (`tests/mock_mpi/mpi.h`, `-DLSDA_MPI_MOCK=ON`), die mehrere Prozesse per `fork` und Sockets
emuliert (`LSDA_MOCK_NP=3 ./lsda input.in`). Mit einer echten MPI-Bibliothek ist der Code noch **nicht**
gelaufen – beim ersten Einsatz bitte `tests/run_regression.py` einmal unter `mpirun` gegen die seriellen
Referenzen laufen lassen (z. B. mit einem Wrapper-Skript, das `mpirun -np 2 lsda` aufruft).

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

* isoliertes Al-Atom in der Box reproduziert die All-Elektronen-Gesamtenergie des Referenzatoms (LDA und PBE);
* beide Eigenlöser reproduzieren die dichte Diagonalisierung der vollen PAW-Hamiltonmatrix;
* analytische Kräfte und Spannungen = finite Differenzen der freien Energie (spinpolarisiert, d-Elektronen);
* Rechnung mit und ohne Symmetrie liefert identische Energien;
* On-site-Hamiltonoperator = numerische Ableitung der On-site-Energie; XC-Potential = Ableitung
  der XC-Energie; Spezialfunktionen, Radialintegrale und FFT-Konventionen gegen analytische Ergebnisse;
* Gitterkonstanten und Kompressionsmoduln von Al, NiAl, FeAl, Fe, Ni gegen LDA-Literaturwerte.

## Aufbau des Quelltexts

```
src/base      Typen, Timer, lineare Algebra (BLAS/LAPACK), FFT, MPI-Schicht, Dualzahlen
src/math      Kugelflächenfunktionen, Gaunt-Koeffizienten, Bessel-Transformation, Radialgitter
src/xc        XC-Funktionale (Schnittstelle XCFunctional, LSDA-PW92)
src/paw       PAW-XML-Leser, On-site-Tensoren und -Energien
src/crystal   Kristall, Raumgruppensymmetrie, k-Punkte
src/dft       Basis, Hamiltonoperator, Eigenlöser (PPCG, LOBPCG), Besetzungen, Mischer, SCF, Kräfte, Spannung,
              Relaxation, NEB, Phononen, elastische Konstanten, Molekulardynamik
src/io        Eingabe, JSON-Ausgabe
tools         Superzellen, Zustandsgleichung, Defekt-Thermodynamik, Größen-Extrapolation, quasiharmonische Näherung
tests         Unit- und Regressionstests
```

### Erweiterungspunkte

* **Weitere Funktionale:** von `XCFunctional` ableiten (`evaluate` für lokale, `evaluate_gga` für
  gradientenkorrigierte Funktionale); Gitter- und On-site-Teil sind für beide Klassen vorhanden.
  Ableitungen lassen sich mit den Dualzahlen aus `base/dual.hpp` automatisch bilden.
* **Parallelisierung:** alle MPI-Aufrufe liegen in `base/parallel.cpp`; k-Punkt-Schleifen prüfen
  `par::owns(ik)`, Summen über k-Punkte laufen über `par::sum`. Eine Verteilung der Bänder oder der
  FFT über Prozesse gibt es nicht.
* **Spin-Bahn-Kopplung**, **GPU-FFT**: bisher nicht enthalten.
* **Weitere Elemente:** jeder PAW-XML-Datensatz mit einem Radialgitter und `sinc`-, `gauss`- oder
  `exp`-Formfunktion (z. B. die gesamte JTH-Tabelle).

## Grenzen

* Spannungen sind nur so gut konvergiert wie der Cutoff: Ni/Fe mit Semicore brauchen dafür ≥ 25–30 Ha
  (bei 20 Ha ist P(V) für NiAl noch auf ~1 GPa rau). Ohne `ecut_smooth` gilt die Spannung für die feste
  Basis und enthält nicht die Pulay-Spannung.
* Nach einem Neustart einer Relaxation beginnt die BFGS-Hesse-Matrix neu (Positionen, Dichte und Wellenfunktionen bleiben erhalten).
* MPI nur über k-Punkte (repliziertes Datenmodell) und bisher nur mit der Attrappe getestet, nicht mit
  einer echten MPI-Bibliothek.
* Schwingungsentropie der Defekte nur harmonisch und aus den Γ-Moden der Superzellen (ohne thermische
  Ausdehnung der Defektzellen).

## Lizenz und Datensätze

Code: siehe `LICENSE`. Die PAW-Datensätze stammen aus der JTH-Tabelle
(F. Jollet, M. Torrent, N. Holzwarth, Comput. Phys. Commun. 185, 1246 (2014)), bezogen über
PseudoDojo (`ATOMPAW-LDA-JTHv1.1`).
