# MaXXXLab Jr XRD Work-up App (xrd-app-v1.1)

A Python web app for lab Cu Kα powder patterns, with four working tabs and a Help tab:

- **Raw pattern**: the data exactly as recorded, with linear, log or √ intensity.
- **Peak indexing**: refits the background and fits the peaks. It then matches the reflections of the phases you list, refines every cell plus one angle correction together (with esds), checks Cu Kβ copies and runs line tests for suspected phases. Results export as a peak table, the refined parameters, the pattern and background, and a report.

- **Le Bail** (optional, needs rietx): a whole-pattern fit started from the refined phase list, with cells, profile widths and the angle correction refined against all the data. It shows the fit, the difference curve and a table comparing every Le Bail cell with the peak-position cell, and exports the fit, the parameters and the reflections.
- **Export figures**: publication figures of the raw pattern, the indexed pattern or the Le Bail fit, with a live preview. You set the width, height, font and font size, the thickness of the curves and the frame, the marker size, and the heights of the Δ (residual) and tick-row (peak position) panels relative to the main panel. The legend, Rwp and phase names can sit at the right or be left out. Download as PDF, SVG or PNG (300, 600 or 1200 dpi); the layout can be saved as JSON.

Phase edits do not refit at once. Edit the list (or add phases from the library, the Materials Project or AIdex-r2), then click **Refine**. Switch on **Refine automatically** to refit after every edit instead.

- **00l and hk0 series**: any phase can be restricted to its 00l lines (stacking period only), its hk0 lines (in-plane cell only, matched to broad turbostratic bands within half their width) or both, which suits layered and misfit films.
- **Candidate phases from CIF files** (needs pymatgen): upload .cif structures of phases you suspect (ICSD, COD, your own). Each is screened against the pattern like a Materials Project entry, and can be added to the phase list with its cell refined to the peaks or as in the CIF.
- **Unit cell from AIdex-r2** (hidden until you switch on **Show the AIdex-r2 unit-cell tool**): the app prepares the unindexed lines for AIdex-r2, reads its pasted results and checks every returned cell against the peaks.
- **Candidate phases from the Materials Project** (optional): you give the elements likely present. The app finds every experimentally made Materials Project entry in those chemical systems, tests each structure against your peaks, and lets you add the plausible ones to the phase list.

The indexing is the same method as the XRD Workup's Part A. On the reference pattern it gives VSe₂ a = 3.3592(3), c = 6.0980(5) Å and SnSe₂ a = 3.8103(3), c = 6.1321(6) Å.

## Run it on your own computer

You need Python 3.11 or newer. Use 3.12 or 3.13 if you can: the Materials Project search needs 3.12. Get it from https://www.python.org/downloads/; on Windows, tick "Add python.exe to PATH".

**1. Set up (once).** This creates a virtual environment `.venv` inside this folder and installs every package the app needs into it. Nothing is installed system-wide.

| System | What to run |
|---|---|
| Windows | double-click `setup.bat` |
| macOS | double-click `setup.command`, or in Terminal: `./setup.sh` |
| Linux | `./setup.sh` |

The setup script:
- finds the newest Python 3.11+ on the computer;
- creates `.venv`;
- installs the core packages (`requirements.txt`), then the optional Materials Project/CIF packages (`requirements-mp.txt`) and rietx for the Le Bail tab (`requirements-lebail.txt`);
- checks that each part imports, and prints a summary.

If an optional part fails, the app still runs and says what is missing. Setup takes a few minutes and about 1.2 GB of disk space.

Options, after the script name (for example `./setup.sh --core-only`):
- `--core-only`: only the indexing and figures (about 300 MB);
- `--no-mp`, `--no-lebail`: leave out one optional part;
- `--dev`: also install pytest and run the tests;
- `--recreate`: delete `.venv` and start again, for example after a Python upgrade.

To choose the Python yourself: `PYTHON=/path/to/python3.12 ./setup.sh`, or run `python3.12 install.py` directly. `install.py` uses only the standard library.

**2. Start the app (every time).** Double-click `run.bat` (Windows) or `run.command` (macOS), or run `./run.sh`. A browser tab opens at http://localhost:8501. Stop the app with Ctrl+C in that window.

The command line uses the same environment: `.venv/bin/python cli.py ...`, or `.venv\Scripts\python cli.py ...` on Windows.

On Debian/Ubuntu, if creating the environment fails, install the venv module first: `sudo apt install python3-venv` (or `python3.12-venv`).

Without the setup script you can also install the packages yourself: `python3 -m pip install -r requirements.txt`, then `streamlit run app.py`. It was tested with Python 3.11 and 3.13, numpy 2.5, scipy 1.18, pandas 3.0, matplotlib 3.11, plotly 7.1, streamlit 1.66, pymatgen 2026.10, mp-api 0.47 and rietx 1.10.

## 00l and hk0 series

Every phase has a **Reflections** setting:

- **all**
- **00l**: only h = k = 0
- **hk0**: only l = 0
- **00l+hk0**

The axes a restricted phase cannot see (a for a 00l series, c for an hk0 series) are held and reported as "not determined". The hk0 lines of an hk0 series may match broad peaks within half their width, with reduced weight in the cell fit. With reflections left on "auto", the index range follows the cell, so a 24 Å stacking period gets all its 00l orders.

There are three ways to add one:

- **Add a 00l or hk0 series** (under the phase table): a 00l superlattice from c alone, or a hexagonal, tetragonal or rectangular hk0 series from a (and b).
- **Library phases**: choose the reflections before **Add phase**.
- **Materials Project candidates**: added with the family they matched in, or one you choose.

On a synthetic misfit film (sharp 00l superlattice c = 12.228 Å, broad hexagonal hk0 bands a = 3.442 Å, broad tetragonal hk0 bands a = 6.055 Å, displacement −0.05°), the three series and the displacement come back within their esds or within 0.001–0.006 Å.

For patterns without a prior refinement model, a peak whose fitted width runs into its fit window is refitted in a window sized to the band. Patterns with a prior model are fitted exactly as before.

## Unit cell from AIdex-r2

In the Peak indexing tab, under **Unit cell from AIdex-r2**, switch on **Show the AIdex-r2 unit-cell tool** and choose the lines to send (sharp, unindexed, at most 20, one phase). Then use either option:

- **Option A: Run AIdex-r2.** The app sends the lines straight to the AIdex-r2 web service. It makes the same request as the page's **Predict** button, with unused peak boxes set to 180°. Only the line positions are sent. This needs internet access from the computer running the app.
- **Option B: fill AIdex-r2 in your browser.** Use this when the app runs on a computer without internet access.
  1. Click **Open AIdex-r2** and open the browser console on that page (F12, then the Console tab).
  2. Paste the one-line script the app shows and press Enter. All 20 peak boxes fill at once.
  3. Click **Predict**.
  4. Copy AIdex-r2's results table back into the app and click **Check the cells**. The extinction-group numbers the page shows (1–101) are read, including their lattice centering.

Either way, the five cells are checked against the lines in the same way: refined, misfit, line density, de Wolff M(n), and a chance test. **Add selected cells** puts a cell in the phase list.

## Materials Project candidate search (optional)

```bash
python3 -m pip install -r requirements-mp.txt     # adds mp-api and pymatgen
```

In the Peak indexing tab, under **Candidate phases from the Materials Project**:

1. Enter the elements likely present, for example `Sn V Se O`. Include O if the film saw air, and the elements of transport agents, precursors or the substrate if they could have reacted.
2. Enter your Materials Project API key. It is free from next-gen.materialsproject.org: your profile, then API key. It is kept only for the browser session. Whoever runs the server can instead set `MP_API_KEY` (an environment variable, or `.streamlit/secrets.toml`) so that users don't need their own.
3. Click **Search**. Every chemical system of the elements is searched (each element alone, every pair, and so on), and only entries marked as experimentally made are kept. Solid phases of gaseous elements (O₂, N₂, Cl₂, …) are skipped.

For Sn, V, Se and O this gives 102 entries in about 4 s, and the screen takes about 25 s.

**Intensity models to screen** sets how candidates are tested:

| Model | Use for |
|---|---|
| random powder (default) | all reflections |
| 00l only (default) | an oriented film or a stacking series |
| hk0 only | in-plane bands |
| 00l + hk0 | an oriented film that also shows its in-plane bands |

In a restricted model, only the axes the family can see are scanned and refined. hk0 lines match broad bands within half their width (a band may also hide within 0.4° of a sharp line), and band intensities are compared as integrated areas.

A restricted-model match is required to explain more lines before it counts:

- 00l and 00l + hk0: three lines, because 00l series coincide easily
- hk0: two lines, because an in-plane family has only a few lines in range

When random lines are matched as often, a restricted-model match with a plausible cell is reported as **possible, check**.

On a synthetic film whose rock-salt (tetragonal) bands are left out of the list:

- A rock-salt candidate with a DFT-like cell (+1 %) explains them.
- The same structure 6 % too large is only "possible, check".
- With VSe₂'s real calculated in-plane intensities, a single detectable band is (correctly) not enough to identify it.

Each structure is tested against your peaks in five steps:

1. **Cell scan.** Each cell length is scanned from 0.93 to 1.02 of its DFT value, because DFT cells are too large by 1–6 % (most across van der Waals gaps).
2. **Refinement.** The cell is refined to the matching peaks.
3. **Coverage.** The app measures how much of the structure's predicted strong-line intensity is observed.
4. **Missing lines.** A predicted line that should be visible but has no peak is tested with a fixed-position line fit. It counts against the phase only when it is clearly below the prediction, more than 3σ and more than 5× weaker. One weak line is tolerated, because films are often textured.
5. **Chance test.** The structure is tested against 25 random line sets.

Each entry gets a verdict:

| Verdict | Meaning |
|---|---|
| explains unindexed lines | explains ≥ 2 unindexed lines better than random lines would, with consistent intensities, a plausible cell change and an entry near the convex hull |
| possible, check | as above, but the cell had to change by more than DFT errors allow, or the entry is far above the hull |
| same as … in your list | the compound you already have; its refined cell is a cross-check |
| consistent, adds nothing | fits, but every line it matches is already indexed (often an oriented-film fit whose 00l lines fall on a listed phase's 00l series) |
| not supported | with the line or test that rules it out |

**Add selected phases** puts a candidate into the phase list. It brings the cell refined to your pattern and its reflections of ≥ 2 % calculated intensity. The joint fit then refines it with the other phases. Monoclinic and triclinic candidates are supported.

On the reference pattern:

- With VSe₂ and SnSe₂ listed, corundum V₂O₃ (mp-18937) is the only strong candidate. It explains 6 of the 13 unindexed lines, with a cell 0.4 % from the DFT value. Adding it gives a = 4.9536(12), c = 14.001(2) Å.
- With all three listed, all three come back as "same as" and nothing new is proposed.
- On a synthetic Si pattern, Fd-3m Si is found and the other experimental Si polymorphs are rejected.

The same search runs from the command line, using `MP_API_KEY` or `~/.config/mp/key`:

```bash
python3 cli.py mysample.csv --mp-search "Sn V Se O"     # also writes <out>_mp_candidates.csv
```

Structures and energies come from the Materials Project (CC BY 4.0). Cite it when you use a candidate. A refined cell is a measured value; the DFT cell it started from is not.

## Le Bail tab (optional)

```bash
python3 -m pip install -r requirements-lebail.txt      # rietx 1.10
```

Click **Run Le Bail fit** in the Le Bail tab. The fit uses the last refined phase list and angle correction of the Peak indexing tab:

- **Background:** the step-2 background of the indexing, held as it is.
- **Phases:** every switched-on phase starts from its indexed cell, with cell lengths bounded to ±0.5 % and angles to ±0.5°. A phase uses its space group when rietx knows it, otherwise the highest symmetry of its lattice and centering.
  - A 00l series is fitted as P4/mmm with a = b = 0.8 Å held, so only its 00l reflections fall in the scan; c is refined.
  - An hk0 series is fitted with c = 0.8 Å held and is added last, with a broad-band size.
  - "00l + hk0" becomes one of each.
- **Weak phases:** a phase whose strongest indexed line is below 3 % of the strongest peak keeps its indexed cell. With every reflection intensity free, Le Bail cannot fix the cell of a ~1 % phase; on the example, V₂O₃ drifts to its bound. The threshold can be changed, or switched off with **Refine weak cells too**.
- **Extras:** unindexed sharp lines become free extra peaks, and unassigned broad features become humps.
- **Cu Kβ:** a Kβ line is held at the measured Kβ/Kα ratio.
- **Plans:**
  - Fast (default, about 1–2 min on the example): one Le Bail intensity pass and no closing stage.
  - Thorough (about 5–10 min): two passes and the rietx closing stage. On the example, its cells agree with the fast plan within their esds.
- **esds:** the rietx esds × the Bérar–Lelann factor √((1+ρ)/(1−ρ)), with ρ = 1 − DW/2. Raw esds on film patterns are typically 3–7× too small.

On the reference pattern (fast plan, 80 s) the Le Bail fit gives Rwp 2.80 %, GoF 1.89 and a displacement of 0.124(12) mm:

| | Le Bail | Peak positions |
|---|---|---|
| VSe₂ a | 3.3599(6) Å | 3.3592(3) Å |
| VSe₂ c | 6.0977(13) Å | 6.0980(5) Å |
| SnSe₂ a | 3.8103(13) Å | 3.8103(3) Å |
| SnSe₂ c | 6.1308(17) Å | 6.1321(6) Å |

All four agree within 2σ. V₂O₃ is held as a weak phase.

## Candidate phases from CIF files (optional)

This needs pymatgen, which `requirements-mp.txt` installs. No API key is needed.

In the Peak indexing tab, under **Candidate phases from CIF files**:

1. Upload one or more .cif files. Each data block becomes one structure. It is read with its symmetry and occupancies and reduced to the conventional cell. Files that cannot be read are named with the reason.
2. Choose the intensity models and click **Screen**. Each structure gets the same verdict as a Materials Project entry: explains unindexed lines, possible (check), same as a listed phase, consistent but adds nothing, or not supported. The cell change is given relative to the CIF.
3. Pick structures under **Add to the phase list**. Choose the reflections (as screened, all, 00l, hk0) and the starting cell: refined to the peaks, or as in the CIF. Then click **Add selected**, then **Refine**. A structure that was not screened, or had too few lines in range to test, is added with its CIF cell and its reflections of ≥ 2 % calculated intensity.

From the command line: `python3 cli.py sample.csv --cif V2O3.cif VO2.cif` prints the verdicts and writes `_cif_candidates.csv`.

## Background options

These are set at the top of the Peak indexing tab and applied with **Refine**. The chosen background is used for the peak fit, the indexing, the figures and the Le Bail fit.

- **Refit: Chebyshev + low-angle decays** (default). An N-term Chebyshev polynomial (24 by default) plus, optionally, four low-angle exponential decays. When the file has a prior refinement, the polynomial is fitted to yobs minus the prior peak model; otherwise it is fitted to a rolling 5th percentile of the data.
- **From the file, as it is.** The file's `bkg` column is used unchanged. This option appears only when the file has one.
- **From the file + Chebyshev correction.** The file's `bkg` column plus a low-order correction (8 terms by default) fitted to what it misses.
- **Refine with the peaks** (not for "as it is"). Refines the polynomial and every fitted peak (pseudo-Voigt Kα1/Kα2 doublets) together over the whole pattern by weighted least squares, before indexing. The refined peaks are the ones indexed.

The result line shows the background used and, after a refinement, Rwp before and after, and how far the background moved.

On the example, refining with the peaks leaves the cells unchanged within 0.0001 Å. For a plain .xy file (no prior model), it lowers Rwp from 14.9 % to 2.4 %. For a refinement CSV, the default refit against the prior peak model fits better (Rwp 1.30 %) than pseudo-Voigt peaks can (2.33 %), so refinement is mainly for files without a prior model.

From the command line: `--background refit|file|file+poly`, `--bkg-order N`, `--no-decays`, `--refine-bkg`.

## Run it for your group

- **On one lab computer:** `streamlit run app.py --server.address 0.0.0.0`. Group members then open `http://<that computer's name or IP>:8501` on the same network.
- **With Docker:**
  ```bash
  docker build -t xrd-workup .
  docker run -p 8501:8501 xrd-workup
  ```
  The image includes the Materials Project search and the Le Bail tab. Leave them out with `--build-arg WITH_MP=0` or `--build-arg WITH_LEBAIL=0`.
- **On a hosted service:** push this folder to a GitHub repository and deploy `app.py` on Streamlit Community Cloud (share.streamlit.io), or on any server that runs Docker. Use a private repository and restrict viewers if the data are unpublished.

There are no logins. Uploaded files are held in memory only for the browser session and are not written to disk.

## Command line

The same code without the browser:

```bash
python3 cli.py examples/SnSeVSe22_27Aug26.csv
python3 cli.py mysample.xy --phases examples/phases_SnSeVSe2.json --correction zero --out results/mysample
python3 cli.py mysample.csv --test 30.38 26.52          # line tests at fixed 2θ
python3 cli.py mysample.csv --raw-figure 3.5 2.6 8       # also the raw-pattern figure at 3.5 × 2.6 in, 8 pt
python3 cli.py --write-phases phases.json               # starting phase file to edit
python3 cli.py mysample.csv --lebail fast                # also the Le Bail fit (or: thorough)
```

It writes `<out>_bkg24.csv`, `_peak_indexing.csv`, `_refined_parameters.csv`, `_report.txt` and `_indexed.pdf/.png`. With `--lebail` it also writes `_lebail_fit.csv`, `_lebail_parameters.csv`, `_lebail_vs_peaks.csv` and `_lebail.pdf/.png`.

## Input files

- Refinement CSV exports with `twotheta, yobs, ycal, bkg, diff`. The prior peak model is held while the background is refitted.
- Two- or three-column `.xy`, `.xye`, `.txt`, `.dat` or `.csv` files (2θ, intensity, optional esd), separated by commas, tabs or spaces, with optional `#` header lines.
- An intensity unit given in the header (`#Intensity_unit=cps`) is shown on the axis. A count rate is flagged, because the weights and esds assume counts.

## Phases

Phases are edited in the table in the Peak indexing tab, or loaded and saved as JSON in the same format `cli.py --phases` reads. Each phase needs:

- a name
- a lattice: cubic, tetragonal, hexagonal, orthorhombic, monoclinic (unique axis b) or triclinic
- a centering: P, I, F, C, or R in the hexagonal setting
- a starting cell
- a list of reflections to look for: three digits each (`101 10-2`), or with commas (`1,0,10`)

The library offers VSe₂, SnSe₂, V₂O₃, SnSe, PbSe, Bi₂Se₃, SnO₂, Se and the Si standard. Their hkl lists are the reflections with ≥ 2 % calculated intensity. Their starting cells are room-temperature literature values, so check them against your own references.

Only peak positions are used. Without atomic coordinates the app does not calculate intensities.

## Files

```
app.py                 Streamlit app
cli.py                 command-line version
HELP.md                text of the app's Help tab
xrdcore/io.py          reading patterns
xrdcore/indexing.py    background, peak fits, assignment, cell fit, Kβ check, line tests, tables
xrdcore/phases.py      phase format, example set, hkl parsing
xrdcore/phase_library.json
xrdcore/figures.py     publication figures (matplotlib) and their Layout
xrdcore/bkgrefine.py   joint refinement of the background polynomial and the peaks
install.py             creates .venv and installs the packages (setup.sh / setup.command / setup.bat call it)
run.sh, run.command, run.bat   start the app with .venv (written by install.py)
xrdcore/views.py       interactive plots (Plotly)
xrdcore/mpsearch.py    Materials Project search and candidate screen (optional packages)
examples/              reference pattern and its phase file
tests/test_core.py     python3 -m pytest -q
tests/test_mpsearch.py candidate screen on saved MP structures (tests/data/mp); a live test runs when MP_API_KEY is set
tests/test_series_aidex.py  00l/hk0 series on a synthetic misfit film, hk0-mode screen, AIdex-r2 parsing and cell checks
tests/synth_misfit.py  synthetic misfit-film pattern used by the tests
xrdcore/aidex.py       AIdex-r2 line list, results parser and cell checks
```

## Tests

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
```

The tests cover:

- reading files
- reproducing the reference cells
- recovering a known cell and displacement from a synthetic Si pattern with no prior model
- holding an axis that no line constrains
- figure sizes, and every option of the figure layout
- the command line
- a headless run of the app
- the candidate screen: finding V₂O₃ when it is missing from the list, recognising listed phases, the synthetic Si case, and adding a monoclinic candidate
- 00l/hk0 series on a synthetic misfit film, and the hk0 screen finding a rock-salt layer
- the background options: file background as is, file + correction, Chebyshev order and decays, and the joint background + peak refinement (synthetic known background, and a plain .xy of the example)
- reading CIF files (good, bad and multi-block), screening them against the example pattern (V₂O₃ found when it is left out of the list), and adding them as phases
- AIdex-r2 table parsing, and cell checks with known answers
- the Le Bail phase mapping for 00l/hk0 series, and a Le Bail fit of a synthetic Si pattern with a known cell
- with `MP_API_KEY` set, a live fetch from the Materials Project
- with `XRD_SLOW=1`, the Le Bail fit of the example pattern (about 80 s)

## Not included

These are in the full XRD Workup but not in this app:

- intensities in the indexing itself (the candidate screen uses calculated intensities; the joint fit uses positions only)
- cell searches for unindexed lines
