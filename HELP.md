### Files

- **Refinement CSV exports** with columns `twotheta, yobs, ycal, bkg, diff` (a leading `#` on the header is fine). The prior refinement's peak model (`ycal − bkg`) is held fixed while the background is refitted, and Rwp is compared with the prior value.
- **Two- or three-column patterns** (`.xy`, `.xye`, `.txt`, `.dat`, `.csv`): 2θ, intensity, optional esd, separated by commas, tabs or spaces. Without a prior model, the background is fitted to a 3° rolling 5th percentile of the data and the "calculated" curve is the sum of the fitted peaks.
- Header lines starting with `#` are read for column names and for an intensity unit (`#Intensity_unit=cps`). A count rate is flagged, because the weights and esds assume counts.
- Radiation is Cu Kα (Kα1 1.540593 Å, Kα2 1.544427 Å).

### What the indexing does

1. **Step-2 background**: 24-term Chebyshev plus four low-angle exponential decays, weighted least squares (w = 1/√y).
2. **Peaks**: seeds from `scipy.signal.find_peaks`, then pseudo-Voigt Kα1 + Kα2 doublets fitted cluster by cluster (peaks closer than 0.8° together). A peak is kept when its height exceeds 3σ of the integrated signal. The fitted position is Kα1. FWHM > 0.25° is "broad"; a width stuck at the fit bound is a hump.
3. **Assignment**: every listed hkl of every switched-on phase goes to the nearest sharp peak, with tolerance 0.15°, then 0.08°, then 0.06° as the cells improve.
4. **Cells**: one joint least-squares fit of all cells and one angle correction, using only lines that belong to one phase. σ = √diag[(JᵀJ)⁻¹ Σr²/(n − p)]. Values "with shared lines" use every assigned line. An axis that no assigned line constrains (for example *a* when only 00l lines matched) is held at its start value and marked "not determined".
5. **Cu Kβ**: every reflection strong enough to leave a visible Kβ copy is fitted directly at its Kβ position.
6. **Line tests**: a line at a fixed position, fitted beside the known peaks of the window, with its height ± esd. Use them to decide whether a suspected phase is present.

### Background

At the top of the Peak indexing tab you can set the background:

- **Refit:** a Chebyshev polynomial plus optional low-angle decays.
- **From the file, as it is:** the file's `bkg` column.
- **From the file + Chebyshev correction:** the file's `bkg` column plus a fitted correction.

Choose the number of polynomial terms. **Refine with the peaks** refines the polynomial and all peaks together before indexing; it is most useful for plain .xy files. Changes take effect when you click **Refine**.

### Refine

Edits to the phase list or the angle correction are not fitted until you click **Refine**. Until then a notice says the results are for the last refined list. Switch on **Refine automatically** to refit after every edit.

### Le Bail tab

**Run Le Bail fit** fits the whole pattern with rietx, starting from the last refined phase list:

- The step-2 background is held.
- Cells are bounded to ±0.5 %.
- Unindexed lines become free extra peaks, and broad features become humps.
- 00l and hk0 series keep their unseen axes held.

A phase whose strongest line is below 3 % of the strongest peak keeps its indexed cell, because Le Bail cannot fix it.

The esds are multiplied by the Bérar–Lelann factor shown. The comparison table marks each axis:

- **agree**: within 2σ of the peak-position value.
- **check**: 2–3σ away.
- **differ**: more than 3σ away.

"At bound" means the data do not determine that value, so don't quote it.

### Reading the results

- **Matched x/y** in the lattice table counts how many of a phase's listed reflections landed on a peak. A phase with only a few matches is probably not there. Switch it off, because the lines it claims are removed from the other phases' fits as shared.
- **Displacement or zero shift**: both often fit equally well. Treat the difference between the two sets of cells as a systematic uncertainty.
- Library cells are room-temperature literature values used as starting points. Check them against your own references before quoting them.

### 00l and hk0 series

A phase's **Reflections** setting can restrict it to its 00l lines (stacking period), its hk0 lines (in-plane cell; broad turbostratic bands are matched within half their width) or both. Use **Add a 00l or hk0 series** to add a superlattice from c alone or an in-plane series from a (and b). The axes a series cannot see are held and shown as not determined.

### Unit cell from AIdex-r2

Switch on **Show the AIdex-r2 unit-cell tool** in the Peak indexing tab and choose the lines.

- **Run AIdex-r2** sends the lines directly. This needs internet access from the computer running the app.
- Without internet access, open AIdex-r2 and paste the one-line script the app shows into the browser console (F12 → Console). It fills every peak box at once. Then click **Predict**, copy the results table back into the app and click **Check the cells**.

Each cell is refined to the lines and judged on misfit, line density, M(n), and how often random lines fit as well. A cell whose largest d is shorter than the lowest-angle line is impossible. AIdex-r2 needs ideally 15–20 lines of one phase.

### Candidate phases from the Materials Project

This needs `requirements-mp.txt` and a free Materials Project API key.

1. Enter the elements likely present, for example `Sn V Se O`.
2. Choose the intensity models (random powder, 00l only, hk0 only, 00l + hk0) and click **Search**. Every experimentally made entry in those chemical systems is tested against your peaks. Its cell is first refined to the peaks, because DFT cells run 1–6 % large.

Verdicts:

- **Explains unindexed lines**: worth adding.
- **Possible, check**: the fit needed an implausible cell change, or the entry is far above the convex hull.
- **Same as … in your list**: the compound you already have.
- **Consistent, adds nothing**: it fits, but only on lines that are already indexed.
- **Not supported**: the reason names the missing line or failed test.

**Add selected phases** puts a candidate in the phase list with its refined cell, ready for the joint fit. Results are tied to the phase list they were screened against, so search again after changing it.

### Export figures

1. Choose the figure: raw pattern, indexed pattern or Le Bail fit (after a fit has been run).
2. Set what it shows: intensity scale, 2θ range, tick rows, peak labels and curves.
3. Set the layout:
   - Size preset, or your own width and height.
   - Font and font size. The legend, peak labels and phase names scale with the font size.
   - Line thickness: the calculated (or raw) curve. The background, Δ and tick marks scale with it.
   - Frame thickness: the axes.
   - Marker size: the observed-data circles. 0 draws a line instead.
   - Δ panel height and tick-row panel height, each relative to the main panel. The tick-row height is automatic until you switch that off.
   - Legend, Rwp and phase names at the right. Switch this off for a narrow figure.
4. Download PDF, SVG or PNG. **Layout settings · JSON** records the layout.

The preview is the exported figure at screen resolution.

### Candidate phases from CIF files

1. Upload .cif files of phases you suspect.
2. Click **Screen**. Each structure is tested like a Materials Project entry, with the same verdicts. Cell changes are given relative to the CIF.
3. Pick structures under **Add to the phase list** and choose the starting cell: refined to the peaks, or as in the CIF.
4. Click **Add selected**, then **Refine**.

This needs pymatgen.

### Command line

`python3 cli.py pattern.csv --phases phases.json --correction displacement --test 30.38` gives the same numbers as the app and writes the same files. **Phases · JSON** in the Export section saves the phase list in the format `--phases` reads.
