# CLAUDE.md: project memory for the Lumina 21 cm / 3.46 cm pipeline

Read this first. The full record is **`JALPHA_NOTES.md`**: what was fixed and why, units, how to run each step, approximations, test results and open items. The physics of the He II-band absorption, with references, is in the Claude Doc "How the He II Lyman-band IGM absorption works".

## Working with the user (Changhyun Cho)
- Expert in the field. Wants correct calculations first, checked with numbers, then better ones. Be concise and direct, and say plainly when something is wrong, including your own earlier claims.
- When asked to *check* a calculation, check it. Don't make physics or modelling choices on the user's behalf: present the options with numbers and let them decide.
- They run everything on the MIT ORCD cluster and paste logs back. Give exact commands. Never assume a job ran.
- They edit the manuscript (`21cm_manuscript/oja_template.tex`) themselves. Show locations and suggested text with inline change marks; don't edit it.
- Work directly on `main` (no feature branches; user instruction 2026-10-13). Commit only what is asked; the user pushes. `logs/` is git-ignored.

## What the code does
Lumina is a 500 cMpc (338.3 cMpc/h) radiation-hydrodynamics simulation (Zier et al. 2026, arXiv:2605.15310). Renders are 640³ grids (0.78 cMpc cells), 709 snapshots. The H/He switch is at z = 4.75, snapshot 428: later snapshots come from a second render directory.

| Step | Code | Output |
|---|---|---|
| Effective SFR (stellar age history) | `make_SFR_eff.py` (`run_SFR_eff.sh`) | `SFR_eff/{H,He}/...` |
| J_α, stars, H I Lyman band | `compute_Jalpha.c` | `Jalpha_H_NNN.hdf5` (`J_alpha`) |
| J_α, stars, He II band (f_esc = 0.18) | `compute_Jalpha_He.c` | `Jalpha_He_NNN[_tr].hdf5` (`J_alpha_He`) |
| J_α, AGN (H or He) | `compute_Jalpha_AGN.c` | `Jalpha_AGN_{H,He}_NNN[_tr].hdf5` |
| He-band IGM absorption | `make_opacity_fields.py` → `compute_transmission_He.c` (+ `jalpha_transmission.h`) | `transmission_He_NNN.hdf5`; J He with `TRANS_DIR`, `TR_TAG` |
| He II density in redshift space | `remap_rsd.py` (`run_remap_He.sh`) | `SpectralDensityHeII_NNN.hdf5` |
| δT_b maps + power spectra | `PS_21cm_mpi.c`, `PS_3cm_mpi.c` (+ `functions.c`, `collisions.c`; `run_21cm.sh`, `run_3cm.sh`) | `RIN_NNN.hdf5`, `PS_NNN.out` |
| Ionization history (z, x_HII, x_HeIII of every render) | `check_ionization_history.py` (`run_ion_history.sh`) | `ion_history/ionization_history.csv` |
| Cell-level δT_b test (full snapshots) | `test_cells_Tb.py` (`run_cells_Tb.sh`, `run_cells_merge.sh`), `decompose_cells_He.py` (`run_decompose_cells.sh`) | `cell_test/` |
| VIDs and one-point statistics | `make_vid.py` (`run_vid.sh`) → figures in `VID.ipynb` (`one-stat.ipynb` is the archive) | `VID/<line>/` |
| Global signal | volume mean of the cell-by-cell RIN maps (decided 2026-10-14); `global_snaps_clean.ipynb` (mean-field) is retired. No-heating reference: `TK_REF=1e-3` in `run_21cm.sh`, anchored per pixel to Lumina's own T at render snapshot 0 (`TK_REF_ANCHOR=1`, decided 2026-10-19; the Eq. 27 law was colder than Lumina's unheated gas; JALPHA_NOTES §6.2); `plot_global_Tb.py` | |

Test runs: `run_Jalpha_test.sh` (snapshots 156, 300, 406, 500). Checks: `check_*.py`, `bracket_Tb_He.py`, `check_tb_choices.py`.

## Conventions established (with evidence; don't re-derive)
- **Render fields:**
  - `HII_Fraction` and `Temperature` are mass-weighted; there are also `*_VolumeFraction` and `TemperatureVolumeWeighted`.
  - `HeII_Fraction` and `HeIII_Fraction` are **per H nucleus** (n_HeII/n_H; max = y = 0.0789), so x_HeII = field / y.
  - `SpectralDensityHI` is the H I mass density (X = 0.76 included) remapped along +z, the 3rd axis. On Lumina, Σ SpectralDensityHI/0.76 = Σ Density·(1 − HII_Fraction) to 1e-5.
  - `SpectralDensity` is the gas density remapped along +z.
  - `Velocities` is (640,640,640,3) in AREPO internal units: **v_pec = Velocities × √a km/s**. Validated: the remap reproduces SpectralDensity and SpectralDensityHI.
- **How the renders are made** (AREPO `src/reionization_movie/RT_movie_3D_rendering.cc`, in `/Users/chcho/arepo`; Zier+26 §9.3):
  - each gas cell is a cube of its own volume, deposited by overlap (adaptive CIC); `Spectral*` fields shift each cell by its own v_pec/(aH) along +z;
  - star-forming (Sfr > 0) gas: MassHI = Mass, He II = He III = 0, so it is neutral in both `SpectralDensityHI` and `HII_Fraction`. **Kept as is for the 21 cm maps (decided 2026-10-14)**: below z ≈ 5.5 ~98% of the H I is this gas; Ω_HI ≈ 3.7e-4 at z = 4 (DLAs: ~1e-3). State it in the paper;
  - the renders were made at 1280³ (Density, HII_Fraction and SpectralDensityHI with 2×2×2 sub-cells in the code, but no 2560³ files exist); 1280³ renders are available (Oliver Zier, 2026-10-14; location to be added), `ren_640` is downsampled from them;
  - there are 32 full snapshots (from z = 23), from which an exact SpectralDensityHeII could be rebuilt;
  - below z = 4.75, H and He I come from a uniform UVB (FG09) and only AGN are RT sources.
- **Units:**
  - J_α output: proper photons s⁻¹ cm⁻² Hz⁻¹ sr⁻¹.
  - AGN `LuminosityAGN` already includes f_esc(L), checked R = 0.1949. It is Lumina's obscuration correction L_obs/L_bol = 0.3 (L_bol/10⁴⁶)^0.07 on L_bol, frequency-independent (Zier+26 Eq. 15), so applying it in both bands is consistent. LUM_UNIT = 6.4459e36 erg/s.
  - Cosmology everywhere is Lumina / Planck 2018: h = 0.6766, Ω_m = 0.30964144, Ω_b = 0.048975, with h read from headers where possible.
- **x_α prefactor:** 27 for H, **9 for ³He⁺**. The 9 is correct because the hyperfine levels are inverted (checked against McQuinn & Switzer 2009, as are x_c^He with σ̄ = (14.3 eV/kT) a₀² and S_α^He ≈ 1).
- **δT_b prefactors (2026-10-14):** the T_b codes use the exact optically thin τ T_S with H(z) including Λ (header OmegaLambda, else 0.6902672) and exact constants (T*₂₁ = 0.0681687 K, A₁₀ = 2.86888e-15 s⁻¹, T*_He = 0.4158856 K, A₁₀(³He⁺) = 3(ν_He/ν₂₁)³A₂₁ = 1.95434e-12 s⁻¹; [³He/H] = 1e-5). `check_Tb_maps.py` checks a C map cell by cell against independent code. The textbook 27 mK / 1.67 µK √((1+z)/10) forms (matter-only H) are +1.3% / +2.4% high at z = 3, ≤ 0.5% / ≤ 1.0% at z ≥ 6. (`global_snaps_clean.ipynb` still has the textbook forms and old constants; it is no longer used.) The optical depth stays linear (δT_b ∝ τ, the usual approximation; |δT_b| ~4% high at mean density at z ≈ 20).
- **T_b codes, defaults:** mass-weighted spin-temperature inputs (`-DTS_VOLUME_WEIGHTED=1` for the alternative).
  - **H in real space** (decided 2026-10-14): `run_21cm.sh` sets `-DH_REDSHIFT_SPACE=0`, δT_b from Density (1 − HII_Fraction), maps in `Tb_maps_prod/H_real`. Reason: the code default multiplies the redshift-space SpectralDensityHI by the spin factor of the real-space pixel (the spin factor belongs to other gas), and the He maps are real space. The paper must say real space for both lines. A fully consistent redshift-space H (spin factor remapped with the H I mass) is an optional extra, not built.
  - **He in real space** (decided 2026-10-13): the renders and light-cone have no redshift-space He II, so `run_3cm.sh` sets `-DHE_REDSHIFT_SPACE=0`. The PS, BS and VID comparisons of the H/He fractions are all real space. The code default (shortcut `HeII_Fraction × SpectralDensity`) and `HE_RS_DIR` (remap) remain only for tests.
  - The size of the redshift-space effect on He is measured by `remap_rsd.py --rsd` (`run_remap_qtest.sh`), as a caveat for the paper.
- **Ionized-fraction labels:** every x̄_HII / x̄_HeIII in the figures is the volume-weighted ⟨x⟩_V (checked 2026-10-15, `check_ionization_history.py`; notebook z_match = interpolated crossings). Map files are not zero-padded (`RIN_17.hdf5`, `PS_17.out`); renders are (`All_017.hdf5`).
- **Full snapshots** (`/orcd/data/mvogelsb/005/Lumina/Lumina_{above,below}_z_4p75/snapshots`, 21 + 11, snap_NNN.hdf5 virtual datasets, 2.1×10¹¹ gas cells): IntCoordinates (uint32, × BoxSize/2³²), no ElectronAbundance (Ne = HII + HeII + 2 HeIII, exact for AREPO-RT); not in spatial order; each matches a render snapshot exactly for z ≤ 12.
- **Power-spectrum `.out` files:** Δ² = column 2 × (line 3 / line 4)², as `load_ps` in the notebooks does.
- **Absorption:** star-forming cells are transparent (`SF_MASK=1`, as in Lumina's RT and f_esc). The unresolved-absorber term (`LLS_A`) uses the broken-power-law f(N) conversion (1.59 and 1.82, not 3.5 and 4.8). **Decided 2026-10-14: `LLS_A = 0`** (Lumina's own resolved IGM only, no external parameters). Known biases: in the He band (`mfpE`) the simulation is 3× too transparent at 3.35 < z < 4.75 and 2–2.5× too opaque below 3.35; the calibrated term (LLS_A = 57.6, slope −5.4, z ≥ 3.35) is kept only as a systematic. With Lumina's absorption, the Lyα share of the He coupling is 0.9% (z = 6.35), 7% (z = 5), 18% (z = 4) of the mean δT_b^He. The J codes' shell-average × direction-average product underestimates J^He by ~30% at z = 6.3, ≤ 3% at z ≤ 5 (`check_angular_covariance.py`): ≤ 0.3% on δT_b^He, not corrected (JALPHA_NOTES §4.5, §7).

- **Resolution (2026-10-15, `test_resolution_Tb.py`, 320→2560³):** δT_b from 640³ pixel-averaged fields is reliable for **21 cm at z ≲ 8** (≲ 1–2%). **21 cm at z ≳ 10** (temperature mixing in neutral pixels: coupled T_S needs ⟨1/T⟩ of the H I, the render gives mass-weighted T) and **3.46 cm at all z** (weak collisional coupling, δT_b^He ∝ n_e n_HeII: clumping) do not converge with pixel size (He grows ×1.4–1.6 per doubling up to 2560³). Those need a gas-cell-level calculation (full snapshots). J_α is unaffected.
  - **Cell-level test (2026-10-16, `test_cells_Tb.py`, full snapshot 136, z = 4.0, whole box):** the 640³ production ⟨δT_b^He⟩ is 0.144 of the gas-cell value (0.0113 vs 0.078 µK), P lower by 50–100×, r(k) ≈ 0.2–0.45 at k ≲ 0.15 h/cMpc; the cube ladder has not converged at 10240³ (0.75). So the earlier "≳ 2× / ≳ 3×" understates it at z = 4. Which gas carries it: `decompose_cells_He.py`. Treatment of 3.46 cm in the paper: open (user decides after the decomposition).
  - **Decided 2026-10-15: keep the 640³ maps and state the limitation as bounds** (no cell-level calculation for now). 21 cm at z ≲ 8: quantitative (≲ 2%). 21 cm at z ≳ 10: the 640³ δT_b is an **upper limit** (absorption underestimated; refinement moves it towards absorption), no conclusions from its absolute value. 3.46 cm: the amplitude is a **lower limit** (≳ 2× in the mean, ≳ 3× in P between 640³ and 2560³); at z ≥ 5 the PS shape and morphology are robust (P_640/P_2560 independent of k, r ≈ 1); z < 5 qualitative.

## Cluster
- Renders: `/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640` (snapshots ≤ 428) and `.../Flagship_below_z_4p75_new/postprocessing/ren_640` (> 428). Each has per-field directories and `All/All_NNN.hdf5`.
- User data: `/orcd/data/mvogelsb/004/chcho/`.
- Modules: hdf5/1.14.3 (an MPI build, so compile with mpicc) and fftw/3.3.10.
- Compressed renders need `HDF5_PLUGIN_PATH` (from hdf5plugin).
- Partition: `sched_mit_mvogelsb`.
- `/orcd/data/mvogelsb/004` is a shared 160 TB group filesystem; it filled on 2026-10-16 and writes (maps, logs) failed silently. Check `df -h` before large jobs.

## State (2026-10-17) and next steps
1. **Production done:** J_α H and He; 21 cm maps (305, verified); the no-heating reference with the per-pixel anchor (102 maps, jobs 26334741 + 26374968, verified at 242; Lumina's neutral IGM is heated to ~10²–10³ K by z = 10 at x_HII < 10⁻³, so the reference stays −130 mK below the fiducial at z ≈ 9–10; JALPHA_NOTES §7); 3.46 cm maps (205; `check_Tb_maps.py` 1.4×10⁻⁴ from rounded constants in `x_c_He`, `check_ps.py` OK); H VIDs (305). He VIDs running.
2. **Open decisions (user):**
   - 3.46 cm resolution: cell level at z = 4 is 7× the 640³ mean, carried by dense gas around haloes (decomposition, JALPHA_NOTES §6.2, §7); the IGM test (job 26136927) shows the 640³ maps are neither the IGM signal (even the IGM loses 42% below 1+δ = 10 at 640³; the maps' mean is ×1.25 the IGM, variance ×9, r = 0.43) nor the total; cell-level runs (with whole-box IGM-only maps) at full snapshots 92 and 116 running.
   - ~~Signal normalisation~~ decided 2026-10-17: not corrected (gas-mean normalisation vs Ω_b prefactor: −0.4% at z ≳ 6 to +0.7% at z = 3 in δT_b); fix with the `x_c_He` constants only if the maps are rerun. Ask the Lumina team about the +0.42% gas excess over Ω_b ρ_crit before star formation (JALPHA_NOTES §6.2, §7).
3. **Notebooks:** `VID.ipynb` reads production outputs; PS_redshift, visual_PS, visual_maps, PS2Bubble, bispec_H/He, subvol_VID still read pre-production maps → repoint, then the bispectrum k units, He bispectrum inputs, placeholder instrument curves.
4. **Manuscript edits** (the user does them; list in `JALPHA_NOTES.md` §6.3): real space for both lines (and drop "peculiar velocities down to the particle resolution"); volume-weighted x labels; the resolution statements (3.46 cm wording waits for decision 2); the flowchart's hot ISM and HMXB; the AGN shell count 400 → 200; escape fractions, SFR_eff and absorption.
