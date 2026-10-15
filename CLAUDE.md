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
| Global signal | volume mean of the cell-by-cell RIN maps (decided 2026-10-14); `global_snaps_clean.ipynb` (mean-field) is retired. No-heating reference: `TK_REF=1e-3` in `run_21cm.sh` (Eq. 27 anchor, JALPHA_NOTES §4.6) | |

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
- **Power-spectrum `.out` files:** Δ² = column 2 × (line 3 / line 4)², as `load_ps` in the notebooks does.
- **Absorption:** star-forming cells are transparent (`SF_MASK=1`, as in Lumina's RT and f_esc). The unresolved-absorber term (`LLS_A`) uses the broken-power-law f(N) conversion (1.59 and 1.82, not 3.5 and 4.8). **Decided 2026-10-14: `LLS_A = 0`** (Lumina's own resolved IGM only, no external parameters). Known biases: in the He band (`mfpE`) the simulation is 3× too transparent at 3.35 < z < 4.75 and 2–2.5× too opaque below 3.35; the calibrated term (LLS_A = 57.6, slope −5.4, z ≥ 3.35) is kept only as a systematic. With Lumina's absorption, the Lyα share of the He coupling is 0.9% (z = 6.35), 7% (z = 5), 18% (z = 4) of the mean δT_b^He. The J codes' shell-average × direction-average product underestimates J^He by ~30% at z = 6.3, ≤ 3% at z ≤ 5 (`check_angular_covariance.py`): ≤ 0.3% on δT_b^He, not corrected (JALPHA_NOTES §4.5, §7).

- **Resolution (2026-10-15, `test_resolution_Tb.py`, 320→2560³):** δT_b from 640³ pixel-averaged fields is reliable for **21 cm at z ≲ 8** (≲ 1–2%). **21 cm at z ≳ 10** (temperature mixing in neutral pixels: coupled T_S needs ⟨1/T⟩ of the H I, the render gives mass-weighted T) and **3.46 cm at all z** (weak collisional coupling, δT_b^He ∝ n_e n_HeII: clumping) do not converge with pixel size (He grows ×1.4–1.6 per doubling up to 2560³). Those need a gas-cell-level calculation (full snapshots). J_α is unaffected.

## Cluster
- Renders: `/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640` (snapshots ≤ 428) and `.../Flagship_below_z_4p75_new/postprocessing/ren_640` (> 428). Each has per-field directories and `All/All_NNN.hdf5`.
- User data: `/orcd/data/mvogelsb/004/chcho/`.
- Modules: hdf5/1.14.3 (an MPI build, so compile with mpicc) and fftw/3.3.10.
- Compressed renders need `HDF5_PLUGIN_PATH` (from hdf5plugin).
- Partition: `sched_mit_mvogelsb`.

## State (2026-10-14) and next steps
1. **He redshift space, for the caveat only** (the He maps are real space): `sbatch run_remap_qtest.sh` gives P_rs/P_real for gas, H I and He II at snapshots 220, 300, 406, 500. Background, the remap validated at those snapshots:
   - √a is best everywhere;
   - gas power is within about 1% for k ≲ 0.3 h/cMpc, a few % at k ≈ 1 (9% at z = 4), and poor at k ≳ 2.
   - It is an approximation: grid-cell mean velocities, not the render's own gas cells.
   - The exact route is to ask the render team for `SpectralDensityHeII`, the same as `SpectralDensityHI` with He II mass.
   - Better approximation, tested in the same job: the q method, exact redshift-space H II (0.76 SpectralDensity − SpectralDensityHI) × remapped He II/H II.
2. **Production J_α:** SFR_eff is complete (`/orcd/data/mvogelsb/004/chcho/SFR_eff`, 2026-10-12). H band running with `run_Jalpha_prod.sh` (BAND=H, outputs in `.../chcho/Jalpha`). He band waits for the absorption: opacity and transmission done for 156/300/406/500 (SF_MASK=1); LLS_A = 0 chosen; testing its effect on δT_b^He (J He test with TR_TAG=tr_sfmask at 300/406/500, then `bracket_Tb_He.py`) → transmission for the He snapshots → BAND=He with TR_TAG=tr_sfmask.
3. **T_b maps** with the new J files, then regenerate the PS, bispectra, VIDs and bubble statistics.
4. **Manuscript edits** (the user does them; list in `JALPHA_NOTES.md` §6.3):
   - both lines are now real space: the He "real space" sentence at line 236 stays and should cover H too; the LIM paper's "peculiar velocities explicitly accounted for down to the particle resolution" (§2.4, after Eq. 1) must go, and the velocity-gradient term of Eqs. 1–2 is not used (state the size of the redshift-space effect from `remap_rsd.py --rsd` as a caveat);
   - the flowchart's hot ISM and HMXB;
   - the AGN shell count of 400 → 200;
   - escape fractions, SFR_eff and absorption.
