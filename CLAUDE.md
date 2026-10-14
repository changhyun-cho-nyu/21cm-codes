# CLAUDE.md: project memory for the Lumina 21 cm / 3.46 cm pipeline

Read this first. The full record is **`JALPHA_NOTES.md`**: what was fixed and why, units, how to run each step, approximations, test results and open items. The physics of the He II-band absorption, with references, is in the Claude Doc "How the He II Lyman-band IGM absorption works".

## Working with the user (Changhyun Cho)
- Expert in the field. Wants correct calculations first, checked with numbers, then better ones. Be concise and direct, and say plainly when something is wrong, including your own earlier claims.
- When asked to *check* a calculation, check it. Don't make physics or modelling choices on the user's behalf: present the options with numbers and let them decide.
- They run everything on the MIT ORCD cluster and paste logs back. Give exact commands. Never assume a job ran.
- They edit the manuscript (`21cm_manuscript/oja_template.tex`) themselves. Show locations and suggested text with inline change marks; don't edit it.
- Work on a feature branch, push, and the user merges the PR into `main`. Commit only what is asked. `logs/` is git-ignored.

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
| Global (mean-field) signal | `global_snaps_clean.ipynb` | |

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
  - star-forming (Sfr > 0) gas: MassHI = Mass, He II = He III = 0, so it is neutral in both `SpectralDensityHI` and `HII_Fraction`;
  - native grids are 1280³ (2560³ for Density, HII_Fraction and SpectralDensityHI as 8 sub-cells); `ren_640` is downsampled;
  - there are 32 full snapshots (from z = 23), from which an exact SpectralDensityHeII could be rebuilt;
  - below z = 4.75, H and He I come from a uniform UVB (FG09) and only AGN are RT sources.
- **Units:**
  - J_α output: proper photons s⁻¹ cm⁻² Hz⁻¹ sr⁻¹.
  - AGN `LuminosityAGN` already includes f_esc(L), checked R = 0.1949. LUM_UNIT = 6.4459e36 erg/s.
  - Cosmology everywhere is Lumina / Planck 2018: h = 0.6766, Ω_m = 0.30964144, Ω_b = 0.048975, with h read from headers where possible.
- **x_α prefactor:** 27 for H, **9 for ³He⁺**. The 9 is correct because the hyperfine levels are inverted. δT_b^He = 1.67 µK √((1+z)/10)(…), equal to the paper's 1.3 µK ((1+z)/6)^½ to rounding.
- **T_b codes, defaults:** mass-weighted spin-temperature inputs (`-DTS_VOLUME_WEIGHTED=1` for the alternative).
  - **H in redshift space** (exact, SpectralDensityHI); `-DH_REDSHIFT_SPACE=0` gives real space, 0.76 Density (1 − HII_Fraction).
  - **He in real space** (decided 2026-10-13): the renders and light-cone have no redshift-space He II, so `run_3cm.sh` sets `-DHE_REDSHIFT_SPACE=0`. The PS, BS and VID comparisons of the H/He fractions are all real space. The code default (shortcut `HeII_Fraction × SpectralDensity`) and `HE_RS_DIR` (remap) remain only for tests.
  - The size of the redshift-space effect on He is measured by `remap_rsd.py --rsd` (`run_remap_qtest.sh`), as a caveat for the paper.
- **Power-spectrum `.out` files:** Δ² = column 2 × (line 3 / line 4)², as `load_ps` in the notebooks does.
- **Absorption:** star-forming cells are transparent (`SF_MASK=1`, as in Lumina's RT and f_esc). The unresolved-absorber term (`LLS_A`) is off unless λ₉₁₂,sim > λ₉₁₂,obs. It uses the broken-power-law f(N) conversion (1.59 and 1.82, not 3.5 and 4.8).

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
2. **Production J_α:** opacity → transmission (`run_opacity.sh`, `run_transmission.sh`, with SF_MASK=1). Use the mfp912 lines to decide on `LLS_A`. Then the He J runs with `TRANS_DIR`.
3. **T_b maps** with the new J files, then regenerate the PS, bispectra, VIDs and bubble statistics.
4. **Manuscript edits** (the user does them; list in `JALPHA_NOTES.md` §6.3):
   - the He "real space" sentence at line 236 stays (He is real space); the LIM paper's "peculiar velocities down to the particle resolution" (§2.4) must say H only, and add the measured He redshift-space size;
   - the flowchart's hot ISM and HMXB;
   - the AGN shell count of 400 → 200;
   - escape fractions, SFR_eff and absorption.
