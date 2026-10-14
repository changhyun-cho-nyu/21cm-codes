# J_α pipeline — what was wrong, what changed, how to run it, what is left

Covers the review of the Lyα-background codes (`compute_Jalpha.c`, `compute_Jalpha_He.c`,
`compute_Jalpha_AGN.c`), their inputs, and the related manuscript text. All changes are on
branch `claude/zen-franklin-pe0ufw`. Commit hashes are given in brackets.

---

## 1. The pipeline at a glance

```
LUMINA snapshots ──► make_Z_snapshots.py ──► Z_weights.txt  (young-star metallicity mix per z)
                                                  │
ren_640 SFR renders ──► make_SFR_eff.py ──► SFR_eff boxes (H: 300 Myr, He: 100 Myr window)
                                                  │
              ┌───────────────────────────────────┼──────────────────────────┐
              ▼                                   ▼                          ▼
     compute_Jalpha.c (H, stars)       compute_Jalpha_He.c (He II, stars)   compute_Jalpha_AGN.c (H or He II)
              │                                   │                          │   reads LuminosityAGN renders
              ▼                                   ▼                          ▼
        J_alpha (stars)                    J_alpha_He (stars)          Jalpha_AGN_H / Jalpha_AGN_He
              └──────────── J_α^H = stars + AGN(H) ───┘   J_α^He = stars + AGN(He) ──► x_α, T_S, δT_b
```

All three codes evaluate

J_α(x, z) = (1+z)²/(4π) Σ_{n=2}^{30} f_rec(n) ∫_z^{z_max(n)} (c dz′/H(z′)) ε(x, ν′_n, z′)

with ε the comoving photon emissivity. The z′ integral is done on 200 log-spaced shells in
comoving radius (plus an inner shell) by FFT convolution. Output is proper
photons s⁻¹ cm⁻² Hz⁻¹ sr⁻¹ on the 640³ grid.

---

## 2. Problems found and how they were fixed

### 2.1 Geometry and numerics (all three codes)

| # | Problem | Fix | Effect / test |
|---|---|---|---|
| 1 | **Spheres instead of shells.** Each z′ slice was averaged over the full sphere of radius R, not the shell R_lo < r < R_hi. Nearby sources were weighted up to 1.5× too much. | Shell window [R_hi³W(kR_hi) − R_lo³W(kR_lo)]/(R_hi³ − R_lo³) [f7f1343, 00e7876] | Mean unchanged (exact). Near-source J was too high. The multi-line effect on the power is about 1.3× at k ≈ 0.1 h/cMpc, with suppression at k ≳ 0.5. Shell result vs the exact 1/(4πr²) kernel: mean 1.00002, voxel rms error 2.5×10⁻⁵, P(k)/P_true = 0.9999–1.0000. |
| 2 | **Inner region dropped.** Emission at r < 1 cell (the source's own cell) was never counted. | Inner shell [0, R_min] with the exact window Si(kR)/(kR) of a constant-per-unit-r kernel | Included in the tests above. |
| 3 | **Per-shell clipping.** Each shell was clipped at zero. Shell windows ring, so clipping every shell biased the mean upward. | Shells are not clipped; only the final J is clipped once, and the clipped fraction is logged. | Mean preserved; clipping is reported as "clipped N negative voxels". |
| 4 | **AGN used 400 shells**, the stellar codes 200. | 200 everywhere [2242d3a]. | Converged; consistent between codes. |
| 5 | HDF5 output with 64³ chunks failed silently for N < 64 (test grids). | Chunk capped at N. | Test-only issue. |
| — | Shells larger than the box (R_max > L). | No change. The FFT window tends to the box mean, so the mean is preserved and large-scale structure is lost only beyond the box scale. | Withdrawn as an issue. |

### 2.2 Atomic physics

| # | Problem | Fix |
|---|---|---|
| 6 | **f_rec table.** Two tables were in use, and the paper cites Pritchard 2007 (the X-ray paper). | You put the Pritchard & Furlanetto 2006 table in all codes [d7917fb]. The same table applies to He II because hydrogenic branching ratios are Z-independent. **Manuscript citation still to fix.** |

### 2.3 AGN spectrum (`compute_Jalpha_AGN.c`)

| # | Problem | Fix | Effect |
|---|---|---|---|
| 7 | **SED gap.** The Shen+2020 table has no nodes between 20.66 and 248 eV, which is where the He II Lyman band (40.8–54.4 eV) sits. The code interpolated *linearly* across the gap. | Power-law (log-log) interpolation between nodes [88361ea] | The He-band ε_AGN was 2.4–4.5× too high. The earlier audit found the AGN J_α^He maps about 11× above the global pipeline. |
| 8 | **L_bol by trapezoid across the same gap.** | Exact power-law integral per segment, up to 500 keV [88361ea, ffe56d4 for the notebook] | L_bol was 1.85× too large, so the H-band ε_AGN was 1.85× too low. Test with a synthetic SED: new code L_bol/exact = 1.0000 and J/exact = 1.0000 for both species; old code 1.906 and 0.52 (H) / 6.1 (He). |
| 9 | **Is the escape fraction in the AGN render?** | Checked with `check_agn_luminosity.py` [9bc0c2d, eeb7ef5]. Σ render / Σ ε_r(1−ε_f)Ṁc² = **0.1949**, equal to the luminosity-weighted f_esc = 0.3(L/10⁴⁶)^0.07 = 0.1949 at z = 8.90. | The render already holds f_esc(L) × L_bol, and no other obscuration factor is applied. Nothing is added in the code, in either band [6f96cc5]. |
| 10 | The code luminosity unit was 6.4478×10³⁶ erg/s. | 6.4459×10³⁶ erg/s, from the snapshot's own unit attributes [6f96cc5]. | 0.03%. |
| 11 | AGN emission at z′ above the first render (z = 29.43) held the first box. The stellar codes set it to zero. | The z′ integral now ends at the first render, as in the stellar codes [b3bbaf0]. | Only matters for z_obs ≳ 24.7. |

### 2.4 Stellar emissivity (`compute_Jalpha.c`, `compute_Jalpha_He.c`)

| # | Problem | Fix | Effect / test |
|---|---|---|---|
| 12 | **Metallicity.** The BPASS yield was evaluated at the mass-weighted mean Z of *all* stars, which is dominated by old stars and is not the Z of the stars emitting now. A single mean Z also misrepresents a broad Z distribution, because the yield is non-linear in Z. | **Level 2:** ε(ν, z) = Σ_j w_j(z) ε_b(ν, Z_j), where w_j is the fraction of the initial mass of young stars (< 20 Myr) at each BPASS metallicity, from `make_Z_snapshots.py` [77df8b6, eb1eea2, 3ea4f40]. The weights are evaluated at the **emission** redshift z′. | With single-Z weights the new code equals the old one to 10⁻¹³ (J_scalar). A z-dependent mixture vs independent quadrature agrees to 10⁻³. |
| 13 | **Instantaneous emission.** J used SFR(now) × lifetime yield, i.e. all photons a star will ever emit were counted at its birth. B stars dominate the H band and live ~100 Myr, so for a rising SFR this overestimates. The global estimate of true/instantaneous was 0.38 at z = 20 and 0.94 at z = 6 (H band); 0.99 for He at z ≲ 6. | **Cell-by-cell effective SFR**, `make_SFR_eff.py` [55f131d]: SFR_eff(x, t_k) = Σ_m W_km SFR(x, t_m). W comes from the BPASS band age profile and the metallicity mix of each cohort at its formation time. SFR is linear in time between snapshots and zero before the first. Snapshot times come from each file's header (2.00–5.40 Myr spacing, median 2.74, 709 snapshots), not assumed. Window: 300 Myr (H), 100 Myr (He). | W vs independent integration ≤ 10⁻³ (irregular spacing, constant/burst/random histories). The slab/sparse pipeline equals dense W·SFR to 1.1×10⁻⁷. The C codes read the output unchanged. For a constant SFR, SFR_eff/SFR = 1.017 (H; older cohorts are more metal-poor and brighter) and 0.976 (He; 100 Myr truncation). |
| 14 | **Stellar He-band escape fraction.** The He band is ionizing for H I and He I, so host absorption applies, but the code used f_esc = 1. AGN photons in the same band did get Lumina's escape factor. | `F_ESC_STAR_HE = 0.18` (Lumina's value at all z) multiplies the stellar He-band emissivity. `-DF_ESC_STAR_HE=1.0` gives the no-host-absorption upper limit [08d8b74]. | Ratio default/1.0: J_scalar 0.180000000000000, every voxel 0.179995–0.180005. The H band has no escape fraction (photons are below 13.6 eV). |
| 15 | BPASS age range. | No change. The table runs 1 Myr–2 Gyr with the first bin from t = 0, in the C codes and in make_SFR_eff.py alike. | — |

### 2.5 Sources considered and deliberately left out

| Source | Result | Decision |
|---|---|---|
| Hot ISM (bremsstrahlung; your shape: flat to 240 eV, E⁻³ above; 3.3×10⁴⁰ erg/s per M☉/yr) | ≈ 0.45 photons per baryon in each band (≤ 0.78 for any kT = 0.2–1 keV). | Negligible: ≤ 8×10⁻⁵ of stars in the H band (stars 1.0–2.2×10⁴), ≤ 5×10⁻³ in the He band for Z ≤ 0.004 (stars 15–920), ≈ 5% only at Z = 0.04. **Text only.** |
| HMXBs (Fragos 2013b intrinsic SED at z ≈ 10, `HMXB.dat`; Madau & Fragos polynomial) | ≤ 0.02 photons per baryon. The table starts at 14.2 eV, so the H band is extrapolated (L_E held constant). | Negligible; text only. Cite Fragos 2013b with the 2016 erratum (normalization ÷2.6, shape unchanged). |
| X-ray secondary-electron Lyα (notebook `J_alpha_X`) | The channel the paper excludes for f_heat = 1. | Not included, consistent with f_heat = 1. |

### 2.6 Robustness and provenance added [b3bbaf0]

- **Input check.** The stellar codes check *every* input file for the make_SFR_eff.py marker (`band`, `tau_max_Myr`):
  - a mix of effective and raw SFR boxes aborts (for example, when one SFR_eff file is missing and the raw render from the fallback directory is read instead);
  - an effective box made for the other band also aborts.
- **Log.** The log states whether the input is the effective SFR or the raw renders.
- **New output attributes:** `SFR_input`, `tau_max_Myr`, `N_shells` (stellar codes), plus `f_esc_star` (He). The AGN output now also carries `Redshift`, `J_mean`, `J_scalar`, `NU_LL_species` and `Units` on the dataset, not only in a `Header` group.
- **Units attribute.** It is now written from a 64-byte buffer; before, it read past a shorter string literal.

### 2.7 Units — checked end to end

| Quantity | Units | Where |
|---|---|---|
| ε_b (BPASS lifetime yield) | photons Hz⁻¹ baryon⁻¹ = Σ_age L_ν/(hν) · (m_p/M☉) · Δt | `load_spectrum` |
| SFR_UNIT | (M☉/yr → g/s) / V_cell, comoving cm³ | `load_snap_metadata` |
| ε (stars) | SFR·SFR_UNIT/m_p · ε_b = photons s⁻¹ Hz⁻¹ (comoving cm)⁻³. m_p cancels exactly between the two places it appears. | prefactor |
| ε_AGN | (L_ν/L_bol)/(hν) [photons Hz⁻¹ erg⁻¹] × L·LUM_UNIT [erg s⁻¹ (comoving cm)⁻³] | AGN prefactor |
| J | (1+z)²/(4π) · ε · c dz′/H → photons s⁻¹ (proper cm)⁻² Hz⁻¹ sr⁻¹ | all codes |
| Render ↔ snapshot | LuminosityAGN in code units = M V³/L (h cancels); verified with BH accretion rates | `check_agn_luminosity.py` |

Cosmology (Ωm = 0.30964, ΩΛ = 0.69027, h = 0.6766) and band edges (0.75–1 × ν_LL) are identical in the C codes, `make_Z_snapshots.py` and `make_SFR_eff.py`.

---

## 3. What is better now, in one paragraph

J_α is now computed with the correct radial weighting (shells plus the source's own cell) and an unbiased mean. The AGN spectrum is interpolated the way it is defined, which removes a ~2–11× error in the He band and a 1.85× error in the H band. Each voxel's stars emit according to that voxel's own recent star-formation history and the metallicity mix of the stars actually forming, instead of SFR(now) × lifetime yield at one mean Z. Escape fractions are treated consistently: none below 13.6 eV, and Lumina's values for stars (0.18) and AGN (f_esc(L), verified in the render) in the He band. The neglected sources are bounded with numbers. Every run states and records which input it used, and refuses inconsistent inputs.

---

## 4. How to run it

### 4.1 Metallicity weights (once)

```bash
python make_Z_snapshots.py            # -> Z_snapshots.csv, Z_history.txt, Z_weights.txt
```

Copy `Z_weights.txt` to the path in `Z_WEIGHTS_FILE`, or override it at compile time. Both stellar codes must use the same file:
- `cell_by_cell/hydrogen/Z_weights.txt` (H);
- `cell_by_cell/helium/Z_weights.txt` (He).

### 4.2 Effective SFR (once, ~0.2–0.4 TB)

```bash
# edit OUT= (needs >= 600 GB free; the job refuses otherwise), WEIGHTS=, PLANES=
mkdir -p logs && sbatch run_SFR_eff.sh
```

- **Start of the log:** the number of snapshots (709), the spacing, the windows, the memory estimate and the time estimate.
- **End of the log:** box-mean SFR_eff/SFR at z = 20 … 3, to compare with the global R(z) above.
- **Output:**
  - `<OUT>/H/All/All_NNN.hdf5`
  - `<OUT>/He/StarFormationRate/StarFormationRate_NNN.hdf5`
  - `<OUT>/SFR_eff_summary.csv`

Cadence convergence test, after the main run:

```bash
python make_SFR_eff.py --out $OUT_s2 --weights $W --plane-range 0 64 --stride 2
python make_SFR_eff.py --out $OUT_s4 --weights $W --plane-range 0 64 --stride 4
python compare_SFR_eff.py $OUT $OUT_s2      # full-cadence error ~ L1/3 if it scales as dt^2
python compare_SFR_eff.py $OUT $OUT_s4      # L1(4)/L1(2) ~ 5 confirms dt^2
```

### 4.3 J_α

```bash
# H, stars (pass the SFR_eff directory twice: a missing file then fails instead of falling back)
mpicc -O3 -march=native -fopenmp compute_Jalpha.c -o compute_Jalpha_H -lfftw3f_omp -lfftw3f -lhdf5 -lm
mpirun -np 16 ./compute_Jalpha_H <snap> $OUT/H $OUT/H Jalpha_H_<snap>.hdf5

# He II, stars (directories are compile-time)
mpicc -O3 -march=native -fopenmp compute_Jalpha_He.c -o compute_Jalpha_He \
  -DSFR_DIR_OLD='"'$OUT'/He"' -DSFR_DIR_NEW='"'$OUT'/He"' -lfftw3f_omp -lfftw3f -lhdf5 -lm
mpirun -np 16 ./compute_Jalpha_He <snap> Jalpha_He_<snap>.hdf5
#   upper limit without host absorption: add -DF_ESC_STAR_HE=1.0

# AGN (species H or He; same LuminosityAGN renders for both)
mpirun -np 16 ./compute_Jalpha_AGN <snap> $R/LuminosityAGN Jalpha_AGN_H_<snap>.hdf5 agn_sed_full.hdf5 \
       $R_below/LuminosityAGN LuminosityAGN H
```

Optional `-D` overrides: `SPECTRUM_FILE`, `Z_WEIGHTS_FILE`, `N_SNAPS`, and `F_ESC_STAR_HE` (He only). Resources: 16 ranks × 4 threads, about 7.4 GB per rank, so 128 GB per job.

**What to check in each log:**
1. `Input: effective SFR (make_SFR_eff.py), band H, tau_max = 300 Myr, 709 snapshots` (or band He, 100 Myr).
2. `Snapshot sources:` shows every range coming from the intended directory.
3. `ratio = J_mean/J_scalar` ≈ 1 (to ~10⁻³).
4. The clipped-voxel fraction is tiny.

**Output datasets:**

| Code | Dataset | Attributes |
|---|---|---|
| H | `J_alpha` | Redshift, BoxSize_cMpc, J_mean, J_scalar, Units, Z_weights_obs, Z_grid, SFR_input, tau_max_Myr, N_shells |
| He | `J_alpha_He` | as above, plus NU_LL_HeII and f_esc_star |
| AGN | `Jalpha_AGN_H` / `Jalpha_AGN_He` | Redshift, BoxSize_cMpc, J_mean, J_scalar, NU_LL_species, Units (also in `Header`) |

The totals are J_α^H = J_alpha + Jalpha_AGN_H and J_α^He = J_alpha_He + Jalpha_AGN_He, on the same snapshot.

### 4.4 Check scripts

| Script | Question it answers |
|---|---|
| `check_agn_luminosity.py --render .../LuminosityAGN_NNN.hdf5 --snap .../snap_MMM.hdf5` | What is in the AGN render (f_esc, obscuration, units)? Result: R = 0.1949 = ⟨f_esc⟩_L. |
| `check_sfr_eff_spectrum.py [--Z ...]` | How good is SFR_eff × lifetime spectral shape inside the band? (Section 5) |
| `bracket_Tb_He.py --render All_NNN.hdf5 --J Jalpha_He_NNN.hdf5 --J Jalpha_AGN_He_NNN.hdf5 [--write]` | Does δT_b^He depend on J_α^He at all? It compares J as computed (no IGM absorption) with J = 0. |
| `compare_SFR_eff.py REF TEST` | Is the SFR interpolation between snapshots converged? |
| `remap_rsd.py --dir <render_dir> --snap NNN --validate [--out DIR]` | True redshift-space He II density: remaps Density × HeII_Fraction along +z with the `Velocities` render (CIC). First validates the remap against the render's own SpectralDensity and SpectralDensityHI, and finds the velocity unit convention. `--out` writes SpectralDensityHeII, which `PS_3cm_mpi.c` reads when built with `HE_RS_DIR`.
Lumina, z = 7.6: `Velocities` is v_pec/√a, the AREPO internal convention. With it, the remap reproduces SpectralDensity and SpectralDensityHI with power ratio ≥ 0.986 and r = 1.000 for k ≤ 1 h/cMpc. At k ≈ 3, H I power is 0.857 and r = 0.988. The shortcut x_HI(real)·SpectralDensity gives power 1.2–1.3× and r = 0.885 at k ≈ 1. Batch job: `run_remap_He.sh`.<br>`--qtest` (job `run_remap_qtest.sh`, snapshots 220/300/406/500) tests the q method: He II_s = [remap(He II)/remap(H II)] × (0.76 SpectralDensity − SpectralDensityHI), exact redshift-space H II times the remapped He II/H II ratio. Check 1 does the same construction on H I against SpectralDensityHI (an upper bound on the error). Check 2 reports how smooth q is, the He II mass kept, and the He power with q remapped vs not. |
| `check_tb_choices.py --dir <render_dir> --snap NNN [--factor 2]` | How accurate is the He redshift-space approximation (tested on H I, where the exact map exists), and how large is the RSD effect on He? Does mass vs volume weighting matter (signal share in partly ionized cells, T_mass/T_vol)? |

### 4.5 IGM absorption of the He II band

The physics, the derivations and the references are in the doc "How the He II Lyman-band IGM absorption works". Pipeline:

| Step | Script | Output |
|---|---|---|
| 1 | `run_opacity.sh` → `make_opacity_fields.py` | `opacity_NNN.hdf5`: n_HI and n_HeI (comoving cm⁻³) at 320³, every second snapshot. Logs, for star-forming cells, the H I / He I share, `self_T` (fraction of a source cell's 40.8 eV photons that survive their own cell) and the share of ionized-cell H I. |
| 2 | `bash run_transmission.sh compile`, then `sbatch run_transmission.sh` → `compute_transmission_He` | the `lambda_912 (simulation …)` line (mfp912 mode), then `transmission_He_NNN.hdf5` holding T_n2 and T_n4 on 48 radii × 160³ origins |
| 3 | `run_Jalpha_test.sh` with `TRANS_DIR` and `TR_TAG` set | `Jalpha_He_NNN_<TR_TAG>.hdf5` and `Jalpha_AGN_He_NNN_<TR_TAG>.hdf5` |

Variants are kept in separate directories:
- **`SF_MASK=1`** (the default since the measurements below) in `run_opacity.sh` and `run_transmission.sh` makes star-forming cells transparent, as in Lumina's own RT. Otherwise their absorption would be counted on top of f_esc. Outputs go to `*_sfmask`; set `TR_TAG=tr_sfmask` for the J runs. The ray tracer refuses a directory that mixes masked and unmasked opacity files.
- **`LLS_A > 0`** adds unresolved absorbers, with outputs in `*_lls`. Use it only if λ₉₁₂,sim > λ₉₁₂,obs, and then calibrate it with 1/λ_extra = 1/λ_obs − 1/λ_sim.
  - The conversion from 912 Å to the band is the effective opacity of a broken power-law f(N): slope 1.5 below 10¹⁷·⁵, 0.8 up to 10¹⁹ (Prochaska+10) and 1.5 above. This gives λ_E/λ₉₁₂ = 1.59 (n = 2) and 1.82 (n ≥ 4).
  - The earlier single power law with β = 1.5 gave 3.5 and 4.8, about 2× too transparent.
  - `-DLLS_FN_B1/B2/B3/LOGN1/LOGN2` change the f(N) slopes and break columns; `-DLLS_BETA=b` restores a single power law.

### 4.6 Brightness-temperature maps (`PS_21cm_mpi.c`, `PS_3cm_mpi.c`)

Run with `run_21cm.sh` and `run_3cm.sh`. Each has a `compile` mode; set `J_DIR`, `TR_TAG` (He) and `OUT_DIR` at the top. Inputs are J^H = `J_alpha` + `Jalpha_AGN_H`, and J^He = `J_alpha_He` + `Jalpha_AGN_He` (the `_<TR_TAG>` files).

Changes from the reviewed versions:
- **He signal is in redshift space** along the 3rd axis, like H: (`HeII_Fraction`/y) · `SpectralDensity`/⟨Density⟩. The fraction is mass-weighted (it was the volume-weighted one).
  - There is no He II density in redshift space, so this multiplies the real-space fraction by the redshift-space gas density. That is exact where x_HeII is uniform over the velocity displacement (~1–3 cells).
  - `check_tb_choices.py` measures the error using the H I analogue, for which the exact map exists.
  - `-DHE_REDSHIFT_SPACE=0` gives real space. The paper's line on the He field being in real space must change.
- **He S_α** uses the He II Gunn–Peterson depth 6×10³ x_HeII((1+z)/7)^1.5, as in the paper. It was the H I depth.
- **Spin-temperature inputs** are the mass-weighted `HII_Fraction`, `Temperature`, `HeII_Fraction` and `HeIII_Fraction`; `-DTS_VOLUME_WEIGHTED=1` restores the volume-weighted fields. A field missing as `<field>/<field>_NNN.hdf5` is read from `All/All_NNN.hdf5`.
- **Cosmology:** h comes from the header (it was 0.6774 hard-coded, which made n_H 0.24% high). Ω_m and Ω_b come from the header, or else Lumina's 0.30964 / 0.048975; the old Planck 2015 values put the prefactor +0.53% off.
- **He prefactor:** 1.67/√10 µK, the same as `global_snaps_clean.ipynb`. It was 0.53, labelled mK.
- **Checks** that abort the run:
  - every read is checked, and dataset shapes must be N³;
  - the J maps must be at the render's redshift;
  - He fractions must be per H nucleus (≤ 1.05 y);
  - Σ SpectralDensityHI / 0.76 must equal Σ Density·x_HI to 5%, which confirms the 0.76 division;
  - an unwritable output aborts.
- **Output:** the map file records `Redshift` and `content`.
- **Power spectra:** the `.out` normalisation is unchanged. Δ²(k) in mK² (µK² for He) is obtained by multiplying column 2 by (line 3 / line 4)², as `load_ps` in the notebooks does.

Test, on a mock render (N = 32) against an independent Python calculation:
- original and new codes, all compile options: map 3×10⁻⁸, Δ² 5×10⁻⁶;
- identical maps with 1 and 4 MPI ranks;
- each guard aborts as intended.

---

## 5. Approximations that remain (and how big they are)

| Approximation | Size | Status |
|---|---|---|
| SFR linear in time between snapshots (2.0–5.4 Myr apart) | **Measured** (stride 2 / 4 on 64 planes, 352 / 176 snapshots): box-mean bias ≤ 10⁻³ at z ≤ 15 (0.7% at z = 20); voxel-level L1 median 1.2% (H) / 1.6% (He) for stride 2. L1(4)/L1(2) ≈ 2.4, i.e. error ∝ Δt^0.5: snapshot-to-snapshot scatter of the voxel SFR, not smooth-interpolation error. Full-cadence error: voxel level 0.4–1.2% (H), box mean ≪ 0.1%; shell and z′ averaging in J_α reduce it further | Done |
| SFR_eff uses band-integrated age weights × lifetime spectral shape | **Measured with BPASS:** J-weighted error +1–3% (H, box-mean history at z ≈ 10), +2–5% at z ≳ 15 (low Z); He ≲ 1%. Single voxels: +7.5% just after a burst, −11% 20–30 Myr after (H) | Accepted; state in the paper |
| Stars stay in their birth voxel | 3–9 proper kpc drift in 30 Myr vs a 71 proper kpc voxel at z = 10 | Negligible for the H band |
| No emission before the first render (z = 29.43) | J_α is a lower limit for z_obs ≳ 24.7 | State in the text if those z are used |
| He II band: IGM absorption by H I / He I | **Implemented** (4.5). Bracket J vs J = 0: mean δT_b^He changes by 7.7% (z = 6.35) and 21% (z = 4.01) at most. Open: λ₉₁₂ calibration, the star-forming-gas mask, the angular-covariance and resolution tests | Run steps 1–3 of 4.5 |
| AGN f_esc applied in the H band too | The factor is the Hopkins-type obscured fraction (dust affects the UV), so it is kept | Confirm the meaning with Oliver Zier |

---

## 6. What is left

### 6.1 Production
1. **Find a disk with ≥ 600 GB free** (004 is full: 7.3 GB left). Then run `run_SFR_eff.sh` and check its log.
2. ~~Convergence test~~ (done, converged) and ~~`check_sfr_eff_spectrum.py`~~ (done: J_α^H ≲ 3%, ≲ 5% at z ≳ 15; J_α^He ≲ 1%).
3. **J_α runs** for the needed snapshots: H stars, He stars, AGN-H and AGN-He. Check each log as in 4.3.
4. **Compare** the new J_mean with the old outputs at a few redshifts (z ≈ 20, 15, 12, 10, 8, 6, 4).
5. **Regenerate everything downstream:** x_α, T_S, δT_b maps, power spectra, bispectra, VIDs and bubble statistics. Every J_α made before these changes is stale. Make sure the T_b code sums stars + AGN for both species.

### 6.2 Decisions still open
- **Helium IGM absorption** (implemented, 4.5):
  - compare λ₉₁₂,sim with Becker+21, Zhu+23 and Worseck+14 to decide on `LLS_A`;
  - `SF_MASK`: decided ON. Measured at snapshot 300 (z = 6.35): star-forming cells hold 15% of all H I and 52% of the H I in ionized cells, with self_T = 0.10. At snapshot 500 (z = 4.01) they hold 98% of all H I, with self_T = 0.08. Unmasked, the sources' own cells would absorb about 90% of their photons on top of f_esc. At z < 4.75 the opacity would also be almost entirely galaxy ISM smeared over 1.56 cMpc cells, an artifact.
  - rerun `bracket_Tb_He.py` on the `_<TR_TAG>` maps.
- **AGN factor.** Confirm with Oliver Zier that f_esc(L) is the Hopkins/Vogelsberger obscuration factor (one factor for all frequencies).

### 6.3 Manuscript edits (drafts with inline marks were given in the conversation)
- **f_rec citation** (l. ~1451): Pritchard2007 → Pritchard & Furlanetto 2006.
- **Hot ISM / HMXB** (l. 1451, 1512, 1515, 1529, 1533): "included" → "neglected, with the bound", plus the Fragos 2013b/2016-erratum citation.
- **Shell method** (l. 1470–1479):
  - shells, not spheres;
  - an inner shell for the source cell;
  - 200 shells for all codes (l. 1476 still says 400 for AGN);
  - remove the "sphere weights nearby emission 1.5×" paragraph;
  - say the FFT convolution is periodic and that shells larger than the box tend to the box mean.
- **AGN SED:** power-law interpolation across the 20.7–248 eV gap; L_bol integrated to 500 keV.
- **AGN f_esc** (l. ~1505): ≈ 0.15–0.25 (luminosity-weighted ≈ 0.2 at z ≈ 9), not 0.2–0.35. In the Black holes subsection, describe f_esc and the obscuration correction as one factor.
- **Stellar escape fraction** (l. ~1500, 1504): none in the H band; 0.18 in the He band; f_esc = 1 is the ×5.6 upper bound.
- **Metallicity** (l. ~1498): young-star metallicity distribution (20 Myr window, BPASS-grid weights), not the mass-weighted mean of all stars. Update Appendix (metallicity) accordingly.
- **New paragraph on the effective SFR:**
  - each voxel's stars emit according to its own SFR history, convolved with the BPASS band age profile;
  - 300 / 100 Myr windows;
  - SFR linear in time between snapshots;
  - the convergence numbers.
- **J_α^He absorption:** replace "upper limit" wording with the bracket result.
- **l. ~433–435:** says T_S^He ≈ T_K ≫ T_γ for helium, which contradicts the weak He coupling stated elsewhere. Check whether that paragraph is still rendered.

### 6.4 Outside J_α (from `21cm_manuscript/audit_report_2026-10-06.md`, not addressed here)
That audit lists issues downstream of J_α that this work did not touch. The main ones:
- the global δT_b^H computed at the mean neutral temperature instead of as a cell average (#1, #2);
- `T_vol` plotted as T_K (#7);
- the "saturation" statements (#8);
- the He T_b maps being partly in redshift space (#6);
- the He bispectrum input mismatch (#5);
- several text–data inconsistencies (§2 of that report).

Its AGN-SED finding (#3) is fixed (2.3 above). Its helium-coupling finding (#10, collisions dominate) is what `bracket_Tb_He.py` will quantify on the maps.

---

## 7. Test record (mock data, all in the session scratchpad)

| Test | Result |
|---|---|
| Shell port, H / He / AGN vs exact kernel | mean 1.00002; voxel rms 2.5×10⁻⁵ (clustered); P/P_true 0.9999–1.0000 |
| AGN SED, synthetic gap SED | L_bol, ε(Ly-n), J_mean = 1.0000 × exact (H and He) |
| Level 2, single-Z vs old code | J_scalar identical; maps within FFTW noise (≤ 10⁻⁶) |
| Level 2, mixture vs quadrature | 1.00002 (J_mean), 0.9987 (J_scalar) |
| SFR_eff weights vs independent integration | ≤ 9.5×10⁻⁴ |
| SFR_eff pipeline vs dense matrix | 1.1×10⁻⁷ |
| `--plane-range` / `--stride` | bit-identical to full run / to a sub-sampled directory; L1 ratio 4.87 (5 expected) |
| He f_esc | J ratio 0.180000 exactly |
| Input guard | mixed raw/effective input aborts; wrong-band input aborts |
| `check_agn_luminosity.py` | recovers an injected factor 0.25 exactly; on Lumina R = 0.1949 |
| `bracket_Tb_He.py` | one voxel vs independent formula 2×10⁻⁹; J = 0 → ratio 1 |
| `check_sfr_eff_spectrum.py` | flat-spectrum mock gives 1.0000 for every history |
| Transmission: uniform absorber vs analytic exp(−τ); ionized sphere | 3×10⁻⁷; T = 1.0000 inside, 0 outside |
| J_α^He with transmission, stars and AGN; T ≡ 1; species H | 0.20527 vs analytic 0.20513; maps identical to 9.3×10⁻⁷; H unchanged |
| LLS conversion vs independent scipy integral (broken law, single β = 1.5, b₃ = 2) | agrees to < 2×10⁻³; LLS-only T(R) vs analytic: 2×10⁻⁶ |
| `--sf-mask` vs direct calculation; mixed-directory guard | 7×10⁻⁸; H I share exact; mixed masked/unmasked aborts |
| Input/output guards (H code on a mock pipeline; same code in He/AGN) | aborts on: unfinished SFR_eff (`complete = 0`), `--plane-range` or `--stride` output, mixed `tau_max_Myr`, box ≠ NumPixels³, missing spectrum/SED, snapshot out of range, unwritable output (non-zero exit). Older full SFR_eff files without `complete` still accepted, bit-identical J |
