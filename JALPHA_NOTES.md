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

**Production** (`run_Jalpha_prod.sh`): the same binaries, for the snapshot lists of the T_b scripts. `BAND=H` runs H stars + AGN-H for `run_21cm.sh`'s SNAPS (149 snapshots, 298 tasks, ~330 GB); `BAND=He` (with `TR_TAG`) runs He stars + AGN-He for `run_3cm.sh`'s. `bash run_Jalpha_prod.sh count` prints the sbatch line, `status` lists missing outputs. Existing outputs are skipped and each is written as `.part` and renamed on success, so resubmitting redoes only what is missing. Outputs in `/orcd/data/mvogelsb/004/chcho/Jalpha`; point `J_DIR` of the T_b scripts there.

### 4.4 Check scripts

| Script | Question it answers |
|---|---|
| `check_agn_luminosity.py --render .../LuminosityAGN_NNN.hdf5 --snap .../snap_MMM.hdf5` | What is in the AGN render (f_esc, obscuration, units)? Result: R = 0.1949 = ⟨f_esc⟩_L. |
| `check_sfr_eff_spectrum.py [--Z ...]` | How good is SFR_eff × lifetime spectral shape inside the band? (Section 5) |
| `bracket_Tb_He.py --render All_NNN.hdf5 --J Jalpha_He_NNN.hdf5 --J Jalpha_AGN_He_NNN.hdf5 [--write]` | Does δT_b^He depend on J_α^He at all? It compares J as computed (no IGM absorption) with J = 0. |
| `compare_SFR_eff.py REF TEST` | Is the SFR interpolation between snapshots converged? |
| `remap_rsd.py --dir <render_dir> --snap NNN --validate [--out DIR]` | True redshift-space He II density: remaps Density × HeII_Fraction along +z with the `Velocities` render (CIC). First validates the remap against the render's own SpectralDensity and SpectralDensityHI, and finds the velocity unit convention. `--out` writes SpectralDensityHeII, which `PS_3cm_mpi.c` reads when built with `HE_RS_DIR`.
Lumina, z = 7.6: `Velocities` is v_pec/√a, the AREPO internal convention. With it, the remap reproduces SpectralDensity and SpectralDensityHI with power ratio ≥ 0.986 and r = 1.000 for k ≤ 1 h/cMpc. At k ≈ 3, H I power is 0.857 and r = 0.988. The shortcut x_HI(real)·SpectralDensity gives power 1.2–1.3× and r = 0.885 at k ≈ 1. Batch job: `run_remap_He.sh`.<br>`--qtest --rsd` (job `run_remap_qtest.sh`, snapshots 220/300/406/500). `--rsd`: P_rs/P_real and r(k) for gas and H I (exact) and He II (q method, remap, shortcut), and for μ < 0.3 and μ > 0.7. `--qtest` tests the q method: He II_s = [remap(He II)/remap(H II)] × (0.76 SpectralDensity − SpectralDensityHI), exact redshift-space H II times the remapped He II/H II ratio. Check 1 does the same construction on H I against SpectralDensityHI (an upper bound on the error). Check 2 reports how smooth q is, the He II mass kept, and the He power with q remapped vs not. |
| `check_Tb_maps.py --line H\|He --dir <render_dir> --snap NNN --J <J files> --rin RIN_NNN.hdf5 [--real-space] [--he-rs] [--ts-volume]` | Is the C map right? Recomputes δT_b on a few slabs with independent code (astropy constants) from the render and J files and compares cell by cell; exit 1 above 1e-4 per cell. Expect ~2–3e-5 (the C codes' m_H = 1.6735575e-24 g vs the ¹H mass 1.6735328e-24 g). |
| `check_angular_covariance.py --snap NNN --opa ... --sfr ... --render ... --zcsv ...` (job `run_covariance_test.sh`, snapshots 300/406/500) | How good is "shell-averaged emissivity × direction-averaged transmission" in the He J codes? Traces the transmission rays from sampled points, samples the sources along them, and reports J_dir/J_avg (direction by direction over product of averages) for stars and AGN, and the covariance factor by R. AGN are point sources, so their ratio is noisier (few rays hit an AGN at large R). |
| `test_resolution_Tb.py --line H\|He --dir640 ... --dir1280 ... --snap NNN --J ...` (job `run_resolution_test.sh`: H at 60/100/156/220/300, He at 300/406/500) | How much do the 640³ pixel averages bias δT_b? Compares δT_b from ren_1280 averaged 2×2×2 with δT_b from ren_640 (production), same J: ⟨δT_b⟩ ratio (global-signal bias), cell difference and its share in partly ionized pixels, P ratio and r(k); also checks that ren_640 is the mass-weighted average of ren_1280. Physics from `tb_physics.py` (shared with `check_Tb_maps.py`). |
| `check_tb_choices.py --dir <render_dir> --snap NNN [--factor 2]` | How accurate is the He redshift-space approximation (tested on H I, where the exact map exists), and how large is the RSD effect on He? Does mass vs volume weighting matter (signal share in partly ionized cells, T_mass/T_vol)? |

### 4.5 IGM absorption of the He II band

The physics, the derivations and the references are in the doc "How the He II Lyman-band IGM absorption works". Pipeline:

| Step | Script | Output |
|---|---|---|
| 1 | `run_opacity.sh` → `make_opacity_fields.py` | `opacity_NNN.hdf5`: n_HI and n_HeI (comoving cm⁻³) at 320³, every second snapshot. Logs, for star-forming cells, the H I / He I share, `self_T` (fraction of a source cell's 40.8 eV photons that survive their own cell) and the share of ionized-cell H I. |
| 2 | `bash run_transmission.sh compile`, then `sbatch run_transmission.sh` → `compute_transmission_He` | the `lambda_912 (simulation …)` line (mfp912 mode), the `lambda_E` lines (mfpE mode), then `transmission_He_NNN.hdf5` holding T_n2 and T_n4 on 48 radii × 160³ origins |
| 3 | `run_Jalpha_test.sh` with `TRANS_DIR` and `TR_TAG` set | `Jalpha_He_NNN_<TR_TAG>.hdf5` and `Jalpha_AGN_He_NNN_<TR_TAG>.hdf5` |

Variants are kept in separate directories:
- **`SF_MASK=1`** (the default since the measurements below) in `run_opacity.sh` and `run_transmission.sh` makes star-forming cells transparent, as in Lumina's own RT. Otherwise their absorption would be counted on top of f_esc. Outputs go to `*_sfmask`; set `TR_TAG=tr_sfmask` for the J runs. The ray tracer refuses a directory that mixes masked and unmasked opacity files.
- **`LLS_A > 0`** adds unresolved absorbers, with outputs in `*_lls`.
  - **Calibrate it in the He band, not at 912 Å** (corrected 2026-10-14). The simulation's absorbers (SF cells masked) are close to a uniform fog: λ₉₁₂,sim is within 5% of the uniform-IGM value at z = 4 on 160³, 320³ and 640³ grids, so its λ_E/λ₉₁₂ ≈ 13 (uniform H I + He I), while clumpy absorbers have λ_E/λ₉₁₂ = r = 1.59. A simulation can then be too opaque at 912 Å and too transparent in the He band at once (z = 4: λ₉₁₂,sim = 13.9 vs ≈ 36 observed; λ_E,sim ≈ 180 vs r × 36 ≈ 57 proper Mpc).
  - Match the He-band opacity: 1/(r λ₉₁₂,obs) = 1/λ_E,sim + 1/(r λ₉₁₂,extra), i.e. 1/λ₉₁₂,extra = 1/λ₉₁₂,obs − r/λ_E,sim, with λ_E,sim from `compute_transmission_He <snap> <dir> mfpE` (n = 2 line, L/(−ln⟨T(L)⟩)). λ₉₁₂,extra(z) is then fitted by LLS_A [(1+z)/5]^LLS_SLOPE, applied for LLS_ZMIN ≤ z < LLS_ZMAX (all four set in `run_transmission.sh`).
  - **Measured (2026-10-14, 320³, SF masked; λ₉₁₂,obs from Worseck+14, 9.5 proper Mpc at z = 5):** for 3.4 < z < 4.75, λ_E,sim / (r λ₉₁₂,obs) = 2.9–3.2, nearly constant, so λ₉₁₂,extra ≈ 1.5 λ₉₁₂,obs: LLS_A = 57.6 proper Mpc with the default slope −5.4. At z = 5.0 the data need more (λ_E,sim / λ_E,obs = 4.3). Below z = 3.35 (Rahmati self-shielding on in Lumina) the simulation is already too opaque, λ_E,sim / λ_E,obs = 0.4–0.5, so LLS_ZMIN = 3.35. λ_E,sim at 160³ and 320³ agrees to 1%.
  - **Decision (2026-10-14): `LLS_A = 0`**, the simulation's own IGM only, so that no external parameters enter. The calibrated term above is kept as a systematic.
  - The conversion from 912 Å to the band is the effective opacity of a broken power-law f(N): slope 1.5 below 10¹⁷·⁵, 0.8 up to 10¹⁹ (Prochaska+10) and 1.5 above. This gives λ_E/λ₉₁₂ = 1.59 (n = 2) and 1.82 (n ≥ 4).
  - The earlier single power law with β = 1.5 gave 3.5 and 4.8, about 2× too transparent.
  - `-DLLS_FN_B1/B2/B3/LOGN1/LOGN2` change the f(N) slopes and break columns; `-DLLS_BETA=b` restores a single power law.

### 4.6 Brightness-temperature maps (`PS_21cm_mpi.c`, `PS_3cm_mpi.c`)

Run with `run_21cm.sh` and `run_3cm.sh`. Each has a `compile` mode; set `J_DIR`, `TR_TAG` (He) and `OUT_DIR` at the top. Inputs are J^H = `J_alpha` + `Jalpha_AGN_H`, and J^He = `J_alpha_He` + `Jalpha_AGN_He` (the `_<TR_TAG>` files).

Changes from the reviewed versions:
- **He signal is in redshift space** along the 3rd axis, like H: (`HeII_Fraction`/y) · `SpectralDensity`/⟨Density⟩. The fraction is mass-weighted (it was the volume-weighted one).
  - There is no He II density in redshift space, so this multiplies the real-space fraction by the redshift-space gas density. That is exact where x_HeII is uniform over the velocity displacement (~1–3 cells).
  - `check_tb_choices.py` measures the error using the H I analogue, for which the exact map exists.
  - `-DHE_REDSHIFT_SPACE=0` gives real space. **Decided (2026-10-13): He is real space**, set in `run_3cm.sh`; the renders and light-cone have no redshift-space He II. The paper's real-space sentence stays. `remap_rsd.py --rsd` measures how large the redshift-space effect on He would be.
- **No-heating reference** (`-DTK_REFERENCE [-DTK_REF_XTHRESH=1e-3]`; `TK_REF=1e-3` in `run_21cm.sh` builds `bin/PS_21cm_mpi_tkref` and writes `${OUT_DIR}_tkref1e-3`): in cells with x_HII < threshold (default 10⁻³, the neutral-gas criterion of Zier+26 App. B) the temperature of the spin calculation is T_ref = T_CMB,0 (1+z)²/(1+z_dec) (1+δ)^(2/3), z_dec = 124.8, Lumina's own initial-condition law (Zier+26 Eq. 27, p. 16: 54.17 K at z_init = 49; checked: 2.7255 × 50²/125.8 = 54.16 K) with adiabatic compression of each cell. Other cells keep the simulated T; J_α, x_HI, n_e, density unchanged. It removes all heating of that gas (X-rays, shocks, Lyα), so it is a no-heating reference: Lumina-50-NoX neutral gas is ~16 K at z = 15 vs 5.5 K here. Not anchored to the first render (34 K at z = 29.4, already heated; and a fixed pixel is not the same gas at later z). Meaningful at z ≳ 6. Logs print T_ref and the replaced cell and H I-mass fractions; `check_Tb_maps.py --tk-ref 1e-3` checks the map. Run 10⁻⁴ once for the threshold sensitivity (X-rays partly ionize what they heat).
- **Mean-density normalisation (known, left as is, 2026-10-14):** the prefactor uses ⟨n_H⟩ from Ω_b while (1+δ) is Density/⟨Density⟩ (gas). Measured ⟨Density⟩/Ω_bρ_c = 1.0041 at z ≈ 29 (render units: Ω_bρ_c = 1.35893e-9), so the maps are −0.4% at high z; stars and BHs lower ⟨ρ_gas⟩ later, so the offset turns positive at low z (≲ +1–2% at z ≈ 3; read `global_gas` in a low-z log). Fix if wanted: use ⟨n_H⟩ from `global_gas`.
- **H signal switch:** `-DH_REDSHIFT_SPACE=0` in `PS_21cm_mpi.c` gives real-space H, 0.76 Density (1 − HII_Fraction) (mass-weighted). The code default is redshift space (SpectralDensityHI).
  - **Decided 2026-10-14: real space** (`run_21cm.sh`, maps in `Tb_maps_prod/H_real`). The default puts the spin factor (1 − T_γ/T_S) of the real-space pixel on the H I that the velocity shift moved there from 1–3 pixels away, so it is neither exact nor consistent; real space is exact and matches the real-space He maps.
- **He S_α** uses the He II Gunn–Peterson depth 6×10³ x_HeII((1+z)/7)^1.5, as in the paper. It was the H I depth.
- **Spin-temperature inputs** are the mass-weighted `HII_Fraction`, `Temperature`, `HeII_Fraction` and `HeIII_Fraction`; `-DTS_VOLUME_WEIGHTED=1` restores the volume-weighted fields. A field missing as `<field>/<field>_NNN.hdf5` is read from `All/All_NNN.hdf5`.
- **Cosmology:** h comes from the header (it was 0.6774 hard-coded, which made n_H 0.24% high). Ω_m and Ω_b come from the header, or else Lumina's 0.30964 / 0.048975; the old Planck 2015 values put the prefactor +0.53% off.
- **He prefactor:** 1.67/√10 µK, the same as `global_snaps_clean.ipynb`. It was 0.53, labelled mK.
- **Exact prefactors (2026-10-14), both codes:** δT_b = τ T_S (1 − T_γ/T_S)/(1 + z) with τ T_S = 3c³h A₁₀ n_HI/(32π k ν² H(z)) for H and c³h A₁₀ n_³HeII/(32π k ν² H(z)) for ³He⁺ (inverted levels: n_l g_u/g_l = n/4), [³He/H] = 1e-5, H(z) with Λ. Constants: T*₂₁ = 0.0681687 K and A₁₀ = 2.86888e-15 s⁻¹ (were 0.068 and 2.85e-15: x_α and x_c 0.41% lower now), T*_He = 0.4158856 K, m_H = 1.6735575e-24 g. Tested against an independent calculation (astropy constants): 1.000000. A₁₀(³He⁺) = 3 (ν_He/ν₂₁)³ A₂₁ = 1.95434e-12 s⁻¹ (electron-spin M1, g_upper = 1; was 1.959e-12; literature 1.95e-12 McQuinn & Switzer 2009, 1.96e-12 Bagla & Loeb 2009). Change from the textbook forms: H −1.3% (z = 3) to +0.4% (z ≥ 9), He −2.4% (z = 3) to −0.7% (z ≥ 15). The optical depth is kept linear.
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
| He II band: IGM absorption by H I / He I | **Implemented** (4.5). Bracket J vs J = 0: mean δT_b^He changes by 7.7% (z = 6.35) and 21% (z = 4.01) at most. Resolved: λ₉₁₂/λ_E calibration (LLS_A = 0 chosen), the star-forming-gas mask (on), resolution (converged 160³–640³), angular covariance (J^He low by ≤ 30% at z = 6.3, ≤ 3% at z ≤ 5; ≤ 0.3% on δT_b^He) | Run steps 1–3 of 4.5 |
| AGN f_esc applied in the H band too | Zier+26 §3.2.2, Eq. 15: L_bol,obs/L_bol = 0.3 (L_bol/10⁴⁶ erg s⁻¹)^0.07 (Hopkins+07 form, Vogelsberger+13 parameters), applied to L_bol, with the SED integrated over the bins afterwards: one frequency-independent factor, no further source-level f_esc. Our grey f_esc(L) in both bands is what Lumina did (R = 0.1949 luminosity-weighted fits 0.16–0.30 for 10⁴²–10⁴⁶ erg s⁻¹) | Confirmed from the paper (2026-10-14) |

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
- ~~**AGN factor.**~~ Resolved from Zier+26 §3.2.2 (Eq. 15): one luminosity-dependent obscuration factor on L_bol, independent of frequency; the paper also joins the 600 Å EUV and 50 Å X-ray SEDs by a power law, as our SED interpolation does.

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
| `remap_rsd.py --qtest --rsd` on Lumina (job 25780817, snapshots 220/300/406/500/600/700) | P_rs/P_real at k ≈ 0.05–0.3 h/cMpc. Gas (exact): 1.72–1.88 at every z (linear Kaiser for b = 1: 1.87). **H I (exact):** 0.69–0.84 (z = 7.6), 0.88–0.91 (6.3), 1.09–1.14 (5.0), 1.16–1.23 (4.0–3.0); along the line of sight (μ > 0.7) 0.35–0.68 at z = 7.6 and 1.4–1.7 at z ≤ 4. **He II (q method; remap and shortcut within ~±15%):** 1.13–1.36 (z = 7.6–6.3), 1.7–2.1 (5.0), 0.93–1.22 (4.0), 0.93–1.06 (3.4–3.0). At k ≳ 1 the real- and redshift-space maps decorrelate (r < 0.9), so ratios there are not a simple boost. Both T_b lines are real space: this is the caveat to state |
| `check_Tb_maps.py --tk-ref 1e-3` (mock, 43% of cells below the threshold) | vs an independent hand calculation of the reference map: ≤ 2.2×10⁻⁵ per cell; the same map checked without `--tk-ref` → MISMATCH |
| `test_resolution_Tb.py` (mocks) | fine fields constant inside 2×2×2 blocks: identical maps (ratio 1.00000, r = 1); half hot ionized / half cold neutral gas inside every pixel: the 640³ map is in emission (+10 mK) where the 1280³ truth is in absorption (−17 mK), the expected sign of the bias; ren_640 = average check 10⁻⁷ |
| `check_Tb_maps.py` after moving the physics to `tb_physics.py` | identical results (H reference 2.15×10⁻⁵, He 3.00×10⁻⁵, negative control MISMATCH) |
| `check_angular_covariance.py` (mocks) | uniform absorber: J_dir/J_avg = 1.0000 exactly; sources only in a transparent half-box: 1.027 (> 1, rising with R), the expected sign |
| `check_angular_covariance.py` on Lumina (2026-10-14, 1000 points, 192 directions, ionized points, SF masked) | J_dir/J_avg, J-weighted: z = 6.35: stars 1.34, AGN 1.47 (covariance rising from 1.04 at R < 5 to 1.8–2.2 at 20–50 cMpc); z = 5.0: 1.024 / 1.030; z = 4.01: 0.992 / 0.993. Spectral slope 0–3 changes it by ≤ 0.006. So the product-of-averages J^He is biased low by ~25–30% mid-reionization, ≤ 3% at z ≤ 5. Effect on δT_b^He (× the Lyα share 0.9% / 7% / 18%): +0.3%, +0.2%, −0.1% — negligible; not corrected. Extended (stars / AGN, J-weighted, ionized points): z = 8.31: 1.33 / 2.04 (13 points), 7.08: 1.35 / 1.40 (119), 5.81: 1.20 / 1.26 (761), 5.31: 1.07 / 1.08 (985); z = 9.85 has 1 ionized point of 1000 (not measurable). So C(z) ≈ 1.3–1.35 for z ≈ 6.3–8.3, falling to ≈ 1 by z ≈ 5; (C − 1) × Lyα share ≲ 1% wherever measured (share at 5.3–5.8 to confirm from the production He J) |
| `check_Tb_maps.py` (2026-10-14, 24³ mock with Lumina units) | He: vs `bracket_Tb_He.py` (separate code), per cell ≤ 3.0×10⁻⁵; H: vs a hand calculation, ≤ 2.2×10⁻⁵ (both the m_H difference); wrong switch (`--he-rs`) → MISMATCH (rel 1.0) |
| `check_sfr_eff_spectrum.py` | flat-spectrum mock gives 1.0000 for every history |
| Transmission: uniform absorber vs analytic exp(−τ); ionized sphere | 3×10⁻⁷; T = 1.0000 inside, 0 outside |
| J_α^He with transmission, stars and AGN; T ≡ 1; species H | 0.20527 vs analytic 0.20513; maps identical to 9.3×10⁻⁷; H unchanged |
| LLS conversion vs independent scipy integral (broken law, single β = 1.5, b₃ = 2) | agrees to < 2×10⁻³; LLS-only T(R) vs analytic: 2×10⁻⁶ |
| `--sf-mask` vs direct calculation; mixed-directory guard | 7×10⁻⁸; H I share exact; mixed masked/unmasked aborts |
| Input/output guards (H code on a mock pipeline; same code in He/AGN) | aborts on: unfinished SFR_eff (`complete = 0`), `--plane-range` or `--stride` output, mixed `tau_max_Myr`, box ≠ NumPixels³, missing spectrum/SED, snapshot out of range, unwritable output (non-zero exit). Older full SFR_eff files without `complete` still accepted, bit-identical J |
