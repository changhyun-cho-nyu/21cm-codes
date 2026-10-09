# Issues in the LUMINA 21cm/3.46cm draft (version of 2026-08-20)

Result of a detailed audit of the draft against the analysis code and data in
`/orcd/data/mvogelsb/004/chcho/` and the LUMINA renders in `/orcd/data/mvogelsb/005/`.
For context: the core physics implementation was verified and is sound — the appendix equations,
constants, and estimators check out against first-principles derivations, and the fields, power
spectra, VID moments, and MFP machinery were reproduced end-to-end from the raw 640³ renders with an
independent implementation (VID moments agree to 4–5 digits, Δ²(k) to a few percent). The items
below are what remains, ordered by impact. Each lists the analysis script(s) the paper was compared
against.

---

## A. Issues that affect stated conclusions

### A1. Bubble-size figure (`figs/BSD.png`): unit and convention errors in the Fourier-space estimators

**Compared with:** `BSD.ipynb` (plotting; hardcoded radius arrays), `PS_21cm/bispec/serial_iso_fHII.py`
and `PS_21cm/bispec*/new_J_alpha/serial_iso_Tb.py` (bispectrum runs, `BOX_LEN = 500.0`),
`PS_21cm/PS2Bubble.ipynb` and `PS_21cm/PS_bubble_size_HII.dat` / `PS_bubble_size_HeIII.dat`
(power-spectrum peak radii), `bubble_statistics_hydrogen.csv` (MFP statistics),
raw bispectrum outputs in `PS_21cm/bispec/` (sign-flip crossings re-derived).

1. **Bispectrum sign-flip radii are divided by h although they are already in cMpc.** The bispectrum
   scripts pass `BOX_LEN = 500.0` (cMpc; render header: BoxSize = 338300 ckpc/h ÷ h = 500.000 cMpc),
   so the bispectrum k are in cMpc⁻¹ and π/k₃,ₜ is already in cMpc. `BSD.ipynb` nevertheless divides
   (`R_bispec_h = np.array([1.08827, 2.01348, 2.12863, 4.25726]) / hubble`, same for He), inflating
   the green diamonds by 1/h = 1.478. Corrected, they fall *below* the MFP median (z = 9.16: plotted
   1.61 cMpc, correct 1.09, median 2.0; z = 7.2: plotted 6.29, correct 4.26, median 5.23) — reversing
   the claim that the sign-flip estimates "track the mean and median MFP scales."
2. **Power-spectrum circles use R = 2π/k_peak while the text defines R ∼ π/k_peak**
   (`PS2Bubble.ipynb`: `R_bubble = 2.0*np.pi/k_peak`; table headers state "R = 2*pi/k_peak"). With
   the stated π/k the circles halve and no longer systematically exceed the MFP median for hydrogen
   (z = 6.35: 2π/k → 29.6 cMpc vs median 16.2; π/k → 14.8 ≈ median). As plotted, the two estimators
   in one panel use inconsistent π- and h-conventions.
3. Consequently, **all bispectrum k and r values quoted in the text/captions as h cMpc⁻¹ / h⁻¹ cMpc
   are actually cMpc⁻¹ / cMpc** (32% offset) and are not commensurable with the (genuinely h-unit)
   power-spectrum section.

### A2. MWA noise floor plotted in mK on a mK² axis (`figs/ps_hydro.png` panel b); detectability claims

**Compared with:** `PS_21cm/visual_PS.ipynb` (obs-overlay cell: helper `_a2`, hardcoded
Nunhokee/HERA/Mertens tables) and Table 4 of Nunhokee et al. 2025 (arXiv:2505.09097).

The MWA 1σ floor is computed as `np.sqrt(_a2(thermal) + _a2(sample))` of the tabulated mK amplitudes
and plotted on the mK² axis. Nunhokee Table 4 gives Δ²_therm = (2.7)², Δ²_sample = (13.8)² mK² etc.,
so the correct floor at z = 6.5, k = 0.142 h/Mpc is ≈ 2.7² + 13.8² ≈ 198 mK², not the plotted 14.1 —
understated by ×7–15 across bins (the same cell treats the same table's upper limits correctly by
squaring them; with the plotted floor MWA's own 2σ limits would sit ~200σ above their noise).
Consequences: (i) the claim that the signal "rises above the MWA noise floor … in principle
detectable at current integration depths" near k ≈ 0.2, z ∼ 7 does not hold — the predicted
Δ² ≈ 8–10 mK² is ~15–20× below the corrected floor; (ii) the new sentence "HERA is already reaching
the sensitivity thresholds necessary to detect the predicted signal at z ∼ 7" conflicts with the
correctly plotted HERA overlay (1σ floor 372 mK² at z = 7.05, k = 0.44 vs predicted ≈ 13 mK²).
`figs/ps_hydro.png` was not regenerated in this revision, so the buggy floor is in the current figure.

### A3. f_heat = 0.7/0.3 curves in Fig. `global.png` (c) are linear blends of the endpoint signals, not evaluations of intermediate heating efficiencies

**Compared with:** `visual.ipynb` (panel-c cell: `Tb_mix = (1.0 - f) * Tb_ad_interp + f * Tb_full`),
`LUMINA_all.csv` (fiducial global), `THESAN-XL_ad.csv` (adiabatic global); cross-checked with an
independent mean-field recomputation using the coupling machinery of `global_snaps.ipynb`
(κ tables from `compute_J_alpha_H/collisions.c`), which reproduces the −10.5 mK adiabatic trough.

The plotted curves are linear interpolations of the two final brightness-temperature curves. δT_b is
nonlinear (concave) in T_K while the temperature response to a scaled heating rate is linear per
cell (T(f) ≈ T_ad + f·ΔT), so the blend overstates absorption for any spatial heating pattern
(chord below a concave function). A consistent mean-field recomputation gives ≈ +2.6 mK at z = 15
for f_heat = 0.3 (no prominent trough — 30% of the fiducial heating still keeps T_K ≈ T_γ there)
versus the plotted −6.4 mK; in the opposite binary-patchy limit the true curve is even closer to
the fiducial. Statements not supported by the construction: "we evaluate intermediate heating
efficiencies"; "decreasing f_heat delays the T_K = T_γ thermal crossing" (no T_K is computed for
these curves); "the f_heat = 0.3 case … recovering a prominent absorption feature"; and the claim
about f_heat effects on the 21cm power spectrum, which cites no figure and has no corresponding
computation in the analysis directory. Note the blend *is* exact for a different model ("a fraction
f of the volume receives full heating") — one fix is relabeling as an interpolation/bracket; the
better fix is cheap: per-cell T(f) = T_ad + f·ΔT and rerun the existing T_S pipeline.

---

## B. Quantitative issues behind the figures

### B1. The bold "global" curve is a product of separately averaged means (biased ~+15–18% during overlap)

**Compared with:** `global_snaps.ipynb` / `LUMINA_all.csv` (T̄_b = 27·x̄_HI·(1 − T̄_γ/T_S(mean
inputs)); no ⟨x_HI(1+δ)(1−T_γ/T_S)⟩ term, no (1+δ)) versus the true volume average recomputed
directly from the 640³ renders (`SpectralDensityHI`, `Density`, `HII_VolumeFraction`,
`TemperatureVolumeWeighted` + the archived `Jalpha_H`/`Jalpha_AGN_H` boxes).

Measured bias of the mean-field curve: +14.7% / +17.7% / +16.8% at x̄_HII = 0.1 / 0.5 / 0.9
(e.g. 22.8 vs 19.9 mK at z = 9.2). The caption's "volume-weighted global average" should be
qualified or the curve replaced by the true average (the per-cell machinery exists). Also, the T_K
entering T_S is the neutral-gas-only mean (`PS_21cm/new_parallel_PS21cm/global/compute_neutral_temp.c`)
— defensible for H, but not for the He signal, whose ions live in ionized gas.

### B2. Adiabatic reference anchored ~1.8× above the canonical adiabatic temperature

**Compared with:** `PS_21cm/TK_ad/get_T_ad.c` (selects x_HII < 0.01 cells; T ∝ (1+z)² anchored to
snapshot 0) and `temp_ad.csv` (z = 29.43 box-mean anchor = 34.2 K).

The recombination-relic adiabatic value at z = 29.4 is ~17–20 K, so T_K,ad — and with it the
−10 mK trough at z ≈ 15 — is substantially shallower than a canonical no-heating calculation
(rough estimate −20 to −30 mK). This also compresses the f_heat bracket of A3. Above z = 29.4 the
interpolation clamps T constant. At minimum the warm anchor should be disclosed.

### B3. High-z MFP bubble sizes are resolution-dominated; two quoted numbers don't match the outputs

**Compared with:** `BSD.ipynb` (its own diagnostic prints the sub-cell ray fraction),
`mfp-bubbles-thesan/lumina-mfp-hist-grid.cc` (method — verified correct), and
`bubble_statistics_hydrogen.csv` / `bubble_statistics.csv` (statistics).

At z = 9.2, 42% of MFP rays are shorter than one 0.78 cMpc cell (67% at z = 12), so
"R_ion ∼ 1.5 cMpc at z ≈ 9.2" depends entirely on the treatment of the unresolved bin (archived
variants give 1.05–1.95 cMpc) — a caveat should accompany these numbers. "R_ion ∼ 40 cMpc at
z ≈ 5.8" matches no archived statistic (median 61–67, log-mean 49–53, mean 91–100 cMpc at
z = 5.81–5.83; the median is ≈40 near z ≈ 6.0). Near completion, rays run to two box lengths
(mean 178 cMpc at z = 5.65), tracing percolated periodic topology rather than bubble radii.

### B4. Lyα background: sphere averages used where shell averages belong; appendix source list overstates the implementation

**Compared with:** `compute_J_alpha_H/compute_Jalpha.c` (shell loop ~lines 670–710: per shell,
`smooth_tophat(..., R_mid)` = cumulative top-hat *sphere* average, weighted by the shell's dz; no
differencing of consecutive radii), `compute_J_alpha_He/compute_Jalpha_He.c`,
`compute_J_alpha_AGN/compute_Jalpha_AGN.c`, and the consumers `compute_J_alpha_H/PS_21cm_mpi.c`,
`compute_J_alpha_He/PS_3cm_mpi.c` (sum exactly two J boxes: stellar + AGN).

1. The sphere-average discretization keeps the global mean exact (verified to ~1% in the run logs)
   but distorts the kernel to (3/2)(1 − r²/R²_max)/(4πr²) instead of 1/(4πr²): nearby emission is
   over-weighted by 50% (with mixed retarded times), emission near R_max gets zero weight, and the
   [0, R_min = 1 cell] lightcone segment is dropped (clipping the high-J_α tail in star-forming
   voxels). This matches the cited Mesinger et al. (2011) scheme, but the fix is essentially free:
   filter each shell with the normalized difference of consecutive top-hat windows,
   W = [V₂W_TH(kR₂) − V₁W_TH(kR₁)]/(V₂ − V₁), and add the central segment. Impact is confined to
   small-scale x_α fluctuations in the weakly coupled regime (z ≳ 10).
2. Appendix A states the hot ISM, HMXBs, and AGN all enter J_α ("summing the emissivity over all
   contributing populations", including "through secondary electrons"). The pipeline computes and
   sums only stellar (BPASS) and AGN direct-UV boxes; no hot-ISM/HMXB emissivity or
   secondary-electron Lyα term exists in the code. Either implement or reword.
3. The per-cell helium code computes S_α with the *hydrogen* Gunn–Peterson depth
   (`PS_3cm_mpi.c`: `compute_S_alpha(z, T_K, f_HI)` → τ_GP = 3×10⁵·x_HI) instead of the appendix's
   He II form (6×10³·x_HeII). Verified numerically inconsequential (<0.05% on the He signal), but a
   species-level inconsistency; the global notebook uses the correct form, so field-level and global
   He results differ in method.

### B5. RSD claims are broader than the implementation

**Compared with:** `compute_J_alpha_H/PS_21cm_mpi.c` (reads `SpectralDensityHI`; no velocity data
used anywhere in the analysis), `compute_J_alpha_He/PS_3cm_mpi.c` (uses the real-space
`Density`-based field for He), `global_snaps.ipynb` (no RSD term in the global curve); plus a direct
anisotropy test on the renders (P(μ>0.8)/P(μ<0.2) = 4.5 along the z-axis at z = 9.2, ≈1 for x/y),
confirming `SpectralDensityHI` is a redshift-space field.

"Explicitly accounted for down to the particle resolution" holds only partially: RSD enters the
hydrogen maps through the redshift-space renders (at the 640³ / 0.78 cMpc grid resolution, not
particle level); the global curve contains no RSD term; the helium field is real-space; the T_S
ingredients are real-space everywhere. Suggest rewording.

### B6. Panel (c) sensitivity curves (`figs/PS_H_z.png`) and the accompanying claim

**Compared with:** `PS_21cm/PS_redshift.ipynb` and the input files `PS_21cm/obs/simu_noise_ps_*.txt`
(headers state Δ², Δ²_err in mK² with no σ-level).

The σ-level of the plotted noise column is not documented — if it is 1σ, the "2σ" labels are a
factor 2 low; worth confirming against the Barry (2022) source. The claim that the signal "lies
above the instrument sensitivity thresholds at both wavenumbers" across the bulk of the EoR is
contradicted by the figure itself for LOFAR at k = 0.1 (LUMINA is below the LOFAR 1000 h curve for
z ≳ 7.3).

---

## C. Smaller inconsistencies and presentation issues

1. **New bispectrum equations** (Sec. "Bispectrum"; compared with the BiFFT source used by
   `PS_21cm/bispec/serial_*.py`): the definition divides by the Dirac delta,
   B = ⟨ΔΔΔ⟩/[(2π)³δ_D], which is ill-formed — standard form is ⟨ΔΔΔ⟩ = (2π)³δ_D·B. The
   normalized-bispectrum equation B̃ = B/√(k₁k₂k₃P₁P₂P₃) is dimensionally wrong (Mpc³); the code
   computes the dimensionless B·√(k₁k₂k₃/(P₁P₂P₃)) — √(k₁k₂k₃) belongs in the numerator.
2. **Bispectrum robustness/documentation** (compared with `visual_bispec.ipynb`, `BSD.ipynb`, raw
   outputs in `PS_21cm/bispec*/`): the sign-flip radius at each redshift is read from a different,
   undocumented choice of k₁ (results vary by ~2× across available k₁ at fixed z); 11 θ-bins, no
   uncertainty estimates; amplitude comparisons between x and δT_b bispectra are made on the
   normalized statistic (weighted by the power spectra of different fields) — worth a caveat.
3. **VID section** (compared with `PS_21cm/one-stat.ipynb` and
   `PS_21cm/vid_stats_summary_156_406.csv`, `All_vid_stats_summary_He.csv`,
   `PS_21cm/subvol_VID.ipynb`): (i) the He midpoint is quoted as "x̄_HeIII ≈ 0.5, z ≈ 4.3" but the
   analysis' own snapshot↔x̄ mapping puts x̄ = 0.5 at z ≈ 4.0 (z = 4.3 → x̄ ≈ 0.3); (ii) in the current
   summary file the hydrogen skewness and kurtosis minima sit at z = 7.29 (snap 237) and z = 7.01
   (snap 255), i.e. x̄_HII ≈ 0.38 and ≈ 0.46 by the analysis' own snapshot↔x̄ mapping, so
   "consistent with the midpoint (x̄ ≈ 0.5)" overstates the coincidence; (iii) the text says "the variance drops by
   nearly two orders of magnitude" while panel (b) now plots the standard deviation (~1 visible
   order); (iv) the plotted PDFs bin only voxels above an undocumented floor of 0.001
   (`PHYSICAL_FLOOR` in `one-stat.ipynb`; mK for H, reused as μK for He) while normalizing by all
   voxels, so up to ~35–55% of the probability mass (the zero-signal spike discussed in the text)
   lies below the plotted range and the curves do not integrate to 1 over the shown domain — state
   the convention; (v) the box-convergence sub-volume test is performed only at x̄ = 0.1 per species
   (`subvol_VID.ipynb`), not at the late epochs where the moments are most tail-dominated.
4. **Global figure caption says 125 sub-volumes; 124 are plotted** (compared with
   `visual.ipynb` + the `LUMINA_ren5_*.csv` set from `(LUMINA) cell_by_cell.ipynb`: one CSV is
   excluded as corrupt). The tracks themselves are consistent with the bold curve (their mean
   matches it to <0.2%).
5. **Appendix A wording vs code** (compared with `compute_J_alpha_H/compute_Jalpha.c`,
   `functions.c`, `global_snaps.ipynb`): τ_GP is described with "globally averaged" fractions but
   coded with local per-cell fractions (physically fine — align the text); the BPASS emissivity is
   evaluated at fixed Z = 10⁻⁴ for the whole history (`compute_Jalpha.c`, `METALLICITY 1e-4`)
   although the text says "the same BPASS SEDs as used on-the-fly" (the simulation used local
   metallicities); a sentence clarifying that no escape fraction is applied to the non-ionizing
   Lyman-band emissivity would prevent confusion with the quoted f_esc = 0.18; f_rec values for
   n ≥ 6 deviate ≤1.9% from the Pritchard & Furlanetto table (net ≲0.5% on J_α); the AGN J_α code
   uses 400 shells where the text says 200.
6. **Text nits:** the introduction says the ³He abundance is "∼10⁻⁵ relative to ⁴He" — it is
   relative to hydrogen (as Eq. 2 and Sec. 4.2 state); "spontaneous-decay rate … roughly two orders
   of magnitude" — the ratio is 1.959×10⁻¹²/2.85×10⁻¹⁵ ≈ 690 (nearly three); the non-isosceles
   captions call the cosθ = −0.5 line "the equilateral configuration" although the body text
   correctly notes such a triangle is never equilateral; the BiFFT README requests Watkinson et al.
   2017 (+ Scoccimarro 2015, Sefusatti et al. 2016) as the citation rather than "Watkinson 2021";
   the MFP parameters (ionization threshold 0.5, 192 HEALPix directions from every ionized cell,
   sub-cell interpolation) are not stated in the paper and are needed for reproducibility.
