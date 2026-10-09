# LUMINA 21 cm / 3.46 cm manuscript — independent audit (2026-10-06)

**Scope.** I read `oja_template.tex` end to end and checked the methods, numbers and figures against the code and data that produced them. I did not use the earlier audit files in this directory.

Code and data I traced:

- **Global signal:** `global/` (C mean/neutral-temperature codes, `subvol_125/lumina_subboxes.py`) and `21cm_analysis/global_snaps_clean.ipynb`.
- **Cell-by-cell J_α and T_b:** `cell_by_cell/compute_Jalpha{,_He}.c` and the binaries that made the maps. The H T_b source is `ren_1280/compute_J_alpha_H/PS_21cm_mpi.c`, which is md5-identical to the binary in `cell_by_cell/hydrogen`. The He T_b source is `corrupted_files_do_not_use/compute_J_alpha_He/PS_3cm_mpi.c`, which is md5-identical to `cell_by_cell/helium/PS_3cm_mpi`. AGN J_α comes from `corrupted_files_do_not_use/compute_J_alpha_AGN/compute_Jalpha_AGN.c`.
- **Figure notebooks:** all of them in `21cm_analysis/`, plus the scripts stored next to their outputs in `PS_21cm/bispec*/final_Tb/` and `mfp-bubbles-thesan/`.

**Independent checks I ran.** I re-derived constants and prefactors and re-computed milestones from the CSVs. I parsed the T_b run logs for map means and compared map-pipeline J_α files with the global pipeline. I also read small slabs of the render cubes (about 4% of the box) to test averaging and redshift-space effects. Total I/O was about 1.3 GB.

**Section references.** "§" means a section of this report; "paper Sec." means a section of the manuscript.

**Severity scale.**
- **A:** changes a result, figure, or headline claim.
- **B:** a text–data or internal-logic inconsistency that a referee would catch.
- **C:** minor, cosmetic, or documentation.

Line numbers refer to `oja_template.tex`.

---

## 0. Executive summary (most important first)

| # | Sev. | Finding | Where |
|---|---|---|---|
| 1 | A | The global δT_b^H in Fig. 2(c) is evaluated at the *mean* neutral-gas temperature, not as a volume average of δT_b. With a hot minority of cells this biases the signal strongly toward emission at z ≳ 11. A cell-level average on real render slabs gives **absorption (−1 to −3 mK) at z ≈ 13–17**, where the paper shows +2 to +8 mK. The "no trough / z_h ≳ z_α / reversed ordering" conclusions depend on this. | paper Sec. 3.1, 4.4–4.5, Concl. |
| 2 | A | The adiabatic reference δT_b,ad and the f_heat = 0.3 curve inherit the same construction, but much more severely. With 95–99% of neutral cells forced to T_ad, the mean T is set by the 1–5% hot cells. The "−45 mK trough at z ≈ 13" and the later "rise into emission" are averaging artifacts: with T = T_ad the reference reaches about **−140 mK near z ≈ 9** and stays in absorption until z < 7. It is also threshold-sensitive (−37 to −45 mK) and anchored to a gas temperature about 2× the standard adiabatic history. | paper Sec. 3.1, 4.4–4.5 |
| 3 | A | **The AGN J_α^He in the cell-by-cell maps is about 11.4× the global-pipeline value.** `compute_Jalpha_AGN.c` interpolates the AGN photon spectrum *linearly* across a 20.7 → 248 eV gap in the SED table. The global notebook interpolates log-log. As a result the He maps (VID, PS, bispectrum) have AGN-dominated He II Lyα coupling at z ≲ 4.5 (58% of J_α^He at z = 4 versus 11% in the global pipeline). The ren_1280 version of the code has the same interpolation. | Methods/App. A, all He map results |
| 4 | A | The global He curve (Fig. 2d, "≈0.03 µK at z ≈ 5.5") does not equal the mean of the He maps used everywhere else. Map means are 17–49% higher at z = 4–6.6 and peak at **0.038 µK at z ≈ 6.2**; they are 1.7–3.3× higher at z < 3.5. | paper Sec. 3.1–3.3 |
| 5 | A | Wrong inputs in the **He bispectrum figures**. The T_b panels use *older* He maps (Sep 16–21, now in `corrupted_files_do_not_use/compute_J_alpha_He/Tb_He`), not the maps used for the He PS/VID; their power differs by −18% to +11%. In Fig. `non_iso_He` the x_HeII panel was computed with `BOX_LEN = 500` (cMpc⁻¹) and the T_b panel with 338.3 (h cMpc⁻¹). **The two panels therefore show different triangles** (k₁ = 0.44 vs 0.30 h cMpc⁻¹). | paper Sec. 3.4 |
| 6 | A | The **He T_b maps are partly in redshift space.** `PS_3cm_mpi.c` multiplies x_HeII by `SpectralDensity`, which I verified is anisotropic along the 3rd axis. Line 231 states that the helium field is in real space. | paper Sec. 2.4, 3.3 |
| 7 | B | Fig. 2(a) plots `T_vol` (all gas, dominated by photo-heated gas) as "T_K", but T_S^H is computed with the neutral-gas T. At z = 9.2 the panel shows 1739 K while the relevant T_K is 243 K. | Fig. 2 |
| 8 | B | "T_S locked to T_K once x_α ≫ 1" and "saturated regime by z ≲ 12 / z ≲ 10" are not what the data show. At z = 12, (1 − T_γ/T_S) = 0.47; at z = 10 it is 0.76; ≥ 0.9 only at z ≲ 8.3. T_S stays at 0.55–0.7 T_K throughout the EoR. | l. 503, 519, 1296, 1553 |
| 9 | B | "At z ≳ 9 the PS is dominated by matter density" contradicts Fig. `PS_H_z`, where power *rises* toward higher z (10 → 19 mK² at k = 0.5) — a spin-temperature signature. | l. 719, 734 |
| 10 | B | The helium-coupling narrative conflicts with the paper's own numbers. Collisions supply about 70% of the coupling at the peak and stars about 99% of J_α^He at z ≥ 5; removing *all* Lyα lowers the peak by only 30%. So the "signal generated near luminous quasars" and the "30× above Spina+2025 primarily because of the resolved He II Lyα background" are not supported. "Stars exceed AGN by a factor of a few" is off: the adopted inputs give about 23× in emissivity and 45× in J at z = 5. | l. 764, 989, 1023, 1184, 1285, 1559 |
| 11 | B | Lightcone (Fig. 1). All outer-lightcone redshifts are shifted by −0.253 (labels *and* physics) to hide the 4.75–5.0 gap. T_b uses a global J_α from an old CSV multiplied by local overdensity. x_HeIII = 1 − x_HeII treats neutral He as He III, inflating n_e in neutral gas by orders of magnitude (≈ 0.16 n_H vs ≈ 10⁻⁵ n_H). | l. 456–463 |
| 12 | B | Several text statements contradict the data (details in §2). These cover the end of reionization (5.2 vs 5.5 vs 5.5–6), the He I ionization physics, He II "shells", the He σ and BC behaviour, "kurtosis ↔ bispectrum", the "1–2 orders of magnitude" gap to current limits, and the noise-curve provenance. | various |

What checked out (§6): constants, Eq. 1 and 2 prefactors, J_α machinery, BPASS normalization, PS normalization, MFP set-up, most quoted milestone numbers, and the H-pipeline consistency between maps and global.

---

## 1. Global signal (Fig. 2, §3.2, §4.4–4.6)

### 1.1 [A] Global δT_b^H uses the mean temperature (Jensen bias)

**What the code does.** `global_snaps_clean.ipynb` (cells 3–7) and `lumina_subboxes.py` take
- T_K = `neutral_temp_means.txt`, the arithmetic mean of `TemperatureVolumeWeighted` over cells with x_HII < 0.01;
- x̄_HI = 1 − ⟨x_HII⟩ and mean densities.

They then compute a single T_S and δT_b ∝ x̄_HI (1 − T_γ/T_S(T̄)), with no (1+δ) factor.

**Why this is a problem.** (1 − T_γ/T_S) is concave in T. The neutral-gas temperature distribution has a small hot tail (cells near sources and partially ionized cells), which pulls the mean far above the typical cell. The observable global signal is ⟨δT_b⟩ over cells, not δT_b(⟨T⟩).

**Test.** I read a 128×128×640 column (about 4% of the volume) of `HII_VolumeFraction`, `TemperatureVolumeWeighted` and `Density` at 7 snapshots. I used the *global* J_α (so J_α fluctuations are ignored), local S_α, local x_c and local δ, and averaged δT_b over cells.

| z | published global δT_b [mK] | cell-averaged ⟨δT_b⟩ [mK] | mean T_neutral [K] | median T_neutral [K] | T_CMB [K] | frac. neutral cells with T < T_CMB |
|---|---|---|---|---|---|---|
| 17.0 | −0.11 | **−0.91** | 48.5 | 30.9 | 49.0 | 0.69 |
| 15.0 | +2.15 | **−2.91** | 58.4 | 33.4 | 43.5 | 0.60 |
| 14.0 | +5.24 | **−2.95** | 65.7 | 35.3 | 40.8 | 0.55 |
| 13.3 | +8.15 | **−1.50** | 72.2 | 37.5 | 39.0 | 0.52 |
| 12.0 | +14.3 | +6.95 | 92.8 | 47.9 | 35.4 | 0.38 |
| 11.0 | +18.2 | +14.9 | 122 | 69.7 | 32.7 | 0.10 |
| 10.0 | +20.3 | +19.0 | 175 | 116 | 30.0 | 0.00 |

The whole-box median file (`neutral_temp_median.txt`) gives the same picture: absorption until z ≈ 12.9, minimum about −3 mK at z ≈ 15. Where maps exist (z ≤ 10), the run logs show that the map mean is 4–17% *below* the global curve (e.g. 16.6 vs 18.0 mK at z = 8.0; 4.10 vs 4.92 at z = 6.15).

**Consequences for the text.**
- With cell averaging the fiducial run has a shallow absorption trough (about −3 mK at z ≈ 14–15) and crosses into emission at z ≈ 12.8–13. That is *after* z_α (x_α = 1 at z = 13.6), i.e. the canonical ordering z_α > z_h.
- Statements that need revision:
  - l. 496 "drives T_K above T_γ by z ≈ 17" — only true for the mean; 60% of neutral cells are still colder than the CMB at z = 15.
  - l. 499 "completely suppress the expected absorption feature"
  - l. 519 "brief, shallow absorption at z ≳ 18"
  - l. 522 "near-simultaneous or slightly reversed ordering (z_h ≳ z_α)"
  - l. 1229–1237, and Conclusions bullet 1 (l. 1279)

**Fix.** Compute the global quantities as volume averages of cell-level T_S and δT_b, at least for z > 10. Global J_α is fine there, since x_α is close to uniform. Keep the mean-T curve only as an illustration, and say so.

### 1.2 [A] The adiabatic reference is dominated by the same artifact

**What the code does.** `compute_neutral_temp_noxray.c` replaces T in cells with x_HII < THRESH_FORCE by T_ad = T0((1+z)/(1+z0))². Cells with THRESH_FORCE ≤ x_HII < 0.01 keep their simulated T. It then returns the *arithmetic mean* over x_HII < 0.01.

`threshold.py` shows that 95–99% of neutral cells are forced, yet:

| z | T_neutral,ad (mean) | T_ad | ratio | published δT_b,ad | δT_b with T = T_ad |
|---|---|---|---|---|---|
| 14.0 | 10.2 | 8.3 | 1.23 | −39.8 | −51.4 |
| 13.3 | 10.9 | 7.6 | 1.44 | **−44.7** | −70.4 |
| 12.0 | 14.7 | 6.2 | 2.35 | −32.8 | −106 |
| 11.0 | 20.4 | 5.3 | 3.8 | −15.0 | −126 |
| 10.0 | 29.1 | 4.5 | 6.5 | −0.8 | −138 |
| 9.0 | 44.1 | 3.7 | 11.9 | +8.7 | −143 |
| 7.0 | 145 | 2.4 | 61 | +10.9 | −105 |

The "trough at z ≈ 13" and the "rise into emission at z ≈ 10" (l. 496, 522, 1233) come from averaging 1–5% hot cells into the temperature. They do not reflect partial re-heating of the bulk. A volume average of δT_b for the *same* forced field would remain in deep absorption (of order −100 mK) to z ≈ 7. The f_heat = 0.3 interpolation (l. 522, 1221) inherits the problem.

**Further issues.**
- **Threshold sensitivity is not stated.** `LUMINA_ALL_ad{,_5e-4,_1e-4}.csv` give troughs of −44.8 / −40.9 / −37.0 mK at z = 13.2–13.5, with zero-crossings at z = 9.9–10.4.
- **The code no longer matches the paper.** It currently has `THRESH_FORCE 0.0005`; the 10⁻³ result exists only as a data file.
- **Anchor temperature.** The anchor is T0 = 34.2 K, the *mean over all cells* at z = 29.4 (median 29.7 K). This is about 2× the standard post-decoupling value (≈ 0.018(1+z)² K ≈ 17 K at z ≈ 29.4), which makes the reference trough shallower.
- **l. 493 is contradicted.** The simulated gas does not "cool adiabatically" at z ≳ 27: T_vol rises 34.2 → 35.0 → 39.9 K from z = 29.4 to 24.9 to 19.9.
- **The T_K,ad curve has different provenance.** It comes from `temp_ad.csv`, dated **2026-05-07**, months before the Sep 24 no-X-ray code and with an undocumented construction.

### 1.3 [B] Fig. 2(a): "T_K" is the all-gas volume mean

`visual.ipynb` cell 2 plots `df_global["T_vol"]` as T_K, while T_S^H uses `T_neutral`. At z = 9.2 the values are T_vol = 1739 K, T_neutral = 243 K, T_S = 174 K. The panel suggests T_S^H is decoupled by about 10×; the real ratio is 0.72.

Either plot T_neutral, or relabel the curve and explain the difference. The caption's "blue dashed curves show the adiabatic reference" is also wrong for panel (a), where the reference curves are red and green dashed.

### 1.4 [B] Coupling "saturation" statements

From `LUMINA_ALL.csv`:

| z | x_α | T_neutral | T_S | 1 − T_γ/T_S |
|---|---|---|---|---|
| 12.0 | 3.5 | 90 | 67 | 0.47 |
| 10.0 | 12.2 | 167 | 124 | 0.76 |
| 9.2 | 18.5 | 243 | 174 | 0.84 |
| 8.0 | 31.7 | 473 | 304 | 0.92 |
| 7.0 | 47.3 | 886 | 488 | 0.955 |

- l. 503 says (T_S ≫ T_γ) and "insensitive to further thermal evolution" by z ≲ 12. That holds only for z ≲ 8.
- l. 519 "Lyα coupling saturates" is too strong, and l. 1296 "saturated EoR regime (z ≲ 10)" is wrong: at z = 10 the thermal factor is 0.76. The emission peak at z ≈ 9.4 is shaped by the rising thermal factor.
- l. 1553 "Once x_α^H ≫ 1, T_S^H becomes locked to T_K" is false. T_S → T_K requires x_α ≫ T_K/T_γ (about 9–40 here). The weak metallicity sensitivity comes from T_S ≫ T_γ, not from T_S = T_K.

### 1.5 [B] Milestones quoted inconsistently

From `LUMINA_ALL.csv`, x_HI = 0.9 / 0.5 / 0.1 / 0.01 / 10⁻³ occur at z = 9.17 / 6.87 / 5.83 / 5.38 / 5.13.
- The end of reionization is given as z ≈ 5.2 (l. 211, 463), ≈ 5.5 (l. 463 later, l. 588), and "5.5–6" (l. 512, 519). At z = 6 the IGM is still 16% neutral and δT_b = 3.6 mK. Use about 5.2–5.4 consistently.
- δT_b peak: 20.7 mK at z = 9.44 (paper "≈20 mK around z ≈ 10" — acceptable).
- He: x_HeII peaks at 0.94 at z = 5.41; x_HeIII = x_HeII at z = 4.01 ✓; δT_b^He peaks at 0.0288 µK at z = 5.64 ✓.

### 1.6 [B] Helium physics statements contradicted by the ionization history

- **l. 516.** "Helium remains predominantly neutral at z ≳ 10, as the soft UV photons driving hydrogen reionization lack the energy to ionize helium." This is contradicted by l. 143 and by the data: x_HeII reaches 0.1 at z = 9.22 versus x_HII = 0.1 at 9.17, and at z = 7, x_HeII = 0.47 versus x_HII = 0.46. Stellar (BPASS) photons singly ionize He in step with H.
- **l. 639.** "massive stars … do so less efficiently … concentrating the signal around the densest star-forming halos" has the same problem. The granular z ≳ 7 morphology tracks the H II/He II bubbles plus localized coupling (as l. 650 itself says), not inefficient He I ionization.

### 1.7 [B] Sub-volume scatter

Measured from the 125 sub-box CSVs:
- Fractional δT_b scatter at the respective peaks: 0.7% (H, z = 9.4) vs 11% (He, z = 5.6). During the H decline it is larger for H (34% at z = 6).
- σ(x) at the midpoints: 0.052 (H) vs 0.089 (He).
- l. 528 and l. 1167 hold at the peaks. l. 516 ("comparable to the hydrogen fields") understates the He scatter.

---

## 2. Text–data and internal-logic problems (by section)

| Line(s) | Statement | Problem / evidence |
|---|---|---|
| 154, 636 | He signal "five orders fainter … direct consequence of the low abundance of ³He" | Abundance and atomic factors give 1.3 µK/27 mK ≈ 4.3 dex; weak coupling (1 − T_γ/T_S ≈ 0.02) supplies the remaining ≈ 1.6 dex, as l. 528 and l. 1136 say. The attribution is inconsistent. |
| 231 | "the helium field is computed in real space" | False for the maps: `PS_3cm_mpi.c` uses `SpectralDensity`. The ratio of z-axis to x-axis gradient variance is 1.39 (cell) and 1.50 (8-cell blocks) for `SpectralDensity`, versus 0.99 and 1.00 for `Density`. The product mixes spaces: x_HeII(real) × (1+δ)_s × sf(real). The H map has the same mixing (redshift-space HI × real-space spin factor); it is small at z ≲ 8 but worth stating. |
| 578 | skewness *and kurtosis* are "configuration-integrated counterparts of the bispectrum" | Kurtosis is the integral of the trispectrum; only skewness integrates the bispectrum. |
| 578, 598, sum_stat | "Kurtosis" | The plotted quantity is *excess* (Fisher) kurtosis (`scipy.stats.kurtosis`); BC uses κ+3 ✓. Say "excess kurtosis" in the caption and legend. |
| 619–622 | H: "small and stable at z ≳ 8"; "lower BC at z ≥ 8" | Skewness falls 2.08 → 1.54 and kurtosis 6.8 → 4.5 between z = 10 and 8. BC *decreases* from 0.54 (z = 10) to a minimum of 0.44 (z ≈ 7.5) before rising to 0.69 (z = 5.69 ✓). |
| abstract, 1283 | "skewness and kurtosis … increasing as reionization proceeds" | For H they decrease until z ≈ 7 (minimum ✓ l. 619) and rise only after. |
| 660 | He VID at z = 5: "dominant peak near ≈ 0.01 µK, which tracks the mean global 3.46 cm brightness temperature" | The bump in `vid_He.png` sits at about 0.006 µK, while the map mean is 0.0288 µK and the global value 0.0246 µK. It does not track the mean. |
| 670 | He σ "remains high and relatively stable from z = 6 to z ≈ 3.5" | σ falls monotonically 0.066 → 0.048 → 0.031 → 0.015 µK (z = 6, 5, 4, 3.5), ≈ 4.3× (`All_vid_stats_summary_He.csv`). The figure shows 100σ on a log axis, which hides the decline. Say "100σ" in the caption. |
| 673 | He skew 6.3 → 7.9, kurt 73 → 114, > 10³ by z ≈ 3.2 | ✓ (6.26 → 7.82, 73.1 → 115.4, 1386 at z = 3.21). |
| 676 | He BC "consistently elevated (0.51–0.55)"; "maximizing the statistical contrast" at z ≲ 3.5 | BC is flat at 0.525–0.532 and *below* the 5/9 bimodality threshold drawn in the same figure, so by the paper's own criterion it is not bimodal. There is no maximum at z ≲ 3.5; it drops to 0.497 (z = 3.21) and 0.368 (z = 3.0) ✓. |
| 716 vs 1172 | LUMINA "under-resolves … the dense absorbers" vs "adaptive resolution … enables the tracking of dense absorbers" | Contradictory. The LUMINA baryon mass (3.6×10⁶ M☉) is about 6× coarser than THESAN-1. |
| 719, 734 | "At z ≳ 9 … dominated by matter density"; dip at z ≈ 8.5 = density → ionization transition | In `PS_H_z.png`, Δ²(k = 0.5) *increases* toward high z (10.3 mK² at z ≈ 8.7 → 18.7 at z = 10). Density power would decrease. The map logs show sf ∈ [−0.76, 1] (mean 0.70) at z = 10. So z ≳ 9 is spin-temperature dominated, consistent with l. 882 (bispectrum); the dip marks the T_S → ionization hand-over. |
| 746 | THESAN-1 "ionization history proceeds more rapidly"; "at z ≲ 7 both converge" | "Earlier" rather than "more rapidly" (THESAN-1's PS reaches about 0 at z ≈ 5.5 vs 5.2 for LUMINA). At z = 6–6.5, THESAN-1 is 25–50% below LUMINA, so the curves do not converge. |
| 749 | LUMINA above LOFAR 1000 h at k = 0.1 for 5.8 ≲ z ≲ 7.3 | ✓ (figure: about 5.75–7.3). |
| 702, 749 | SKA-Low/LOFAR "2σ thermal noise" (Barry2022; van Haarlem 2013) | The curves come from `PS_21cm/obs/simu_noise_ps_*.txt`, which look like ps_eor-type simulations with `filter_wedge=0.0` (no foreground avoidance) and dk/k = 0.4. The notebook plots column 6 (Δ²_err) and labels it 2σ without a factor of 2. Check whether that column is 1σ. Cite the tool, state the assumptions (BW = 10 MHz, no wedge cut), and note that Barry2022 / van Haarlem 2013 do not provide these curves. |
| 764 | 30× above Spina+2025 attributed "primarily" to the resolved He II Lyα background | In the global pipeline x_c^He/(x_c + x_α) ≈ 0.70 at the peak. The collision-only run (`LUMINA_ALL_Coll_only.csv`) lowers the peak only from 0.0288 to 0.0202 µK. Removing all Lyα changes the power by about 2×, not 30×. The quoted Basu "collisional-only" agreement also points to collisions. Give a test (e.g. map PS with J_α^He = 0) or soften. |
| 1133 | "current constraints remain 1–2 orders of magnitude above predictions" | Fig. `ps_hydro`(b): 2σ limits are ≳ 10³ mK² (LOFAR ≈ 3×10³ at k = 0.075; HERA ≈ 2–3×10³ at k ≈ 0.5) against predictions ≲ 15 mK², i.e. ≈ 2–3.5 dex. l. 728 ("well above") is fine. |
| 1139 | 21 cm / 3.46 cm cross-correlation | In LUMINA the 21 cm signal is about 0 by z ≈ 5.3, while He peaks at 5.6. The overlap window (z ≈ 5.3–7) is narrow; worth stating. |
| 1142 | "additive Gaussian noise leaves raw cumulants untouched" | Only the third and higher cumulants are unchanged; the second (variance) increases. |
| 989, 1023, 1184, 1285 | He signal confined to / generated near luminous quasars | In the global pipeline stars give 99% of J_α^He at z ≥ 5.5 and 89% at z = 4; collisions dominate the coupling. Only the map pipeline is AGN-dominated at z ≲ 4.5, and that comes from the AGN-SED interpolation bug (§3.1). This also contradicts App. B (l. 1559). |
| 1559 | "at z ≈ 5 [stellar] He II Lyman-band emissivity exceeds that of AGN by a factor of a few" | In the adopted inputs (f_esc,★ = 1) stars/AGN = 23× in emissivity and 45× in J at z = 5, 4× / 8× at z = 4, and 3× / 5× at z = 3.5. "A few" applies to *escaped* emissivities (Zier+26) or z ≲ 4. Clarify which. |
| 960, 969, 989, 1016–1020 | x_HeII forms "overdense shells around He III regions, as He I is singly ionized in outer radiation fields" | At z ≤ 5, x_HeI ≈ 0 (x_HeII + x_HeIII ≈ 1). Outside He III bubbles the medium is uniformly He II, ionized by stars long before (§1.6). x_HeII ≈ 1 − x_HeIII, so B(x_HeII) ≈ −B(x_HeIII). The shell picture contradicts the paper's own Sec. 3.1 and Fig. 2(b). |
| 923 | "Consequently, this positive rise reflects …" | Non-sequitur; the sentence that explained why the folded limit is not squeezed was removed. |
| 938, 1020 | non-isosceles trough "shifted toward intermediate configurations (vertical dashed line)" vs "trough near cos θ ≈ −0.5 … broken symmetry shifts this minimum" | Self-contradictory. For k₂ = 2k₁, cos θ = −0.5 has no special geometric meaning (k₃ = 1.73 k₁: neither equilateral nor isosceles). |
| 1023 | signal restricted to peaks is "positive by construction" | Overstated; positive skewness does not imply a positive bispectrum in every configuration. |
| 1115 | He sign-flip scales "lie closer to the MFP mean than the median" | True for x_HeII = 0.8, 0.7, 0.6, but at x_HeII = 0.9 the diamond (≈ 2.3 cMpc) is *below* the median (≈ 3.3 cMpc). |
| 1118 | He III bubbles "systematically larger" at comparable fractions | They are larger at x ≥ 0.2, but at x = 0.1 the He median (≈ 50 cMpc) is below the H median (≈ 62 cMpc). "Until the late stages" (abstract) is the right wording. |
| 1369 | protons "lack the spin-exchange channel" | The physical reason is Coulomb repulsion between p and ³He⁺. |
| 1557 | "represents a an upper limit" | Typo. |
| 1083, 1111 | Bubble estimators use x_HI / x_HeII as the progress variable | The other He figures use x_HeIII. Harmonize (x_HeII ≈ 1 − x_HeIII only because x_HeI ≈ 0). |

---

## 3. Code and pipeline issues

### 3.1 [A] AGN J_α^He: linear interpolation across a 1-dex SED gap

- Both SED tables (`agn_sed_full.hdf5` used by C and `AGN_SED.dat` used by Python) have no nodes between **20.67 eV and 248 eV**.
- `compute_Jalpha_AGN.c::eps_b()` interpolates Ñ(ν) = L_ν/(L_bol hν) *linearly in ν* across the gap. The global notebook interpolates log-log, i.e. a power law.
  - At 40.8 eV: νL_ν/L_bol = 0.172 (C) vs 0.038 (Python).
  - At 54.4 eV: 0.215 vs 0.026.
- Result: per-file `J_mean` / global J_α,AGN^He = **11.2–11.7 for every AGN He file** (all generations, Jun 18–Oct 1). For H the ratio is 0.98–0.99 because the H Lyman band lies inside the dense part of the table. Stellar J files agree with the global pipeline to ≤ 1.5% (H and He).
- Impact on the He maps: the AGN share of J_α^He becomes about 20% / 58% / 71% at z = 5 / 4 / 3.5, versus 2% / 11% / 18% in the global pipeline. Map ⟨x_α^He⟩ is about 2× global at z ≈ 4 and 3× at z ≈ 3.5. I estimate the He δT_b would drop by about 8% at z = 5, 30% at z = 4 and 50% at z = 3.5 with the log-log SED. The spatial pattern would also change: there would be less quasar-centred coupling, which affects the VID tails, PS and bispectrum.
- Fix: use the analytic Shen+2020 EUV power law (or at least log-log interpolation). Regenerate `Jalpha_AGN_He_*` and the He maps.

### 3.2 [A] He map and global He pipelines differ for structural reasons too

From the run logs (`helium/PS_3cm_He_*.out`), comparing ⟨δT_b^He⟩_map with the global value:

| z | ratio |
|---|---|
| 10.0 | 10.6 |
| 8.6 | 5.1 |
| 6.6 | 1.84 |
| 6.15 | 1.49 |
| 5.6 | 1.23 |
| 5.0 | 1.17 |
| 4.05 | 1.33 |
| 3.52 | 1.66 |
| 3.03 | 3.3 |

⟨sf⟩ agrees between the two (0.023), so the difference comes from correlations: x_c^He ∝ n_e ∝ (1+δ), multiplied by (1+δ) x_HeII. At z ≲ 4.5 the AGN SED bug adds to this. The global He curve (Fig. 2d and the "0.03 µK" quoted in the abstract-level text) should be the volume average of the maps.

### 3.3 [A] Bispectrum inputs

The scripts that produced the published files are the ones stored next to the outputs in `PS_21cm/bispec*/final_Tb/`, not the older copies one directory up.

| Figure / panel | Field file | BOX_LEN | Status |
|---|---|---|---|
| bispec_H_iso, non_iso_H — x_HI (files named `f_HII` but computed as 1 − x_HII ✓) | renders | 338.3 h⁻¹ | ✓ |
| bispec_H_iso, non_iso_H — δT_b | `cell_by_cell/hydrogen/Tb_H/RIN_*` | 338.3 | ✓ |
| bispec_He_iso — x_HeII | renders × 12.67 | 338.3 | ✓ (identical k₃ lattice to the T_b files) |
| bispec_He_iso — δT_b^He | `/…/compute_J_alpha_He/Tb_He/RIN_*` (now `corrupted_files_do_not_use/…`, Sep 16–21) | 338.3 | ✗ old maps; PS ratio old/new = 0.82–1.12 |
| non_iso_He — x_HeII | renders × 12.67 | **500.0** | ✗ k in cMpc⁻¹ (k₁ = 0.44, k₂ = 0.89 h cMpc⁻¹); verified by differing k₃ lattices |
| non_iso_He — δT_b^He | old maps (as above) | 338.3 | ✗ old maps |

The normalized bispectrum is amplitude-invariant, but the old maps also predate other pipeline changes. Recompute the He T_b bispectra from `cell_by_cell/helium/Tb_He`, and recompute non_iso_He x_HeII with 338.3.

### 3.4 [B] Lightcone (`fan_diagram.ipynb` cells 15–17)

1. `generate_combined_dataset` shifts all outer-lightcone redshifts by `z_gap` = 5.0015 − 4.7483 = **0.253**. This shift enters the radial labels *and* the physics (T_CMB, (1+z)³ densities, J_α look-up). The outer file contains data down to z = 4.753, so `outer_zmin = 4.75` without the shift would remove the need for it.
2. J_α = J_α,global(z) × (ρ/ρ̄), read from `/orcd/…/LUMINA_all.csv` (now in `corrupted_files_do_not_use`), with no S_α. This is not the cell-by-cell method described in paper Sec. 2.4 and App. A.
3. `xHeIII_all = 1 − xHeII_all`, so neutral He is treated as He III. That gives n_e ⊃ 2 × 0.079 n_H ≈ 0.16 n_H in neutral gas (versus ≲ 10⁻³). At z ≳ 12 the κ_eH n_e term then exceeds the H–H term by about 6×, exaggerating collisional coupling and the high-z absorption shown in the δT_b^H sector.
4. A 3-cell (2.3 cMpc) slab average is used, while the slices in the maps are one cell; this is minor.

l. 460 ("All fields are extracted with on-the-fly light-cone geometry") is true for the inputs, but T_b should either be computed as in the paper or the simplification should be stated.

### 3.5 [C] Smaller code observations

- **Cosmology mismatch.** The render headers give LUMINA = Planck 2018 (h = 0.6766, Ω_m = 0.30964, Ω_b = 0.048975, L = 338.3 h⁻¹ cMpc = 500.0 cMpc), consistent with l. 198. `PS_21cm_mpi.c`, `PS_3cm_mpi.c` and the global δT_b prefactor hard-code Planck 2015 (h = 0.6774, Ω_m = 0.3089, Ω_b = 0.0486). In the C codes this also enters `LENGTH/h` and `MASS/h`. The effect is ≈ 0.5% but should be unified; the notebook comment "kept separate on purpose" is not justified, since Eq. 1 uses the model's own Ω's. `compute_Jalpha.c` labels Planck 2018 values as "Planck 2015".
- **Gunn–Peterson coefficients.** τ_GP coefficients 3×10⁵ and 6×10³ are about 23% below the values for this cosmology (3.9×10⁵ and 7.7×10³ at z = 6), shifting the S_α exponent by about 9%. The He map code calls `compute_S_alpha(z, T, f_HI)` — the H optical depth — rather than x_HeII and τ_GP^He, contrary to l. 1426. Both effects are numerically negligible in hot gas but inconsistent.
- **Stellar metallicity.** `Z_mean_mw` is the mean over *all* star particles. The emissivity is for newly formed stars, whose metallicity is higher (e.g. `Z_new` = 0.0135 vs 0.0098 at z = 4.5). The He-band BPASS yield drops about 30% between these values, so the stellar J_α^He is biased high by up to about 30% at z < 5. Consider SFR-weighted Z (and see App. B).
- **f_rec table.** The AGN J_α code used for the maps has an f_rec table that differs from Pritchard & Furlanetto (2006) by ≤ 2% (e.g. n = 30: 0.3523 vs 0.3590). This is fixed in the ren_1280 copy and is negligible.
- **Log diagnostics.** The H T_b log prints `rin` min/max without dividing by V_cell; the He log prints the mean multiplied by V_cell (= 1.477×10⁸) and labels µK values as "mK". Only the printouts are affected; the maps and PS are not.
- **Code comments vs code.** `compute_neutral_temp.c` documents x_HII < 0.01 but now uses < 0.001. The file actually used (`neutral_temp_means.txt`) differs from the 0.001 output, so it came from an earlier 0.01 build. `compute_neutral_temp_noxray.c` now says 5×10⁻⁴ while the paper uses 10⁻³ (§1.2). These are reproducibility risks.
- **RT switch at z = 4.75.** Δ²_He(k = 0.5) in `ps_helium_z.png` has a kink at z = 4.75, where the RT switches to a single bin plus a uniform UVB. Acknowledge it in paper Sec. 3.3.

---

## 4. Provenance and reproducibility

The user-labelled `corrupted_files_do_not_use/` currently holds inputs to published figures:

- **AGN J_α files.** `compute_J_alpha_H/J_alpha_AGN/Jalpha_AGN_{H,He}_*.hdf5` fed *every* T_b map (both run scripts point to `/orcd/…/compute_J_alpha_H/J_alpha_AGN`). They were produced by at least five generations of the AGN code (Jun 18, Jul 6–7, Aug 12–13, Sep 4, Sep 30) and only the last source is archived. I checked that all generations agree in normalization with each other (the ratio to the global value is stable per species), but each carries the §3.1 He bug.
- **Old He maps.** `compute_J_alpha_He/Tb_He/RIN_*` feed the He T_b bispectra (§3.3).
- **Old global CSV.** `LUMINA_all.csv` feeds the lightcone J_α (§3.4).

If those files really are corrupted, every T_b-derived figure (lightcone, maps, VID, PS, bispectra, sub-box VID) needs regenerating. If they are not, move them back or document them, so the paper's provenance does not run through a "do not use" directory.

Other provenance notes:
- `temp_ad.csv` (T_K,ad in Fig. 2a) dates from 2026-05-07 and cannot be tied to the documented adiabatic construction.
- 56 He maps at z < 4.8 were regenerated on Oct 1 (after a Sep 30 AGN-J regeneration). The map means are smooth across the two batches. The He VID statistics file (`All_vid_stats_summary_He.csv`, Sep 30 01:29) simply does not include those 56 snapshots; its values match the final maps.
- The He VID stats cell in `one-stat.ipynb` (cell 12) reads `vid_stats_summary_He.csv`, dated **2026-04-17**, a stale model. The published panel matches the Sep 30 file instead. Point the notebook at the right file.
- `21cm_analysis/` cannot reproduce the figures as-is. Most notebooks read paths that have moved (`compute_J_alpha_*`, `PS_21cm/TS_TK/H/Tb`), and several hold multiple stale versions of the same figure cell.

### Figure → producer → status

| Fig. | Producer | Inputs | Status |
|---|---|---|---|
| lightcone | fan_diagram cells 15–17 | rlc_640 + old `LUMINA_all.csv` | ✗ §3.4 |
| global | visual cell 2 | LUMINA_ALL, LUMINA_ALL_ad (10⁻³), temp_ad (May), sub-boxes | ✗ §1.1–1.3 |
| tmap/f (H, He) | visual_maps cells 6, 12 | `cell_by_cell/*/Tb_*` (1-cell slice) ✓ | ✓ (He inherits §3.1, §2 l. 231) |
| vid / sum_stat | one-stat | `vid_stats_summary_156_406.csv` (Sep 29 ✓), `All_vid_stats_summary_He.csv` (Sep 30 ✓) | ✓ data; text issues §2 |
| ps_hydro, PS_H_z | PS_redshift / visual_PS | `cell_by_cell/hydrogen/Tb_H/PS_*.out`, `f_HII`, Thesan-1, `PS_21cm/obs` | ✓ normalization; noise-curve provenance (§2) |
| ps_helium(_z) | same | `cell_by_cell/helium/Tb_He`, `f_HeIII/new` | inherits §3.1, redshift space |
| bispec H | final_Tb scripts | current maps | ✓ |
| bispec He (iso, non-iso) | final_Tb scripts | old He maps; non-iso x_HeII BOX_LEN = 500 | ✗ §3.3 |
| BSD | BSD.ipynb | mfp outputs (x > 0.5, 192 dirs, He × 12.67) ✓, PS_bubble_size_*.dat ✓ | ✓ |
| subbox_VID | subvol_VID cells 3, 6 | current maps ✓ | ✓ (He inherits §3.1) |
| metallicity | visual cell 5 | LUMINA_ALL_Z_*, only_AGN, Coll_only (global pipeline) | ✓; but these are global-pipeline numbers (see §3.2) |

---

## 5. Citations and bibliography

- `vanderHolst2011` (l. 173) is *CRASH: A Block-adaptive-mesh Code for Radiative Shock Hydrodynamics* (high-energy-density physics). The cosmological RT code CRASH is Ciardi et al. 2001 and Maselli, Ferrara & Ciardi 2003 (+2009; Graziani et al. 2013).
- `Pritchard2007` (l. 1446, f_rec) is the X-ray heating fluctuations paper. The recycling fractions are from Pritchard & Furlanetto 2006 (MNRAS 367, 1057), which is also the source the code comments cite.
- `Barry2022` / `vanHaarlem2013` for the SKA-Low and LOFAR sensitivity curves: see §2 (l. 702/749).
- Every `\cite` key resolves in the .bib, and every `\ref` has a label. `eq:tophat` and `sec:intro` are defined but never referenced, which is harmless. I could not compile the PDF (no TeX on the node).

---

## 6. Checked and correct

Physics and constants:
- Box 500 cMpc (header), 2×6000³, m_DM = 1.9×10⁷ and m_b = 3.6×10⁶ M☉, volume ratios 125× and 143×.
- ΔE = 5.87 µeV, T★ = 0.068 K; T★^He = 0.416 K; A_He/A_H = 687 ("690").
- **Eq. 2 prefactor:** re-derived as 1.29 µK including the inverted ³He⁺ hyperfine level ordering (g_u/g_l = 1/3); paper 1.3 ✓, code 1.67((1+z)/10)^½ = 1.29((1+z)/6)^½ ✓.
- x_α prefactors 16π²T★e²f/(27 A m_e c T_γ) for H and the 4/9 He analogue; x_c^He with σ̄ = 14.3 eV/kT a₀².

J_α machinery:
- f_rec table (P&F06), z_max(n), ν'_n, ν_LL (H and He II), and the J_α formula and its discretization (200 log shells from one cell, SFR interpolated in z, Z at the emission redshift, log-Z interpolation).
- The "1.5×" sphere-versus-shell weighting (l. 1474): it follows analytically for a constant radial weight.

Stellar inputs:
- BPASS normalization (≈ 1.0–2.2×10⁴ Lyman-band photons per baryon; 1 Myr–2 Gyr age bins, first bin from t = 0).
- Z(z) values (4.9×10⁻³ at z = 6; 0.016 at z = 3).
- Stellar J_α map means match the global pipeline (≤ 1.5%), and H-AGN J matches to 1%.

Power spectra and bubbles:
- PS normalization (Δ² × (M_cell)²), k units (h cMpc⁻¹), THESAN-1 k_min.
- R = 2.5/k_peak (the one-bubble x³W² term peaks at kR = 2.46), with h → cMpc conversion in the bubble figure.
- MFP set-up (x > 0.5, 192 directions, 2 box crossings, He per He nucleus); "≈ 40% of rays in one cell at z ≈ 9.2" (41.8%).

Quoted numbers that match the data:
- x_HII = 0.1 / 0.5 / 0.9 at z = 9.17 / 6.87 / 5.83; x_HeIII = 0.1 / 0.5 / 0.9 at 5.00 / 4.01 / 3.44.
- T_vol = T_CMB at z = 16.9; adiabatic trough −44.8 mK at z = 13.2; f_heat = 0.3 trough −28.8 mK.
- H VID: σ 13.3 → 1.3 mK; skewness/kurtosis minima near z ≈ 7; BC = 0.689 at z = 5.69. He skewness/kurtosis values as quoted.
- PS(z): k = 0.5 peak at z ≈ 7.0; k = 0.1 peak at z ≈ 6.4; dip near 8.5; THESAN-1 peak at z ≈ 7.8; amplitudes 14–15 mK²; LOFAR crossing 5.8–7.3; Δ²_He(0.1, z ≈ 4) ≈ 3×10⁻⁵ µK²; He bump at z ≈ 3.9.
- Bubble numbers in paper Sec. 3.5: medians 1 → 65 cMpc; PS saturation ≈ 13 cMpc; He PS 12 → 28 cMpc.
- Metallicity sensitivity: H peak ≈ 6%; He ×3.4 from Z = 10⁻⁵ to 0.04.

Bispectrum:
- Normalization as in Watkinson (bifft `normalise=True`); H and He-iso k units; sign-flip radii (≈ 0.8 h⁻¹ and 1.4 h⁻¹ cMpc).

## 7. Not checked

- Simulation-level inputs: f_esc = 0.18, X-ray normalizations, obscuration, ε_f,high (these sit in Zier+26).
- The THESAN-1 T_b maps in `cell_by_cell/Thesan-1` (I did not check that they use the same spin-temperature model).
- The x_HII / x_HeIII PS generation code, the MFP C++ internals, bifft internals.
- The values in the HERA/MWA/LOFAR data files.
- The literature numbers quoted for Spina+25, Basu+26, Eide+20 and Iliev+14.

## 8. Suggested order of work

1. Fix the AGN-SED interpolation (§3.1). Regenerate `Jalpha_AGN_He`, the He maps, He PS/VID/bispectra (from the *current* maps, with `BOX_LEN = 338.3` everywhere), the sub-box VID and `ps_helium*`.
2. Recompute the global H and He signals and the adiabatic reference as volume averages of cell-level δT_b (§1.1, §1.2, §3.2). Then rewrite paper Sec. 3.1, 4.4–4.5 and Conclusion 1, especially the z_h vs z_α ordering and the "no trough" language. State the adiabatic threshold dependence and the anchor temperature.
3. Rebuild Fig. 1 without the redshift shift, with the paper's J_α method and correct n_e (§3.4). Plot T_neutral in Fig. 2(a) (§1.3).
4. Decide between real and redshift space for He (§2, l. 231) and make the text and code agree.
5. Revise the text items in §1.4–1.6 and §2. Fix the citations (§5) and the noise-curve description.
6. Move all published inputs out of `corrupted_files_do_not_use/`, or regenerate them, and pin each figure to a single notebook cell and data version.
