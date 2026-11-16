# 21 cm spin temperature: Wouthuysen–Field coupling and the 21 cm optical depth

**Lumina 21 cm pipeline, change of 2026-11-15** (`SPIN_MODEL = 1`). Changhyun Cho, with Claude Code.

This file is self-contained: what changed in the 21 cm brightness-temperature calculation, why, the evidence, the measured effect on Lumina, how it is implemented and tested, and what is still approximate.

---

## 1. Summary

The production 21 cm maps (`PS_21cm_mpi.c`) computed the spin temperature with the simplest standard treatment:
- the Wouthuysen–Field (WF) suppression factor S_α from a one-line fit;
- a colour temperature equal to the gas temperature, T_c = T_K;
- a Gunn–Peterson depth from a rounded, mean-density formula;
- a brightness temperature linear in the 21 cm optical depth.

From 2026-11-15 the code uses (`SPIN_MODEL = 1`):

| # | Change | Source | Effect on Lumina |
|---|---|---|---|
| 1 | τ_GP from first principles, 1.30× the rounded 3×10⁵ form | Hirata 2006 Eq. 35 | ≤ 1.8% (mean), ≤ 1.2% (Δ²) |
| 2 | τ_GP with the pixel's own n_HI, i.e. x_HI(1+δ) | Hirata 2006 §3.2 | ≤ 1.4% (Δ²), opposite in sign to 1 |
| 3 | Hirata's S̃_α(T_K, T_S, τ_GP) and colour temperature T_c ≠ T_K | Hirata 2006 Eqs. 38–42; Chuzhoy & Shapiro 2006 | **up to −9% in Δ², +14% (+0.19 mK) in the mean, at z = 12.7–16** |
| 4 | Finite 21 cm optical depth: x_CMB = (1 − e^{−τ₂₁})/τ₂₁ in T_S, and δT_b ∝ 1 − e^{−τ₂₁} instead of τ₂₁ | Venumadhav et al. 2018 Eqs. 13–15 | ⟨x_CMB⟩ ≥ 0.991 (pixels ≥ 0.987): ≲ 0.3% on δT_b (measured in production, §4) |

Effects 1–3 were measured on the full 640³ Lumina box at 15 snapshots (Section 4), and all four together by the production run (job 29237390): the global signal changes by +17% (+0.16 mK) at z = 15, +7% at z = 17, +2.5% at z = 13, and ≤ 0.5% at z ≤ 12. All four act only while the Lyα coupling is building up: in Lumina, x̄_α goes from 0.05 (z = 16.3) to 1.3 (z = 12.7). At z ≲ 11 every change is below 0.6%.

---

## 2. Equations, before and after

T_γ = 2.7255 (1+z) K. x_c is the collisional coupling (Furlanetto, Oh & Briggs 2006 rate tables; unchanged).

**Before (`SPIN_MODEL = 0`, the maps of 2026-10-15):**

- T_S = (1 + x_c + x_α) / (T_γ⁻¹ + (x_c + x_α) T_K⁻¹)
- x_α = [16π² T⋆ e² f_α / (27 A₁₀ T_γ m_e c)] S_α J_α
- S_α = exp[−0.803 T_K^{−2/3} (10⁻⁶ τ_GP)^{1/3}]
- τ_GP = 3×10⁵ x_HI ((1+z)/7)^{3/2}
- δT_b = P · x_HI(1+δ) · (1 − T_γ/T_S), with P = τ₂₁T_S/(1+z) at ⟨n_H⟩

**After (`SPIN_MODEL = 1`):**

- T_S⁻¹ = (x_CMB T_γ⁻¹ + x̃_α T_c⁻¹ + x_c T_K⁻¹) / (x_CMB + x̃_α + x_c)  (Venumadhav+18 Eq. 13)
- x̃_α = [8π λ_α² γ T⋆ / (9 A₁₀ T_γ)] S̃_α J_α  (Hirata Eq. 38). The prefactor is the same number as before: 1.7928×10¹¹/(1+z) cgs, equal to 21cmFAST's to 6×10⁻⁶.
- S̃_α = (1 − 0.0631789 T_K⁻¹ + 0.115995 T_K⁻² − 0.401403 T_S⁻¹T_K⁻¹ + 0.336463 T_S⁻¹T_K⁻²) / (1 + 2.98394ξ + 1.53583ξ² + 3.85289ξ³), with ξ = (10⁻⁷ τ_GP)^{1/3} T_K^{−2/3}  (Hirata Eqs. 40–41)
- T_c⁻¹ = T_K⁻¹ + 0.405535 K · T_K⁻¹ (T_S⁻¹ − T_K⁻¹)  (Hirata Eq. 42)
- τ_GP = (π e² f_α λ_α / m_e c) n_HI / H(z) = 1.3428×10⁻⁷ cm³ s⁻¹ · n_HI / H(z), with n_HI the pixel's own proper H I density  (Hirata Eq. 35)
- τ₂₁ = 3 c³ h A₁₀ n_HI / (32π k ν₂₁² H(z) T_S), and x_CMB = (1 − e^{−τ₂₁})/τ₂₁  (Venumadhav+18 Eqs. 2, 14)
- δT_b = x_CMB τ₂₁ (T_S − T_γ)/(1+z) = P · x_HI(1+δ) · (1 − T_γ/T_S) · x_CMB  (Venumadhav+18 Eq. 15)

T_S is implicit, because S̃_α, T_c and x_CMB depend on it. It is solved by fixed-point iteration from T_S = T_γ, as Hirata (§3.3) and 21cmFAST do, to a relative change below 10⁻¹⁰.

---

## 3. The four changes: physics and evidence

All quotes are verbatim from the arXiv versions of the papers, read on 2026-11-15.

### 3.1 Gunn–Peterson coefficient

The 3×10⁵ was a rounded value for another cosmology. Furlanetto, Oh & Briggs (2006, Phys. Rep. 433, 181), Eq. 43:

> "in our cosmology the mean Lyα optical depth experienced by a photon that redshifts across the entire resonance is τ_GP = χ_α n_HI(z) c / (H(z) ν_α) ≈ 3 × 10⁵ x̄_HI ((1+z)/7)^{3/2}."

Hirata (2006), Eq. 35, gives it in closed form: "τ_GP = 3 n_H x_HI λ³_Lyα γ / (2H)". This evaluates to 1.3434×10⁻⁷ n_HI/H. That equals π e² f λ/(m_e c) = 1.3428×10⁻⁷ cm³ s⁻¹ and 21cmFAST's `taugp()` constant 1.342881×10⁻⁷. For Lumina's cosmology the exact depth is 1.296–1.299× the 3×10⁵ form at every snapshot, e.g. 1.2735×10⁶ against 9.804×10⁵ at z = 14.42.

### 3.2 Local density in τ_GP

Hirata (2006), §3.2:

> "The value of S_α thus depends only on T_k, T_s, the injection profile ψ(ν), H, and n_H x_HI."

So S_α belongs to the local n_H x_HI. 21cmFAST's `taugp(z, delta, xe)` uses (1+δ)(1−x_e).

The local velocity gradient (τ_GP ∝ n_HI/(dv/dr)) is not included. Chuzhoy & Shapiro (2006) list "the effect of local departure from pure Hubble flow" among the processes they add. In linear theory, overdense gas expands more slowly, with the direction-averaged dv/dr ≈ H(1 − fδ/3). That makes τ_GP steeper than (1+δ), so the change made here goes in the right direction and slightly underestimates the density dependence.

### 3.3 Hirata's S̃_α and colour temperature

**The old fit is an approximation by its own source's statement.** Furlanetto, Oh & Briggs (2006), Eq. 55 and text:

> "when spin exchange can be neglected, the suppression is S_α ≈ … ∼ exp[−0.803 T_K^{−2/3} (10⁻⁶/γ)^{1/3}] … the Lorentzian wing approximation is least accurate at large temperatures; fortunately in that regime S_α ≈ 1 anyway, so equation (55) is accurate to a few percent at all T_K ≳ 1 K. At smaller temperatures, a numerical solution including the Voigt profile, the full integral over φ_α, and spin exchange, is necessary."

They also state, on T_c = T_K: "Field […] showed that atomic recoil during scattering, which tilts the spectrum to the red, is primarily responsible for establishing this equilibrium. However, as we will see, spin exchange slightly modifies this expectation."

**Hirata includes the missing physics.** Hirata (2006, MNRAS 367, 259), abstract:

> "(2) The fine structure of the Lyα resonance alters the photon frequency distribution and leads to a suppression of the scattering rate. (3) The spin-flip scatterings change the frequency of the photon and cause the photon spectrum to relax not to the kinetic temperature of the gas but to a temperature between the kinetic and spin temperatures, effectively reducing the strength of the Wouthuysen-Field coupling. (4) Near line centre, a photon can change its frequency by several times the line width in a single scattering event, thus potentially invalidating the usual calculation of the Lyα spectral distortion based on the diffusion approximation. It is shown that (1) suppresses the Wouthuysen-Field coupling strength by a factor of up to ∼ 2, while (2) and (3) are important only at low kinetic temperatures. Effect (4) has a ⩽ 3 per cent effect for kinetic temperatures T_k ⩾ 2 K."

**Accuracy of the fits.** Hirata (2006), §3.3:

> "The following formula for S̃_α reproduces our numerical results to within 1 per cent in the range T_k ⩾ 2 K, T_s ⩾ 2 K, and 10⁵ ⩽ τ_GP ⩽ 10⁷ for continuum photons" (Eq. 40). For T_c,eff (Eq. 42): "This reproduces the T_c,eff⁻¹ values from the Fokker-Planck equation to 1 per cent." "For the injected photons, it is found that Eq. (40) reproduces the Fokker-Planck results for S̃_α to better than 3 per cent."

**An independent analytic derivation agrees.** Chuzhoy & Shapiro (2006, ApJ 651, 1; astro-ph/0512206) obtain the colour temperature

> "T_α = (1 + 2.5 T_k) T_s / (1 + 2.5 T_s)"

and write: "We find an excellent agreement between our analytic solutions and his numerical results." To first order their coefficient (0.4) is Hirata's (0.4055 K). Numerically, T_c agrees to 0.15% at (T_K, T_S) = (10, 30) K and to 0.7% at (5, 50) K.

**When it matters.** Hirata (2006), Conclusions:

> "The corrections described here pertain to the strength of the Wouthuysen-Field effect and are important only during the era when x_c < x_α ⩽ O(1)." Once "x_α ≫ 1 and T_s ≈ T_k … changes in the coupling strength have no impact on the observable temperature fluctuations."

That is what the Lumina measurement shows (Section 4).

**Lumina is inside the fits' range:**
- no H I at T_K < 2 K at any of the 15 tested snapshots (`check_wf_variants.py` diagnostic: 0.0000);
- τ_GP(x_HI = 1) = 4×10⁵ to 3.5×10⁶.

Pixels with low x_HI have τ_GP < 10⁵. There S̃_α → its numerator ≈ 1 and the coupling hardly matters. The production log reports the H I share with T_K or T_S < 2 K.

**Hirata's largest effect (1) was already in the pipeline.** The cascade probabilities P_np of his Table 1 are, at all 29 entries n = 2–30, the `F_REC_TABLE` of `compute_Jalpha.c` (the f_rec of Pritchard & Furlanetto 2006).

**Approximation kept:** one S̃_α is used for continuum and injected photons together (as 21cmFAST does). Hirata's fit is 1% for continuum and 3% for injected photons.

### 3.4 Finite 21 cm optical depth (x_CMB)

Venumadhav, Dai, Kaurov & Zaldarriaga (2018, PRD 98, 103513), §III: with the 21 cm line's own spectral distortion,

> "We get a modified version of the usual equation for the spin temperature: T_s⁻¹ = (x_CMB T_γ⁻¹ + x̃_α T_c⁻¹ + x_c T_k⁻¹)/(x_CMB + x̃_α + x_c), where x_CMB = (1/τ₂₁)(1 − e^{−τ₂₁})" (Eqs. 13–14); "The brightness temperature ΔT_b against the CMB is defined from the phase-space density on the red side of the line in Eq. (11): ΔT_b = x_CMB (τ₂₁/(1+z)) (T_s − T_γ)" (Eq. 15). "We numerically compute both x̃_α and the effective color temperature … in the same manner as H06. We have checked that our answers match the fitting formulae of H06 in the relevant range of parameters."

This does two things:
- In T_S it weakens the CMB's pull by x_CMB < 1, a new effect.
- In δT_b, x_CMB τ₂₁ = 1 − e^{−τ₂₁} is the exact optical-depth factor. It replaces the linear τ₂₁, which was recorded as ~4% too large in |δT_b| at mean density at z ≈ 20.

Here τ₂₁ uses the pixel's own n_HI and the Hubble rate (real space, no velocity gradient), consistent with the real-space signal.

**Not included: CMB and Lyα heating.** The headline result of Venumadhav et al. is heating: "The CMB heating alone makes a ≃ 8.6% (≃ 15%) correction to both T_k and ΔT_b at z = 17 (15)" in a model with no X-rays, and Lyα heating 1.3% (5%). These change the gas temperature, which is Lumina's own thermal history and cannot be corrected consistently in post-processing. Lumina's neutral IGM is already at 10²–10³ K by z = 10, so this is probably smaller here than in their X-ray-free model.

---

## 4. Measured effect on Lumina (changes 1–3)

`check_wf_variants.py`, whole 640³ box, production configuration (real space, mass-weighted spin inputs, J = stars + AGN). Fifteen render snapshots, `run_wf_variants.sh`, 2026-11-15. The variants:
- V0: the production calculation; matches the production maps per cell to ≤ 3×10⁻⁵ at all 15;
- V1: V0 plus change 1;
- V2: V1 plus change 2;
- V3: V2 plus change 3.

Change 4 was not in this run; it is measured from the new production maps.

**Global signal, volume mean ⟨δT_b⟩ [mK]; the changes are relative to V0:**

| render | z | x̄_α (V0) | V0 | V1 − V0 | V2 − V0 | V3 − V0 |
|---|---|---|---|---|---|---|
| 0 | 29.428 | 0 | −2.7442 | 0 | 0 | 0 |
| 30 | 20.617 | 4.3×10⁻⁴ | +0.2974 | +0.0002 (+0.06%) | −0.0000 | +0.0015 (+0.51%) |
| 60 | 16.287 | 0.051 | +0.6721 | +0.0098 (+1.46%) | −0.0056 (−0.84%) | +0.0783 (+11.64%) |
| 80 | 14.415 | 0.30 | +1.3330 | +0.0239 (+1.79%) | −0.0233 (−1.75%) | +0.1872 (+14.05%) |
| 100 | 12.746 | 1.27 | +5.8930 | +0.0134 (+0.23%) | −0.0262 (−0.44%) | +0.0955 (+1.62%) |
| 110 | 12.130 | 2.10 | +9.4138 | +0.0030 | −0.0174 | +0.0070 (+0.07%) |
| 120 | 11.638 | 3.09 | +12.6516 | −0.0025 | −0.0107 | −0.0381 (−0.30%) |
| 130 | 11.164 | 4.41 | +15.6085 | −0.0045 | −0.0061 | −0.0513 (−0.33%) |
| 140 | 10.708 | 6.11 | +17.9247 | −0.0044 | −0.0035 | −0.0462 (−0.26%) |
| 150 | 10.269 | 8.21 | +19.4682 | −0.0034 | −0.0020 | −0.0352 (−0.18%) |
| 156 | 10.014 | 9.67 | +20.0350 | −0.0028 | −0.0015 | −0.0287 (−0.14%) |
| 174 | 9.282 | 15.0 | +20.3554 | −0.0015 | −0.0006 | −0.0143 (−0.07%) |
| 200 | 8.310 | 24.9 | +18.1248 | −0.0005 | −0.0001 | −0.0047 (−0.03%) |
| 242 | 7.207 | 41.5 | +12.3624 | −0.0001 | −0.0000 | −0.0011 (−0.01%) |
| 300 | 6.346 | 61.0 | +5.6962 | −0.0000 | +0.0000 | −0.0002 (−0.00%) |

**Power spectrum, Δ²(k) of V3/V0 over k = 0.027–5.3 h/cMpc (23 log bins), and the minimum cross-correlation r(V0, V3):**

| z | V3/V0 | min r |
|---|---|---|
| 29.4 | 1.0000 | 1.00000 |
| 20.6 | 0.998–1.006 | 0.99999 |
| 16.3 | 0.932–0.961 (one low-k bin 1.008) | 0.9989 |
| 14.4 | **0.905–0.917** (two lowest-k bins 0.898, 0.951) | 0.9985 |
| 12.7 | 0.938–0.957 | 0.9996 |
| 12.1 | 0.965–0.977 | 0.9999 |
| 11.6 | 0.983–0.989 | 1.0000 |
| 11.2 | 0.994–0.997 | 1.0000 |
| ≤ 10.7 | 0.992–1.006 | 1.0000 |

At z = 14.4 the individual steps are V1/V0 = 0.988–0.994, V2/V0 = 1.000–1.004 and V3/V0 = 0.905–0.917.

**Reading:**
- **A rescaling, not a reshaping.** r ≥ 0.998, and V3/V0 is nearly flat in k: morphology and power-spectrum shape are unchanged.
- **The mean rises while the power falls (z ≈ 12.7–16).** The colour temperature makes cold absorbing gas absorb less (T_c > T_K when T_S > T_K). That lowers the contrast between warm emitting and cold absorbing pixels, so Δ² falls by up to 9% while the mean, a near-cancellation of the two, moves towards emission by up to 14%.
- **The sign flips at z ≈ 12.** Where the coupled gas is warm (10²–10³ K), Hirata's S̃_α is 1–2% below the old exponential because of fine structure (his Fig. 2, "Old" vs "+FS"), so emission falls slightly.
- **Change 3 dominates; changes 1 and 2 are ≲ 1–2% and partly cancel.**

**Production result (job 29237390, 2026-11-16; all four changes, `Tb_maps_prod/H_real_wf`):**
- **Run:** 305/305 maps, 0 failed; 0 unconverged pixels in every snapshot; no H I at T_K or T_S < 2 K in any snapshot.
- **Check:** `check_Tb_maps.py --real-space` at render 80 (z = 14.42): worst cell 2.1×10⁻⁵, best-fit ratio 0.99998, the same level as the SPIN_MODEL 0 checks.
- **Check at low z (2026-11-16):** `check_Tb_maps.py --real-space` at render 300 (z = 6.35; 3 slabs, 4.9×10⁶ cells): best-fit C/independent 0.9999852, worst cell 1.48×10⁻⁵ (the same constant offset as the earlier verified maps, e.g. the reference at 242); `check_ps.py` at 300: Δ² C/numpy 0.999996–1.000001 over 1240 k bins (0.019–6 h/cMpc). OK.
- **x_CMB (change 4):** volume mean 0.9912 (z = 27.6), 0.9925 (21.9), 0.9932 (18.4), 0.9937 (15.9), 0.9943 (14.1), 0.9957 (12.4), 0.9971 (11.4), 0.9983 (10.5), 0.9995 (9.0), ≥ 0.99999 at z ≤ 6; pixel minimum ≥ 0.987. So τ₂₁ ≲ 0.026 everywhere, and the optical-depth factor on δT_b is ≤ 1.3% in every pixel, whatever the weighting. (The volume mean includes ionized pixels at exactly 1; from 2026-11-16 the log also prints the H I-weighted mean.) The earlier estimate "linear τ makes |δT_b| ~4% too high at mean density at z ≈ 20" assumed cold adiabatic gas; Lumina's gas is warmer.
- **At z ≈ 29, J_α = 0 and x_c ≪ 1:** δT_b ∝ x_CMB x_c/(x_CMB + x_c) (1 − T_γ/T_K) ≈ x_c (1 − T_γ/T_K), so the two x_CMB factors cancel to first order (measured −2.506 vs −2.507 mK).

Global ⟨δT_b⟩ [mK], SPIN_MODEL 1 vs SPIN_MODEL 0 (`plot_global_Tb.py`, logs 29237390 vs 25884455):

| z | 28.97 | 24.92 | 19.93 | 16.96 | 14.95 | 12.98 | 11.98 | 10.98 | 10.01 | 9.01 | 8.00 | 7.01 | 6.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPIN_MODEL 1 | −2.506 | −0.834 | +0.424 | +0.719 | +1.073 | +4.930 | +10.355 | +16.547 | +19.986 | +19.996 | +16.815 | +10.943 | +3.007 |
| SPIN_MODEL 0 | −2.507 | −0.834 | +0.421 | +0.671 | +0.917 | +4.809 | +10.382 | +16.622 | +20.035 | +20.017 | +16.823 | +10.945 | +3.008 |
| change | 0.0% | 0.0% | +0.7% | +7.2% | **+17.0%** | +2.5% | −0.26% | −0.45% | −0.24% | −0.10% | −0.05% | −0.02% | −0.03% |

This is consistent with the V0–V3 measurement above (+14% at z = 14.4 for changes 1–3); x_CMB adds about −0.1 to −0.3% at z ≈ 10–12.

---

## 5. Implementation

| File | What |
|---|---|
| `spin_wf.h` | `wf_spin()`: S̃_α, T_c, x_CMB and the T_S iteration for one pixel. Included only by `PS_21cm_mpi.c`; the 3.46 cm code is untouched. |
| `PS_21cm_mpi.c` | `SPIN_MODEL` compile option: 1 (default) uses `spin_wf.h`, 0 the old functions. τ_GP and τ₂₁T_S per proper n_HI are computed once per snapshot from H(z) with Λ; per pixel n_HI = (1 − HII_Fraction) n_H. The map is multiplied by x_CMB. New log lines report τ_GP(x_HI = 1, ⟨n_H⟩), τ₂₁T_S(⟨n_H⟩), ⟨x_CMB⟩ and its minimum, the H I share with T_K or T_S < 2 K, and the number of unconverged pixels (also a stderr warning). The map's `content` attribute records the model. |
| `run_21cm.sh` | `-DSPIN_MODEL=1`; output `Tb_maps_prod/H_real_wf` (the SPIN_MODEL 0 maps stay in `Tb_maps_prod/H_real`); an 8-task array; the no-heating reference build is switched off (`TK_REF=""`, not being redone; its maps were made with SPIN_MODEL 0); the binary must be newer than `spin_wf.h`. |
| `tb_physics.py` | `Line(..., spin_model=1)` (default; 0 for the old maps). It is an independent implementation: T_S is found by bisection in ln T_S, not by iteration. Used by `check_Tb_maps.py`, `lightcone_tb.py` and `test_resolution_Tb.py`. |
| `check_Tb_maps.py` | `--spin-model {0,1}` (default 1). |
| `plot_global_Tb.py` | `--log` accepts a quoted wildcard, which merges the array job's logs. |
| `check_wf_variants.py`, `run_wf_variants.sh` | the V0–V3 measurement of Section 4. |

`-DSPIN_MODEL=0` reproduces the old maps bit for bit (tested).

---

## 6. Validation (2026-11-15, session sandbox; OpenMPI, FFTW-MPI and parallel HDF5 installed for the test)

1. **The two solvers agree.** `wf_spin` (C, fixed point) and `tb_physics.solve_ts_wf` (Python, bisection) were compared on a grid of 6,480 inputs: z = 6–30, T_K = 2–10⁶ K, 1+δ = 0.05–10⁵, x_HI = 0–1, J = 0–10⁻⁹ cgs. τ₂₁ reaches 7×10⁴ and x_CMB 1.3×10⁻⁵. Max |T_S,C/T_S,py − 1| = 2.3×10⁻¹⁰; all converged (median 5, max 66 iterations).
2. **End to end** on a synthetic 32³ render at z = 14.4 (cold, T_K ~ 6 K, wide δ) and z = 8.3:
   - `SPIN_MODEL=0` maps are bit-identical to the pre-change code;
   - `check_Tb_maps.py` passes with the matching `--spin-model`: worst cell 5.7×10⁻⁷ (model 1) and 5.9×10⁻⁷ (model 0);
   - the wrong model fails, as a negative control (ratio 0.19 and 0.92);
   - `check_ps.py` agrees: Δ² ratio 1.00001.
3. **Cost:** about 260 ns per pixel, i.e. about 70 s extra per 640³ map single-threaded. That is about 280 s per snapshot, about 24 h for 305 maps serially, about 3 h as the 8-task array.

---

**Independent review (2026-11-16):** a separate reviewer re-derived every coefficient from the paper text and tested the code.
- **Confirmed:**
  - Hirata Eqs. 38–42 (coefficients, where T_S⁻¹ enters S̃_α, ξ, the sign of T_c), Venumadhav Eqs. 2, 13–15, gp_coef and tauTs_coef (units, proper densities);
  - the map normalisation with x_CMB, and the OpenMP and MPI reductions (identical for 1 vs 4 threads, 1 vs 2 ranks);
  - SPIN_MODEL=0 bit-identical to the pre-change code (rin and PS files, two optimisation levels, three mock snapshots);
  - C iteration vs Python bisection over 2×10⁵ random inputs, max 1.1×10⁻⁹;
  - vector-z broadcasting.

  No defect changes the production SPIN_MODEL 1 maps.
- **Fixed after the review** (re-tested: SPIN_MODEL=0 still bit-identical to the original code; SPIN_MODEL 1 maps identical before and after the fixes):
  1. Hirata's Eq. 42 gives 1/T_c ≤ 0 below T_K ≈ 0.4 K. `wf_spin` now flags it, and `PS_21cm_mpi.c` stops with "FATAL: invalid T_S" and writes no map. Unreachable with Lumina's MinGasTemp but reachable with the no-heating reference; tested with four 0.3 K pixels. The Python bisection already raised.
  2. The iteration is damped after 100 steps. It could stall when τ₂₁ ≫ 1 and T_K ≫ T_S; such a case now converges in 106 steps.
  3. `run_21cm.sh`:
     - a fixed NTASK = 8, so a partial resubmission (`--array=3`) does its share instead of nothing;
     - the compile flags are stored in `bin/PS_21cm_mpi.flags` and checked at run time (rebuild once: `bash run_21cm.sh compile`);
     - OUT_DIR must match the model (`H_real_wf` ⇔ SPIN_MODEL 1).
  4. `check_Tb_maps.py` reads the model from the map's `content` attribute and stops on a mismatch.
  5. `tb_physics` callers take the model explicitly:
     - `test_cells_Tb.py`, `test_resolution_Tb.py`, `make_lightcone_tb.py` take `--spin-model` (default 1; 0 reproduces earlier results);
     - `make_lightcone_tb.py` tags its files and refuses to resume a file of the other model;
     - `check_lightcone_tb.py` and `make_lightcone_ps.py` now compare with `H_real_wf` by default.
  6. `check_wf_variants.py`: the fundamental modes were lost from the first k bin (float32 rounding); fixed.
  7. The log adds the H I-weighted ⟨x_CMB⟩ (the volume mean includes ionized pixels at 1).
- **Kept, documented:** τ₂₁ for x_CMB uses the absolute n_HI, while the signal normalises to the gas mean; the two differ by ⟨ρ_gas⟩/(Ω_b ρ_c) ≈ 1.004, i.e. ≲ 4×10⁻⁵ in δT_b.

## 7. What is still approximate in the 21 cm spin temperature

- **The 640³ pixel temperature (resolution).** At z ≳ 10 each pixel's T_K is a mass-weighted mix of hot and cold gas. S̃_α, T_c, x_c and 1 − T_γ/T_S are nonlinear in T_K, so exact coupling physics applied to a mixed pixel is still approximate. This is the largest known error at z ≳ 10 (the 21 cm map there is an upper limit; JALPHA_NOTES §5–7). It has not been measured at z = 13–16, the window where changes 3 and 4 act.
- **Velocity gradient** in τ_GP and τ₂₁ (Sec. 3.2): second order; same sign as the density term.
- **CMB and Lyα heating of the gas** (Venumadhav+18, Sec. 3.4): part of Lumina's thermal history, not post-processing.
- **One S̃_α for continuum and injected photons** (≤ 3%, Hirata).
- **Hirata's fit range:** T_K, T_S ≥ 2 K, 10⁵ ≤ τ_GP ≤ 10⁷. The log reports the H I share outside it.

---

## 8. References (arXiv versions read 2026-11-15)

- Hirata C. M., 2006, MNRAS 367, 259: *Wouthuysen-Field coupling strength and application to high-redshift 21 cm radiation*. astro-ph/0507102
- Chuzhoy L., Shapiro P. R., 2006, ApJ 651, 1: *UV pumping of hyperfine transitions in the light elements, with application to 21-cm hydrogen and 92-cm deuterium lines from the early universe*. astro-ph/0512206
- Venumadhav T., Dai L., Kaurov A., Zaldarriaga M., 2018, PRD 98, 103513: *Heating of the intergalactic medium by the cosmic microwave background during cosmic dawn*. arXiv:1804.02406
- Furlanetto S. R., Oh S. P., Briggs F. H., 2006, Phys. Rep. 433, 181: *Cosmology at low frequencies: the 21 cm transition and the high-redshift Universe*. astro-ph/0608032
- Furlanetto S. R., Pritchard J. R., 2006, MNRAS 372, 1093: *The scattering of Lyman-series photons in the intergalactic medium*. astro-ph/0605680
- Gunn J. E., Peterson B. A., 1965, ApJ 142, 1633
- 21cmFAST (`src/py21cmfast/src/heating_helper_progs.c`: `taugp`, `Salpha_tilde`; `SpinTemperatureBox.c`: the T_S iteration and x_CMB), GitHub master, read 2026-11-15

---

## 9. How to run (ORCD)

```bash
git pull origin main
bash run_21cm.sh compile
mkdir -p logs && sbatch -p ou_mki run_21cm.sh
```

Then:

```bash
grep -h "SPIN_MODEL 1:" logs/PS_21cm_<jobid>_*.out | sort | uniq -c | sort -rn | head
python check_Tb_maps.py --line H --dir <ren_640> --snap 80 --J <J_DIR>/Jalpha_H_80.hdf5 --J <J_DIR>/Jalpha_AGN_H_80.hdf5 --rin /orcd/data/mvogelsb/004/chcho/Tb_maps_prod/H_real_wf/RIN_80.hdf5 --real-space
python plot_global_Tb.py --log "logs/PS_21cm_<jobid>_*.out" --label "SPIN_MODEL 1" --log logs/PS_21cm_25884455.out --label "SPIN_MODEL 0" --out global_Tb_wf.png
```

Downstream steps to repoint to `Tb_maps_prod/H_real_wf` once the maps exist:
- H VIDs (`run_vid.sh`, with a new output directory, since existing VIDs are skipped);
- the light cone (`run_lightcone_tb.sh`, which uses `tb_physics` and so the new physics by default);
- `VID.ipynb` and the other figures that read the H maps.
