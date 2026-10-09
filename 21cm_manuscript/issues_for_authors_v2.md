# Follow-up audit: status of outstanding issues (version of 2026-10-06)

This is a follow-up to two earlier documents:

1. `issues_for_authors.md` (version of 2026-08-20) — a detailed code-vs-manuscript audit,
   items A1-A3 (affect stated conclusions), B1-B6 (quantitative issues), C1-C6 (smaller
   inconsistencies).
2. `LUMINA_scientific_review_and_revision_plan.pdf` (manuscript version of 2026-08-17) — a
   higher-level scientific review with a P0/P1/P2 priority matrix, a pre-submission
   checklist, and a sign-off tracker.

Every item from both documents was re-checked against the current `oja_template.tex` and
the underlying analysis code/data (`/orcd/data/mvogelsb/004/chcho/...`). Status key:
**CLOSED** (fixed or accurately disclosed), **PARTIAL** (real progress, but the fix is
incomplete, cosmetic, or introduces a new gap), **OPEN** (unchanged).

**Correction note (2026-10-06):** the first pass of this document (dated 2026-10-05) used
several analysis notebooks (`visual.ipynb`, `BSD.ipynb`, `global_snaps.ipynb`,
`PS2Bubble.ipynb`, `visual_PS.ipynb`, `PS_redshift.ipynb`, `one-stat.ipynb`,
`subvol_VID.ipynb`, and the `old_21cm/LUMINA_ren5_*.csv` sub-volume set) at their old
top-level / `PS_21cm/` / `old_21cm` locations. Those notebooks have since been consolidated
into a new, current directory, `/orcd/data/mvogelsb/004/chcho/21cm_analysis/`, and the
authoritative sub-volume dataset now lives at
`/orcd/data/mvogelsb/004/chcho/global/subvol_125/LUMINA_sub/`. Every notebook-dependent
finding below was re-run against these correct, current locations. C-code (`.c` files) and
`.py` production scripts were not affected by this reorganization and did not need
re-checking. One conclusion changed as a result (**C4**, now closed at the data level), one
weakened (**A2**, figure provenance now in question), and one new systemic issue emerged
(duplicate/orphaned notebook cells writing to the same output filenames as the real,
current figure-production cells — see A1, A2, C4 below).

**Overall (updated): 4 of 15 lettered items CLOSED, 10 PARTIAL, 1 OPEN.** Three P0/P1
validation workstreams from the revision plan (3He+ decomposition, dense-gas/effective-ISM
masking, helium coupling benchmark) are essentially untouched. The revision plan's own
sign-off tracker (last page) is still the unfilled "Not started / In progress / Closed"
template — it has not been used as an actual tracking log. The general pattern: unit/
code-level bugs were fixed; interpretive overclaims were mostly reworded/caveated; the more
expensive validation work (reruns, decompositions, synthetic-field tests) was not done. A
second, cross-cutting pattern emerged on re-verification: **several of the manuscript's
figures cannot be traced with certainty to the notebook cell that produced them** — active
and commented-out `savefig` calls are scattered across multiple cells (sometimes with the
*same* output filename pointing at *different*, stale data), so a careless "run all" risks
silently reverting a fixed figure to its old, buggy version. See A1 (`BSD.png`), A2
(`ps_hydro.png`), and C4 (`global.png`/`local.png`).

---

## A. Issues that affect stated conclusions

### A1. Bubble-size figure (`figs/BSD.png`) — **PARTIAL**

- The π-vs-2π convention bug is fixed: `PS_21cm/PS2Bubble.ipynb` now computes
  `R_bubble = 2.5 / k_peak` (not `2.0*np.pi/k_peak`), matching the tex's derivation
  (`oja_template.tex:1061-1064`, "$kR\approx2.46$ ... adopt $R_\mathrm{ion}\approx2.5/k_\mathrm{peak}$").
  Stale print-string labels in the same notebook cells still say `"(using R = 2*pi/k_peak)"`
  — cosmetic, should be cleaned up.
- The h-unit mismatch is fixed at the root: `PS_21cm/bispec/serial_iso_fHII.py:27` now sets
  `BOX_LEN = 338.30` ($h^{-1}$cMpc), matching the figure captions' "h cMpc$^{-1}$" labeling.
- **Unresolved provenance**: the `/hubble` division on the bispectrum sign-flip radii in
  `BSD.ipynb` is only correct if the hardcoded radius arrays were regenerated under the new
  `BOX_LEN`. This can't currently be verified — the only cell that saves `BSD.png` writes a
  local copy dated 2026-08-14 (pre-fix), and no script in scope writes directly to
  `21cm_manuscript/figs/BSD.png`. **Action: confirm which script produced the embedded
  figure and re-verify the radius arrays before resubmission.**

### A2. MWA noise floor (`figs/ps_hydro.png` panel b) — **REVISED: arithmetic CLOSED, figure provenance OPEN**

*(First pass checked this against a now-superseded notebook copy; re-verified
2026-10-06 against the current `21cm_analysis/visual_PS.ipynb`.)*

The squaring fix is real: `_a2(arr) = arr**2` is present, and the computed floor
(2.7²+13.8² ≈ 198 mK² at z=6.5, k=0.142) matches Nunhokee Table 4. The text fix also
holds — "HERA is already reaching the sensitivity thresholds" / "detectable at current
integration depths" are gone; `oja_template.tex:728` reads "current observational limits
remain well above both predictions."

**But the figure-regeneration claim does not hold up.** The cell containing the `_a2` fix
ends in `plt.show()`; its only `savefig` is commented out and targets `"ps_helium.png"`,
not `ps_hydro.png`. The cell that actually writes `figs/ps_hydro.png`
(`21cm_analysis/PS_redshift.ipynb`, cell 18, `plt.savefig("ps_hydro.png", dpi=200)`) plots
only the raw theoretical power spectra — grepping that notebook for `_a2`/`Nunhokee`/
`thermal`/`sample` returns zero hits. **There is no code path that both contains the
noise-floor fix and writes to the actual manuscript figure filename.** Neither notebook
references a `21cm_manuscript/figs/` path or an absolute path anywhere, so provenance
can't be established from the code; the mtime ordering (figures dated 17:06, notebooks
dated 23:42-00:13 — i.e. notebooks "newer") is not meaningful either, since those notebook
timestamps are from last night's directory reorganization, not necessarily a re-run.
**Recommend the authors explicitly re-run and re-export `figs/ps_hydro.png` from the
corrected cell before resubmission**, the same action item as A1's BSD.png.

### A3. f_heat = 0.7/0.3 curves (Fig. `global.png` panel c) — **CLOSED (via relabeling)**

The figure caption (`oja_template.tex:480`) now explicitly describes the curve as "an
interpolation between the fiducial and adiabatic signals," not an evaluation of
intermediate heating efficiency. `oja_template.tex:1221` adds the concavity caveat almost
verbatim from the prior audit ("this interpolation overstates the absorption expected from
a self-consistent calculation... illustrate the direction and sensitivity of the effect
rather than its precise amplitude"). The old `\subsection{...Adiabatic Limit}` subsection
with the unsupported EDGES/power-spectrum claims is now wrapped in
`\begin{comment}...\end{comment}` (lines 1242-1268) and no longer compiles.

Note this is the cheaper of the two fixes the original audit offered: `visual.ipynb`
(lines ~232, 1566) still computes `Tb_mix = (1-f)*Tb_ad_interp + f*Tb_full`, a post-hoc
blend of two final curves, not a per-cell $T(f) = T_\mathrm{ad} + f\Delta T$ rerun. That's
fine given the relabeling, but the claims removed (e.g. "we evaluate intermediate heating
efficiencies") are confirmed gone by direct grep.

---

## B. Quantitative issues behind the figures

### B1. Global curve bias (product of separately-averaged means) — **OPEN**

`global_snaps.ipynb` (cells ~23/27) still evaluates $\bar T_b = 27\,\bar x_\mathrm{HI}\,
(1-\bar T_\gamma/T_S(\text{mean inputs}))$ on pre-averaged scalar inputs — exactly the
flagged construction, not $\langle x_\mathrm{HI}(1+\delta)(1-T_\gamma/T_S)\rangle$.
`global/compute_neutral_temp.c` confirms $T_K$ entering $T_S$ is still the neutral-gas-only
mean. No true-volume-average recomputation is referenced anywhere in the Global evolution
section (`oja_template.tex:450-569`); the caption still says "volume-weighted global
average" with no qualifier. **The ~15-18% bias measured in the original audit is neither
corrected nor disclosed.**

### B2. Adiabatic reference anchor — **PARTIAL**

`oja_template.tex:499` now hedges the adiabatic curve as "an approximate baseline rather
than a strict bound," close to the revision plan's suggested wording. But the ~1.8× anchor
offset itself (34.20 K vs. canonical ~17-20 K at z=29.4, confirmed unchanged in
`temp_ad.csv`) is never quantified in the text.

**New discrepancy**: `oja_template.tex:496` now states the adiabatic selection applies
"only to highly neutral cells ($x_\mathrm{H\,II} < 10^{-3}$)," but
`PS_21cm/TK_ad/get_T_ad.c:203` uses threshold `1.0e-2` (comment shows `1.0e-4 / 1.0e-2` were
the two tested variants — matching the `LUMINA_ALL_ad_1e-4.csv`/`_5e-4.csv` files on disk;
there is no `1e-3` variant). **Text and code disagree on the selection threshold.**

### B3. High-z MFP bubble sizes — **PARTIAL**

`oja_template.tex` (~line 1091) now explicitly states the resolution caveat ("~40% of MFP
rays at z≈9.2 end within a single 0.78 cMpc cell"), and the quoted z≈5.8 number ("≈65 cMpc")
now matches `bubble_statistics_hydrogen.csv` (≈61-67 cMpc) — both fixed relative to the
original audit.

**New discrepancy**: the quoted z≈9.2 number ("≈1 cMpc") is now off by roughly 2× against
the archived statistic (median ≈2.00-2.04 cMpc at z=9.09/9.16).

### B4. Lyα background: sphere vs. shell averaging; source list — **PARTIAL**

- Sphere-vs-shell kernel: now accurately **disclosed**, not fixed. New text
  (`oja_template.tex:1465-1474`) states the field is built from a cumulative top-hat sphere
  average, gives the window-function equation, and quantifies the impact ("weights nearby
  emission up to 1.5 times more heavily than a true shell average... confined to z≳10").
  Code is unchanged (`compute_Jalpha.c:457,673-699`, `smooth_tophat()` at `R_mid`, no
  differencing of consecutive radii) — consistent with the new, now-accurate description.
- Source-list overstatement: **still open**. The same paragraph
  (`oja_template.tex:1446`) and `sec:others` (~1507-1510) still claim hot ISM, HMXBs, and
  AGN all "contribute to the Lyα field directly" / are summed over "all contributing
  populations." Code (`PS_21cm_mpi.c` comment "[CHANGE 3] sum stellar + AGN J_alpha", line
  ~308) confirms only stellar + AGN boxes are summed; no hot-ISM/HMXB term exists in
  `compute_Jalpha.c`.
- He Gunn-Peterson depth: **still open**. `PS_3cm_mpi.c:343`
  (`compute_S_alpha(z, T_K, f_HI)`) still passes the hydrogen neutral fraction, not
  $x_\mathrm{HeII}$, into the He Sobolev calculation — unchanged from the original audit
  (numerically inconsequential, but a species-level inconsistency).

One genuine fix in the same area: the X-ray energy double-counting concern from the
revision plan (§3.2) is now closed — `oja_template.tex:1446` explicitly states secondary-
electron Lyα is omitted "consistent with our assumption of maximal X-ray heating efficiency
($f_\mathrm{heat}=1$), which by construction leaves no absorbed X-ray energy in the
excitation channel."

### B5. RSD claims — **CLOSED**

`oja_template.tex:231` now accurately states RSD enters the hydrogen maps only through the
redshift-space renders at $640^3$/0.78 cMpc grid resolution, that the global curve omits
the RSD term, and that the helium field is computed in real space. This matches code
exactly (`PS_21cm_mpi.c` reads `SpectralDensityHI`; `PS_3cm_mpi.c` reads real-space
`Density`). The old "accounted for down to the particle resolution" overstatement is gone.

### B6. Panel (c) sensitivity curves (`figs/PS_H_z.png`) — **PARTIAL**

The LOFAR contradiction is fixed: `oja_template.tex:749` now explicitly bounds the LOFAR
claim to "$5.8 \lesssim z \lesssim 7.3$," matching the figure. But the σ-level of the noise
columns is still undocumented: headers of `PS_21cm/obs/simu_noise_ps_*.txt` still give only
"$\Delta^2$ (mK$^2$), $\Delta^2_\mathrm{err}$ (mK$^2$)" with no σ-level stated, and
`PS_redshift.ipynb` hardcodes the legend label "$2\sigma$" without applying a ×2 factor or
citing confirmation against Barry (2022) that the tabulated error column is already 2σ.
**If it is 1σ, both the plotted curve and the "2σ" label are a factor of 2 too permissive.**

---

## C. Smaller inconsistencies and presentation issues

### C1. Bispectrum normalization equation — **CLOSED** (main issue); minor nit remains

The dimensionally-wrong $\tilde B$ equation is fixed: `oja_template.tex:799-801` now reads
$\tilde B = B/\sqrt{(k_1k_2k_3)^{-1}P_1P_2P_3}$, matching the standard literature form cited
in the revision plan. The raw bispectrum definition at line 793
($B = \langle\Delta\Delta\Delta\rangle/[(2\pi)^3\delta_D]$) still "divides by the Dirac
delta," but this is standard shorthand in the cited literature (Watkinson 2017 etc.), not
a real error — low priority.

### C2. Bispectrum documentation — **PARTIAL**

A new, genuine disclosure: `oja_template.tex` (~lines 876, 898) now documents the k₁
selection rule ("select $k_1=k_2$ such that the sign flip falls in the stretched regime,
$0.5\le\cos\theta\le1$") and gives a bracketed radius range
$\pi/(2k) \le R_\mathrm{ion} \le \pi/(\sqrt3 k)$, replacing the old undocumented, ~2×
varying choice. Still missing: the number of θ-bins (11) and any uncertainty/error-bar
estimate on the sign-flip measurements; no caveat on comparing normalized-statistic
amplitudes across different fields.

### C3. VID section — **PARTIAL** (3 of 5 sub-items closed)

- (i) He midpoint "x̄≈0.5, z≈4.3" overstatement — **CLOSED**, the specific erroneous quote
  is gone from the captions.
- (ii) H skewness/kurtosis minima vs. midpoint — **CLOSED**,
  `oja_template.tex:619` now says the minima "occur shortly before the midpoint" rather
  than claiming exact coincidence — matches the csv (minima at z≈7.0-7.3).
- (iii) "variance drops ~2 orders of magnitude" vs. plotted std — **CLOSED**, line 616 now
  correctly describes "~1 mK by z=5" with no orders-of-magnitude claim.
- (iv) Undocumented `PHYSICAL_FLOOR=0.001` PDF floor/normalization convention — **OPEN**,
  still hardcoded in `PS_21cm/one-stat.ipynb`, still undisclosed in the VID section.
- (v) Box-convergence sub-volume test only at x̄=0.1 per species — **OPEN**, unchanged
  (Figure `fig:subbox` caption still only covers $x_\mathrm{HII}=0.1$ / $x_\mathrm{HeIII}=0.1$).

### C4. Global figure caption: 125 sub-volumes claimed, 124 on disk — **REVISED: now CLOSED at the data level, provenance unverified**

**Correction to this document's own previous entry.** The first pass checked
`old_21cm/LUMINA_ren5_*.csv` (124 files) — a directory literally named "old," i.e. the
stale, pre-fix dataset. The current, correct location is
`/orcd/data/mvogelsb/004/chcho/global/subvol_125/LUMINA_sub/`, regenerated 2026-09-27,
which contains the full set: `find .../LUMINA_sub -maxdepth 1 -iname "LUMINA_sub_*.csv" |
wc -l` → **125** (all combinations of $x,y,z \in \{0..4\}$ present, none missing).

The current figure-production cell (`21cm_analysis/visual.ipynb`, cell 2 — the 4-panel
cell matching the Fig. global caption's panels (a)-(d) exactly, including the f_heat-mixing
panel (c)) loops over all $5^3=125$ combinations reading from this corrected path
(`.../global/subvol_125/LUMINA_sub/LUMINA_sub_{x}_{y}_{z}.csv`) with a silent
`try/except: continue` — but since all 125 files now exist, none are actually skipped.
**If this cell is what produced the embedded `figs/global.png`, the "125 vs. 124"
undercount is fixed and the caption is accurate.**

Caveat, same pattern as A1/A2 above: this cell's `plt.savefig(...)` is commented out, so
provenance can't be proven directly. Worse, the notebook also contains two clearly
**orphaned, stale cells (22, 23) that still call `plt.savefig("global.png", ...)` and
`plt.savefig("local.png", ...)` with an uncommented, active savefig** — but these use
completely different, old data (`./THESAN-XL_Z4_Pop2.csv`, `./output_21cm/XL_ren5_{x}-{y}-{z}.csv`,
the pre-fix path) and a different, cruder 3-panel layout that does not match the published
caption at all. Content-wise, cell 2 is almost certainly the true source (it alone matches
the caption), but **the presence of a second, active, same-filename savefig pointing at
stale data is a landmine** — a future "run all" on this notebook would silently overwrite
the correct figure with the old 124-subvolume, wrong-panel-layout version. Recommend
deleting cells 22/23 (or redirecting their filenames) and uncommenting the real savefig in
cell 2.

### C5. Appendix A wording vs. code — **mostly CLOSED**, one new issue

- τ_GP "globally averaged" vs. local per-cell fractions — **CLOSED** (line 1426 now states
  local values are used per cell, volume averages for the global signal).
- AGN shell count (200 vs. 400) — **CLOSED** (line 1471 now states "$N_\mathrm{shells}=200$
  (400 for the AGN contribution)," matching code).
- Escape-fraction clarification — **CLOSED** (new paragraphs at lines 1496, 1499).
- BPASS metallicity / "same SEDs as on-the-fly" — **WORSENED on re-verification
  (2026-10-06)**: line 1493 states the emissivity uses "the mass-weighted mean
  metallicity of all \lumina star particles **at the emission redshift z′**" — implying
  time-variation. `sec:metallicity` (the "Sensitivity to the stellar spectral model"
  section, ~line 1541) does **not** retract this — it repeats the same time-varying claim
  for Fig. `fig:metallicity`'s own black "fiducial" reference curve.

  Checking the code turned up a **second, previously-unseen complication**: there are now
  two divergent `compute_Jalpha.c` implementations on disk —
  `/orcd/data/mvogelsb/004/chcho/ren_1280/compute_J_alpha_H/compute_Jalpha.c` (the
  full-resolution production code, N_PIX=1280, dated Oct 2) still hardcodes
  `#define METALLICITY 1e-4`, a single fixed value for the whole run — and
  `/orcd/data/mvogelsb/004/chcho/cell_by_cell/compute_Jalpha.c` (a lower-resolution,
  N_PIX=640 code, dated Sep 25) replaces that fixed define with a `Z_HISTORY_FILE`
  mechanism that reads a genuine time-varying $Z(z)$ table built from
  `Z_above4p75.csv`/`Z_below4p75.csv` — i.e., *this* code path matches the tex's
  description. `21cm_analysis/metallicity.ipynb` only builds the $Z(z)$ history and a
  photon-rate diagnostic; it never computes $\delta T_b$, so it can't be confirmed as the
  source of Fig. metallicity's published curve either.

  **Net effect: it's now unclear which pipeline — the full-resolution fixed-Z production
  run, or the lower-resolution time-varying-Z run — actually produced the paper's fiducial
  results**, and whether Fig. metallicity's reference curve used the same fiducial run as
  the rest of the VID/power-spectrum/global-signal figures. This is a sharper problem than
  originally flagged: not just a text-vs-code mismatch, but a fiducial-definition
  inconsistency between two code paths feeding different parts of the same manuscript.
  **Recommend the authors state explicitly which `compute_Jalpha.c` (and which metallicity
  treatment) produced each figure.**
- f_rec deviation (≤1.9%) — not a wording issue; no action needed.

### C6. Text nits — **CLOSED** (all 6 sub-items)

³He abundance now correctly stated relative to hydrogen (lines 140, 154, 636, 1136, 1369);
spontaneous-decay ratio now states "a factor of 690" (line 508); non-isosceles captions no
longer call $\cos\theta=-0.5$ "equilateral" (isosceles captions correctly retain the term,
since $k_1=k_2=k_3$ genuinely holds there); BiFFT citation now matches the README
(`{Scoccimarro2015, Sefusatti2016, Watkinson2017}`, line 844); MFP parameters (threshold
0.5, 192 HEALPix rays, sub-cell interpolation) now disclosed at line 1078.

---

## D. Revision-plan items with no counterpart in `issues_for_authors.md`

These were flagged P0/P1 in the 2026-08-17 review and are not fully reflected above.

1. **3He+ signal decomposition (§3.3)** — **OPEN**. Only a narrative literature-comparison
   paragraph was added (`oja_template.tex:764`, vs. Spina2025/Basu2026 at matched
   k=0.1 h/cMpc, z≈4). The required decomposition (saturated-coupling reference,
   density-only/ionization-only fields, collisional- vs. Lyα-only coupling, uniform vs.
   spatially-varying J_α, real- vs. redshift-space, diffuse-IGM-only vs. all gas,
   cross-spectra vs. AGN distance) does not exist in the tex or in any notebook checked.
   Minor additional nit: the Basu2026 comparison redshift (z=4.18) isn't matched exactly to
   "z≈4" the way the Spina2025 comparison is.
2. **Dense-gas / effective-ISM masking (§3.4)** — **OPEN**. No star-forming-cell, halo, or
   overdensity-masked rerun of helium statistics found anywhere.
3. **Helium Wouthuysen-Field coupling benchmark (§3.5)** — **OPEN**. Appendix still reads
   as a direct extension of the hydrogen formalism; no one-zone benchmark grid or explicit
   agreement/disagreement statement beyond the one narrative sentence from item D1.
4. **Bispectrum interpretive language (§6.6 checklist)** — **OPEN**. "Confirms" is still
   used for shape interpretations (e.g. "...confirms that \ion{H}{2} regions expand as
   relatively compact, spherical bubbles," line 865; similarly for helium and the
   non-isosceles inside-out claim) where the revision plan asked for "consistent with."
   "Positive by construction" is still present verbatim (~line 1023), which the plan notes
   is mathematically false for a mean-subtracted field. The isosceles family is still
   captioned $B_\mathrm{iso}(k,k,k)$ (line 852) rather than $B(k,k,k_3)$ as requested for a
   family with variable $k_3$.
5. **Statistics and topology language (§5)** — **OPEN**. Line 578 still calls both
   skewness *and* kurtosis "configuration-integrated counterparts of the bispectrum" (only
   skewness is; kurtosis connects to the trispectrum). Line 622 still calls the BC peak a
   "sharp, localized phase transition" and uses "late percolation stage" with no
   connected-component/Euler-characteristic/FoF statistic to back it. Line 673 still
   reports helium kurtosis "$>10^3$" near signal disappearance with no convergence or
   bootstrap test.
6. **Subvolume / "converged" terminology (§3.8)** — **PARTIAL**. "125 non-overlapping" is
   now used in the live, compiling text (lines 480, 490, 1156). The old "125 independent"
   wording survives verbatim but only inside an orphaned `\begin{comment}...\end{comment}`
   block (lines 532-558, which also contains the never-referenced `fig:local` discussion) —
   dead source, not a live error, but should be deleted rather than commented to prevent
   reintroduction on a future edit. Separately, "converged distribution" (lines 1156, 1184)
   was never replaced with "full-volume reference distribution" as the plan requested, and
   no resolution/realization test is shown to justify "converged."
7. **Low-z hybrid radiation treatment (§3.9)** — **PARTIAL**. The z=4.75 switch to a
   single AGN-dominated group is now disclosed in the Methodology (lines 211-214) and
   Discussion (line 1279), not just the appendix. Still missing: a continuity-across-z=4.75
   plot, a switch-redshift sensitivity test, and a smaller-volume multi-group comparison.
8. **Detectability language (§3.7/§8.3)** — **PARTIAL**, beyond what's covered in B6. The
   plan's suggested hedged wording ("approaches or exceeds idealized thermal-noise
   estimates... does not by itself establish detectability... requires instrument-specific
   forward modelling") was not adopted in the Power-spectrum subsection itself
   (`oja_template.tex:728, 749` still read as direct detectability claims); equivalent
   caveats do exist in the Discussion's "Observational challenges" section (~1133, 1142)
   but are not cross-referenced from the Results claim.
9. **RSD/gridding/Fourier-convention documentation (§3.10)** — not independently re-verified
   beyond B5 in this pass; the broader asks (mass-assignment scheme, deconvolution, binning,
   covariance, explicit Fourier normalization) were not confirmed present or absent and
   should be checked separately.
10. **Sign-off tracker** — **OPEN**. Still the literal unfilled template
    ("Not started / In progress / Closed," no owner, no date, no evidence) for all 8
    workstream rows.

---

## New issues surfaced during this pass (not in either prior document)

- `figs/BSD.png` provenance is unverifiable — no script in scope writes to that path, and a
  stale pre-fix local copy exists dated 2026-08-14 (see A1).
- `get_T_ad.c` uses an adiabatic-cell threshold of $10^{-2}$; the new tex wording says
  $10^{-3}$ (see B2).
- Bubble-size number at z≈9.2 is now off by ~2× against the archived MFP statistic, where
  the z≈5.8 number was fixed (see B3).
- The reworded BPASS-metallicity sentence (line 1493) makes a more specific, and more
  clearly false, claim about z′-dependence than the text it replaced, since the code still
  uses one fixed metallicity for the whole run (see C5).
- Stale `"(using R = 2*pi/k_peak)"` print-string labels remain in `PS2Bubble.ipynb` cells
  whose actual computation now uses `2.5/k_peak` (see A1) — cosmetic only.
- An orphaned `\begin{comment}` block (`oja_template.tex:532-558`) contains both the
  stale "125 independent sub-volumes" wording and the only reference to `fig:local`, which
  is otherwise unused in the compiled document — recommend deleting rather than leaving
  commented.
