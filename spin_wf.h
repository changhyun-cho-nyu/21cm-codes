/* spin_wf.h
 * =========
 * 21 cm spin temperature and optical-depth factor of one pixel (SPIN_MODEL = 1 of PS_21cm_mpi.c, 2026-11-15).
 * Four changes from the earlier treatment (SPIN_MODEL = 0); record and evidence in WF_PHYSICS.md:
 *
 *  1. Gunn-Peterson depth from first principles, tau_GP = (pi e^2 f_a lambda_a / m_e c) n_HI / H(z)
 *     (Hirata 2006 Eq. 35; = 1.3428e-7 n_HI/H), instead of 3e5 x_HI ((1+z)/7)^1.5 (Furlanetto, Oh & Briggs 2006
 *     Eq. 43, rounded for their cosmology; 1.30x lower for Lumina's).
 *  2. ... with the local n_HI = x_HI n_H of the pixel (S_alpha depends on n_H x_HI / H, Hirata 2006 Sec. 3.2).
 *     The local velocity gradient is not included (real-space maps; Hubble flow only).
 *  3. Hirata (2006) Wouthuysen-Field coupling: S~_alpha(T_k, T_s, tau_GP) (Eqs. 40-41; 1% for T_k, T_s >= 2 K,
 *     1e5 <= tau_GP <= 1e7, continuum photons; 3% injected) and the colour temperature
 *     1/T_c = 1/T_k + 0.405535/T_k (1/T_s - 1/T_k) (Eq. 42), with x~_alpha = 8 pi lambda^2 gamma T_star / (9 A10 T_g) S~ J
 *     (Eq. 38; the same prefactor as before). The same S~ is used for continuum and injected photons (as 21cmFAST).
 *  4. The finite 21 cm optical depth (Venumadhav et al. 2018 Eqs. 13-15): the CMB coupling is weighted by
 *     x_CMB = (1 - exp(-tau_21))/tau_21 in T_s, and delta T_b = x_CMB tau_21 (T_s - T_g)/(1+z), i.e. the exact
 *     (1 - exp(-tau_21)) instead of the linear tau_21. tau_21 = [3 c^3 h A10 n_HI / (32 pi k nu21^2 H)] / T_s.
 *
 *   1/T_s = (x_CMB/T_g + x~_a/T_c + x_c/T_k) / (x_CMB + x~_a + x_c),   solved by fixed-point iteration from
 *   T_s = T_g (Hirata 2006 Sec. 3.3; 21cmFAST), to a relative change < WF_TOL.
 *
 * Inputs (per pixel): T_g = T_CMB(z) [K], T_k [K], A = x_alpha prefactor x J_alpha (x~_a = A S~), x_c,
 * tau_GP (n_HI local, x_HI included), tauTs = tau_21 T_s [K] (n_HI local).
 * Output: T_s, x~_a, S~, x_CMB, tau_21, iterations; converged flag.
 */
#ifndef SPIN_WF_H
#define SPIN_WF_H

#include <math.h>

#define WF_TOL   1e-10
#define WF_ITMAX 500

typedef struct {
    double Ts, xa, S, xcmb, tau21;
    int iter, ok;
} wf_result;

/* (1 - exp(-tau))/tau, accurate for small tau */
static inline double wf_xcmb(double tau)
{
    return (tau > 1e-8) ? -expm1(-tau) / tau : 1.0 - 0.5 * tau;
}

static inline wf_result wf_spin(double Tg, double Tk, double A, double xc, double tauGP, double tauTs)
{
    wf_result r;
    const double Ti  = 1.0 / Tk;
    const double xi  = cbrt(1e-7 * tauGP) * cbrt(Ti * Ti);                       /* Hirata Eq. 41 */
    const double den = 1.0 + xi * (2.98394 + xi * (1.53583 + xi * 3.85289));     /* Hirata Eq. 40 denominator */
    double Ts = Tg, S = 1.0, xa = 0.0, xcmb = 1.0, tau21 = 0.0;
    int it, ok = 0;
    for (it = 1; it <= WF_ITMAX; it++) {
        const double Tsi = 1.0 / Ts;
        S     = (1.0 - 0.0631789 * Ti + 0.115995 * Ti * Ti - 0.401403 * Tsi * Ti + 0.336463 * Tsi * Ti * Ti) / den;
        xa    = A * S;
        const double Tci = Ti + 0.405535 * Ti * (Tsi - Ti);                       /* Hirata Eq. 42 */
        tau21 = tauTs * Tsi;
        xcmb  = wf_xcmb(tau21);
        const double Tnew = (xcmb + xa + xc) / (xcmb / Tg + xa * Tci + xc * Ti);
        const double d = fabs(Tnew - Ts);
        Ts = Tnew;
        if (d <= WF_TOL * Ts) { ok = 1; break; }
    }
    /* final consistency: tau_21 and x_CMB of the returned T_s */
    tau21 = tauTs / Ts;
    xcmb  = wf_xcmb(tau21);
    r.Ts = Ts; r.xa = xa; r.S = S; r.xcmb = xcmb; r.tau21 = tau21; r.iter = it; r.ok = ok;
    return r;
}

#endif
