"""
tb_physics.py
=============
delta T_b of the 21 cm and 3.46 cm lines cell by cell, written independently of the C codes
(PS_21cm_mpi.c / PS_3cm_mpi.c) and shared by check_Tb_maps.py and test_resolution_Tb.py.

  21 cm, spin_model = 1 (default; the production maps from 2026-11-15, PS_21cm_mpi.c SPIN_MODEL = 1; WF_PHYSICS.md):
          1/T_S = (x_CMB/T_g + x~_a/T_c + x_c/T_K) / (x_CMB + x~_a + x_c),  T_g = 2.7255 (1+z) K
          x~_a = 16 pi^2 T* e^2 f / (27 A10 T_g m_e c) S~_a J   [= 8 pi lambda^2 gamma T*/(9 A10 T_g), Hirata 2006 Eq. 38]
          S~_a, T_c: Hirata (2006) Eqs. 40-42 (depend on T_K, T_S, tau_GP)
          tau_GP = (pi e^2 f lambda_a / m_e c) n_HI / H(z), n_HI = x_HI n_H of the cell (local density)
          x_CMB = (1 - exp(-tau_21))/tau_21, tau_21 = 3 c^3 h A10 n_HI / (32 pi k nu^2 H T_S) (Venumadhav+18 Eqs. 13-14)
          dT_b = P sig (1 - T_g/T_S) x_CMB  [= x_CMB tau_21 (T_S - T_g)/(1+z), Venumadhav+18 Eq. 15]
          T_S is found here by bisection in ln T_S (the C code iterates; an independent solver).
  21 cm, spin_model = 0 (the maps made before 2026-11-15):
          T_S = (1 + x_c + x_a) / (1/T_g + (x_c + x_a)/T_K)   (T_c = T_K, no x_CMB, linear tau_21)
          x_a = 16 pi^2 T* e^2 f / (27 A10 T_g m_e c) S_a J,  S_a = exp(-0.803 T^-2/3 (1e-6 tau_GP)^1/3),
          tau_GP = 3e5 x_HI ((1+z)/7)^1.5
  both:   x_c = T*/(A10 T_g) (k_HH n_HI + k_eH n_e + k_pH n_p)
          (Furlanetto, Oh & Briggs 2006 tables, log-log interpolation, held beyond the ends)
  3.46cm: the same with 9, A10_He, T*_He, tau_GP = 6e3 x_HeII ((1+z)/7)^1.5;
          x_c = T*_He/(A10_He T_g) n_e sqrt(8kT/(pi m_e c^2)) c (14.3 eV/kT) a0^2   (McQuinn & Switzer 2009)
  dT_b = P sig (1 - T_g/T_S) [x_CMB], sig = x(1+delta) of the signal; P = tau T_S/(1+z) at <n_H>, exact, with
  H(z) including Lambda: 3 c^3 h A10 n/(32 pi k nu^2 H) (H), c^3 h A10 [3He/H] n/(32 pi k nu^2 H) (3He+).
"""
import numpy as np

try:
    from astropy import constants as _C, units as _u
    C, H_PL, K_B, G = _C.c.cgs.value, _C.h.cgs.value, _C.k_B.cgs.value, _C.G.cgs.value
    M_E, E_CGS = _C.m_e.cgs.value, _C.e.esu.value
    MPC, M_H = _u.Mpc.to("cm"), (1.00782503207 * _u.u).to("g").value
    A0 = _C.a0.cgs.value
    EV = _u.eV.to("erg")
    SRC = "astropy"
except Exception:
    C, H_PL, K_B, G = 2.99792458e10, 6.62607015e-27, 1.380649e-16, 6.67430e-8
    M_E, E_CGS, MPC, M_H, A0, EV = 9.1093837e-28, 4.80320471e-10, 3.0856775814913673e24, 1.6735575e-24, 5.29177211e-9, 1.602176634e-12
    SRC = "CODATA 2018 (hard-coded)"

X_H = 0.76
Y_HE = (1.0 - X_H) / (4.0 * X_H)
F_ALPHA = 0.4162
NU21, A21 = 1.420405751768e9, 2.86888e-15
NU_HE = 8.665649867e9
A_HE = 3.0 * (NU_HE / NU21)**3 * A21          # electron-spin M1, g_upper(3He+) = 1 vs 3 for H
HE3_H = 1.0e-5
OMEGA_L_DEFAULT = 0.6902672
Z_DEC = 124.8                                 # Zier+26 Eq. 27 (no-heating reference)
LAMBDA_LYA = 1215.67e-8                       # cm
SPIN_MODEL_DEFAULT = 1                        # 21 cm: 1 = Hirata S~/T_c, exact local tau_GP, x_CMB (2026-11-15); 0 = before

# Furlanetto, Oh & Briggs (2006) Table 2: kappa_10 [cm^3 s^-1]; p-H from Furlanetto & Furlanetto (2007)
T_HH = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000]
K_HH = [1.38e-13, 1.43e-13, 2.71e-13, 6.60e-13, 1.47e-12, 2.88e-12, 9.10e-12, 1.78e-11, 2.73e-11, 3.67e-11, 5.38e-11,
        6.86e-11, 8.14e-11, 9.25e-11, 1.02e-10, 1.11e-10, 1.19e-10, 1.75e-10, 2.09e-10, 2.56e-10, 2.91e-10, 3.31e-10,
        4.27e-10, 4.97e-10, 6.03e-10, 6.87e-10, 7.87e-10]
T_E = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000]
K_EH = [2.39e-10, 3.37e-10, 5.30e-10, 7.46e-10, 1.05e-9, 1.63e-9, 2.26e-9, 3.11e-9, 4.59e-9, 5.92e-9, 7.15e-9, 7.71e-9,
        8.17e-9, 8.32e-9, 8.37e-9, 8.29e-9, 8.11e-9]
K_PH = [0.40e-9, 0.45e-9, 0.430e-9, 0.369e-9, 0.317e-9, 0.3047e-9, 0.3379e-9, 0.4043e-9, 0.5471e-9, 0.7051e-9, 0.9167e-9,
        1.070e-9, 1.301e-9, 1.480e-9, 1.695e-9, 1.975e-9, 2.201e-9]


def kappa(T, Tt, Kt):
    return 10.0**np.interp(np.log10(T), np.log10(Tt), np.log10(Kt))   # held at the end values, as the C tables


def wf_xcmb(tau):
    """(1 - exp(-tau))/tau"""
    tau = np.asarray(tau, dtype=float)
    return np.where(tau > 1e-8, -np.expm1(-tau) / np.where(tau > 1e-8, tau, 1.0), 1.0 - 0.5 * tau)


def wf_terms(Ts, Tg, T, A, tauGP, tauTs):
    """Hirata (2006) S~_a (Eqs. 40-41), 1/T_c (Eq. 42) and x_CMB at spin temperature Ts"""
    Ti, Tsi = 1.0 / T, 1.0 / Ts
    xi = (1e-7 * tauGP)**(1.0 / 3.0) * Ti**(2.0 / 3.0)
    S = (1.0 - 0.0631789 * Ti + 0.115995 * Ti**2 - 0.401403 * Tsi * Ti + 0.336463 * Tsi * Ti**2) \
        / (1.0 + 2.98394 * xi + 1.53583 * xi**2 + 3.85289 * xi**3)
    Tci = Ti + 0.405535 * Ti * (Tsi - Ti)
    return S, Tci, wf_xcmb(tauTs * Tsi)


def solve_ts_wf(Tg, T, A, xc, tauGP, tauTs, nbis=200):
    """T_S of 1/T_S = (x_CMB/T_g + x~_a/T_c + x_c/T_K)/(x_CMB + x~_a + x_c) by bisection in ln T_S"""
    Tg, T, A, xc, tauGP, tauTs = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (Tg, T, A, xc, tauGP, tauTs)))

    def F(Ts):
        S, Tci, xcmb = wf_terms(Ts, Tg, T, A, tauGP, tauTs)
        xa = A * S
        return 1.0 / Ts - (xcmb / Tg + xa * Tci + xc / T) / (xcmb + xa + xc)

    lo, hi = 0.25 * np.minimum(Tg, T), 4.0 * np.maximum(Tg, T)
    flo, fhi = F(lo), F(hi)
    if np.any(flo <= 0) or np.any(fhi >= 0):
        raise ValueError(f"T_S not bracketed in {int(np.sum(flo <= 0) + np.sum(fhi >= 0))} cells")
    llo, lhi = np.log(lo), np.log(hi)
    for _ in range(nbis):
        mid = 0.5 * (llo + lhi)
        pos = F(np.exp(mid)) > 0
        llo, lhi = np.where(pos, mid, llo), np.where(pos, lhi, mid)
        if np.max(lhi - llo) < 1e-14:
            break
    return np.exp(0.5 * (llo + lhi))


class Line:
    """constants of one line at redshift z for a given cosmology (spin_model: 21 cm only, see the module notes)"""
    def __init__(self, line, z, h, om, ob, ol, spin_model=SPIN_MODEL_DEFAULT):
        self.line, self.z, self.spin_model = line, z, spin_model
        self.Tg = 2.7255 * (1.0 + z)
        H0 = 100.0 * h * 1e5 / MPC
        Hz = H0 * np.sqrt(om * (1.0 + z)**3 + ol)
        nH_mean = X_H * ob * 3.0 * H0**2 / (8.0 * np.pi * G) * (1.0 + z)**3 / M_H
        if line == "H":
            self.Tst, self.A, nu, kxa = H_PL * NU21 / K_B, A21, NU21, 27.0
            self.P = 1e3 * 3.0 * C**3 * H_PL * self.A * nH_mean / (32.0 * np.pi * K_B * nu**2 * Hz * (1.0 + z))   # mK
            self.unit = "mK"
            self.tauTs_c = 3.0 * C**3 * H_PL * self.A / (32.0 * np.pi * K_B * nu**2 * Hz)    # tau_21 T_S per n_HI [K cm^3]
            self.gp_c = np.pi * E_CGS**2 * F_ALPHA * LAMBDA_LYA / (M_E * C * Hz)          # tau_GP per n_HI [cm^3]
        else:
            self.Tst, self.A, nu, kxa = H_PL * NU_HE / K_B, A_HE, NU_HE, 9.0
            self.P = 1e6 * C**3 * H_PL * self.A * HE3_H * nH_mean / (32.0 * np.pi * K_B * nu**2 * Hz * (1.0 + z))  # uK
            self.unit = "uK"
        self.xa_c = 16.0 * np.pi**2 * self.Tst * E_CGS**2 * F_ALPHA / (kxa * self.A * self.Tg * M_E * C)

    def t_ref(self, rho, rho_mean):
        """no-heating reference temperature, analytic (Zier+26 Eq. 27 law with adiabatic compression; TK_REF_ANCHOR=0)"""
        return 2.7255 * (1.0 + self.z)**2 / (1.0 + Z_DEC) * (rho / rho_mean)**(2.0 / 3.0)

    def t_ref_pixel(self, T0, rho0, rho, z0):
        """no-heating reference temperature anchored to the render at z0, pixel by pixel (TK_REF_ANCHOR=1):
        T(p, z0) [(1+z)/(1+z0)]^2 [rho(p, z)/rho(p, z0)]^(2/3), comoving densities"""
        return T0 * ((1.0 + self.z) / (1.0 + z0))**2 * (rho / rho0)**(2.0 / 3.0)

    def tb(self, sig, nH, T, xHII, fII, fIII, J):
        """delta T_b [mK or uK] and (x_c, x_a); nH proper cm^-3, fII/fIII per H nucleus, sig = x(1+delta)"""
        z = self.z
        ne = nH * (xHII + fII + 2.0 * fIII)
        if self.line == "H":
            xHI = 1.0 - xHII
            xc = self.Tst / (self.A * self.Tg) * (kappa(T, T_HH, K_HH) * xHI * nH + kappa(T, T_E, K_EH) * ne
                                                  + kappa(T, T_E, K_PH) * xHII * nH)
            if self.spin_model == 1:
                nHI = xHI * nH
                tauGP, tauTs, A = self.gp_c * nHI, self.tauTs_c * nHI, self.xa_c * J
                TS = solve_ts_wf(self.Tg, T, A, xc, tauGP, tauTs)
                S, _, xcmb = wf_terms(TS, self.Tg, T, A, tauGP, tauTs)
                return self.P * sig * (1.0 - self.Tg / TS) * xcmb, xc, A * S
            S = np.exp(-0.803 * T**(-2.0 / 3.0) * (1e-6 * 3.0e5 * xHI * ((1.0 + z) / 7.0)**1.5)**(1.0 / 3.0))
        else:
            S = np.exp(-0.803 * T**(-2.0 / 3.0) * (1e-6 * 6.0e3 * (fII / Y_HE) * ((1.0 + z) / 7.0)**1.5)**(1.0 / 3.0))
            sigma = (14.3 * EV / (K_B * T)) * A0**2
            xc = self.Tst / (self.A * self.Tg) * ne * np.sqrt(8.0 * K_B * T / (np.pi * M_E * C**2)) * C * sigma
        xa = self.xa_c * S * J
        TS = (1.0 + xc + xa) / (1.0 / self.Tg + (xc + xa) / T)
        return self.P * sig * (1.0 - self.Tg / TS), xc, xa
