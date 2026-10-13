#ifndef CONSTANTS_H
#define CONSTANTS_H

// ============================================================
// Lyman Series Frequencies (Hz)
// ============================================================
static const double NU_LYMAN_ALPHA = 2.466e15;   // n=2
static const double NU_LYMAN_BETA  = 2.925e15;   // n=3
static const double NU_LYMAN_GAMMA = 3.083e15;   // n=4
static const double NU_LYMAN_DELTA = 3.158e15;   // n=5
static const double NU_LYMAN_LIMIT = 3.289e15;   // n=infinity

// ============================================================
// Fundamental Constants (cgs)
// ============================================================
static const double PLANCK      = 6.62607015e-27;   // erg*s
static const double C_LIGHT     = 2.99792458e10;    // cm/s
static const double k_B         = 1.381e-16;        // erg/K
static const double G_cgs       = 6.6743015e-8;     // dyn cm^2 g^-2
static const double e_cgs       = 4.803204712570263e-10;  // statC
static const double me_cgs      = 9.10938356e-28;   // g
static const double m_p         = 1.6738e-24;       // g

// ============================================================
// 21cm and Lyman Constants
// ============================================================
static const double LAMBDA_21cm = 21.106;           // cm
static const double LAMBDA_1216 = 1215.67e-8;       // cm
static const double A_10        = 2.85e-15;         // s^-1

// ============================================================
// Cosmology
// ============================================================
static const double h               = 0.6774;
static const double H_0             = 100.0 * 0.6774;               // km/s/Mpc
static const double H_0_cgs         = 100.0 * 0.6774 * 3.241e-20;   // cm^-1

static const double OMEGA_b         = 0.0486;
static const double OMEGA_m         = 0.3089;
static const double OMEGA_Lambda    = 0.6911;

// ============================================================
// Hydrogen & Baryon Fractions
// ============================================================
static const double HYDROGEN_MASSFRAC = 0.76;
static const double X_H               = 0.76;
static const double y_frac            = (1.0 - 0.76) / (4.0 * 0.76);

// ============================================================
// Units
// ============================================================
static const double M_SOL   = 1.988e33;     // grams
static const double YR      = 31556736.0;   // seconds
static const double yr_sec  = 3.154e7;      // seconds
static const double msol_cgs = 1.988e33;

// ============================================================
// Derived Quantity: T_star (K)
// ============================================================
static const double T_star =
    (6.62607015e-27 * 2.99792458e10) /
    (1.381e-16 * 21.106);

#endif