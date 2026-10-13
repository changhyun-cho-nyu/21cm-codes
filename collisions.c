#include <math.h>
#include <stdio.h>
#include "collisions.h"

// ============================================================
// Collisional de-excitation rate tables (precomputed once)
// ============================================================

// --- H–H collisions ---
static const int N_HH = 27;

static const double T_HH[27] = {
    1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
    200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000
};

static const double k_HH[27] = {
    1.38e-13, 1.43e-13, 2.71e-13, 6.60e-13, 1.47e-12, 2.88e-12,
    9.10e-12, 1.78e-11, 2.73e-11, 3.67e-11, 5.38e-11, 6.86e-11,
    8.14e-11, 9.25e-11, 1.02e-10, 1.11e-10, 1.19e-10, 1.75e-10,
    2.09e-10, 2.56e-10, 2.91e-10, 3.31e-10, 4.27e-10, 4.97e-10,
    6.03e-10, 6.87e-10, 7.87e-10
};

// --- e–H collisions ---
static const int N_eH = 17;

static const double T_eH[17] = {
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
    2000, 3000, 5000, 7000, 10000, 15000, 20000
};

static const double k_eH[17] = {
    2.39e-10, 3.37e-10, 5.30e-10, 7.46e-10, 1.05e-9, 1.63e-9,
    2.26e-9, 3.11e-9, 4.59e-9, 5.92e-9, 7.15e-9, 7.71e-9,
    8.17e-9, 8.32e-9, 8.37e-9, 8.29e-9, 8.11e-9
};

// --- p–H collisions ---
static const int N_pH = 17;

static const double T_pH[17] = {
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
    2000, 3000, 5000, 7000, 10000, 15000, 20000
};

static const double k_pH[17] = {
    0.40e-9, 0.45e-9, 0.430e-9, 0.369e-9, 0.317e-9, 0.3047e-9,
    0.3379e-9, 0.4043e-9, 0.5471e-9, 0.7051e-9, 0.9167e-9,
    1.070e-9, 1.301e-9, 1.480e-9, 1.695e-9, 1.975e-9, 2.201e-9
};

// ============================================================
// Precomputed log-space tables (populated once at first call)
// ============================================================

static double log_T_HH[27], log_k_HH[27];
static double log_T_eH[17], log_k_eH[17];
static double log_T_pH[17], log_k_pH[17];
static int tables_ready = 0;

static void init_tables(void)
{
    for (int i = 0; i < N_HH; i++) {
        log_T_HH[i] = log10(T_HH[i]);
        log_k_HH[i] = log10(k_HH[i]);
    }
    for (int i = 0; i < N_eH; i++) {
        log_T_eH[i] = log10(T_eH[i]);
        log_k_eH[i] = log10(k_eH[i]);
    }
    for (int i = 0; i < N_pH; i++) {
        log_T_pH[i] = log10(T_pH[i]);
        log_k_pH[i] = log10(k_pH[i]);
    }
    tables_ready = 1;
}

// ============================================================
// Interpolation — operates entirely on precomputed log arrays
// ============================================================

static double log_interp_pre(double log_x,
                              const double *log_x_table,
                              const double *log_y_table,
                              int n)
{
    if (log_x <= log_x_table[0])   return log_y_table[0];
    if (log_x >= log_x_table[n-1]) return log_y_table[n-1];

    int i = 0;
    while (i < n-1 && log_x_table[i+1] < log_x)
        i++;

    double t = (log_x - log_x_table[i]) / (log_x_table[i+1] - log_x_table[i]);
    return log_y_table[i] + t * (log_y_table[i+1] - log_y_table[i]);
}

// ============================================================
// Public rate functions
// ============================================================

double kappa_HH(double T_K)
{
    if (!tables_ready) init_tables();
    return pow(10.0, log_interp_pre(log10(T_K), log_T_HH, log_k_HH, N_HH));
}

double kappa_eH(double T_K)
{
    if (!tables_ready) init_tables();
    return pow(10.0, log_interp_pre(log10(T_K), log_T_eH, log_k_eH, N_eH));
}

double kappa_pH(double T_K)
{
    if (!tables_ready) init_tables();
    return pow(10.0, log_interp_pre(log10(T_K), log_T_pH, log_k_pH, N_pH));
}
