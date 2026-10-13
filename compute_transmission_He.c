/*
 * compute_transmission_He.c
 * =========================
 * Transmission of He II Lyman-band photons (40.8-54.4 eV) through the IGM,
 * for the shell integrals of compute_Jalpha_He.c and compute_Jalpha_AGN.c (He).
 *
 * A photon reaching the observer cell x from comoving distance R is absorbed
 * on the way by H I and He I photoionization (He II does not absorb below
 * 54.4 eV). For each origin x (a coarse grid) and N_DIR directions n, rays are
 * marched through the absorber densities of make_opacity_fields.py:
 *     dtau = [n_HI sigma_HI(E) + n_HeI sigma_HeI(E)] (1+z)^2 ds_comoving
 * (proper densities and proper path length), where along the ray
 *   - z(r) follows the light cone: r = comoving distance from z_obs,
 *   - E(r) = E_n (1+z(r))/(1+z_obs): the photon is bluer the farther back,
 *   - the absorber field is taken from the snapshot nearest z(r), so a photon
 *     emitted at z' crosses the gas as it was at the time it passed.
 * The output is the direction-averaged transmission
 *     T(x, R) = < exp(-tau(x, R n)) >_n
 * on N_R log-spaced radii, for two energy groups:
 *     T_n2 : the Ly-beta -> Ly-alpha window (E_n = 40.8 eV, f_rec = 1),
 *     T_n4 : the n >= 4 windows, E_n = mean weighted by f_rec x redshift window (~52.5 eV).
 * Cross sections: Verner et al. (1996) fits for H I and He I.
 *
 * Optional unresolved absorbers (Lyman-limit systems below the render
 * resolution), off by default (LLS_A = 0):
 *     lambda_912,extra(z) = LLS_A [(1+z)/5]^LLS_SLOPE  proper Mpc,  z < LLS_ZMAX,
 * converted to the group energy with the effective opacity of Poisson clouds,
 *     dtau_eff/dl ~ Int f(N) (1 - exp(-N sigma)) dN,
 *     lambda_E = lambda_912 * I(sigma_912) / I(sigma_E),
 * for a broken power-law column-density distribution f(N) ~ N^-b with
 * b = LLS_FN_B1 below 10^LLS_FN_LOGN1, LLS_FN_B2 up to 10^LLS_FN_LOGN2 and
 * LLS_FN_B3 above (defaults 1.5 / 0.8 / 1.5 with breaks at 10^17.5 and 10^19 cm^-2;
 * Prochaska, O'Meara & Worseck 2010 measure b ~ 0.8 for the Lyman-limit range,
 * which is the range that sets the opacity at 40-54 eV). The integral runs over
 * 10^11-10^22 cm^-2. A single power law (the old -DLLS_BETA=b) is the special
 * case b1 = b2 = b3; for 1 < b < 2 it gives (sigma_912/sigma_E)^(b-1).
 * The term is applied to every ray as exp(-Int dl_proper/lambda_E). Use the
 * mfp912 mode to decide whether it is needed.
 *
 * Mode mfp912: the 912 A mean free path of the simulation itself at the target
 * snapshot, lambda_912 = < Int exp(-tau_912(r)) dl_proper > over rays from
 * ionized origins, to compare with observations (e.g. Worseck+14, Becker+21).
 *
 * Usage
 * -----
 *   ./compute_transmission_He <snap> <opacity_dir> <output.hdf5>
 *   ./compute_transmission_He <snap> <opacity_dir> mfp912
 * opacity_dir holds opacity_NNN.hdf5 from make_opacity_fields.py.
 *
 * Compile
 * -------
 *   gcc -O3 -fopenmp compute_transmission_He.c -o compute_transmission_He -lhdf5 -lm
 */
#include <hdf5.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef N_SNAPS
#define N_SNAPS    709
#endif
#ifndef N_ORIG
#define N_ORIG     160       /* origins per dimension                         */
#endif
#ifndef N_DIR
#define N_DIR      192       /* directions per origin (Fibonacci sphere)      */
#endif
#ifndef N_R
#define N_R        48        /* log-spaced output radii                       */
#endif
#ifndef STEP_FRAC
#define STEP_FRAC  0.5       /* ray step in units of the absorber-grid cell   */
#endif
#ifndef TAU_STOP
#define TAU_STOP   30.0      /* stop a ray once even the n>=4 group has tau > */
#endif
#ifndef MAX_FIELDS
#define MAX_FIELDS 24        /* absorber snapshots kept in memory             */
#endif
#ifndef LLS_A
#define LLS_A      0.0       /* extra lambda_912 at (1+z)/5 = 1 [proper Mpc]; 0 = off */
#endif
#ifndef LLS_SLOPE
#define LLS_SLOPE  (-5.4)
#endif
#ifdef LLS_BETA             /* single power law, kept for comparison */
#define LLS_FN_B1  LLS_BETA
#define LLS_FN_B2  LLS_BETA
#define LLS_FN_B3  LLS_BETA
#endif
#ifndef LLS_FN_B1
#define LLS_FN_B1  1.5       /* f(N) slope below 10^LLS_FN_LOGN1 (Ly-alpha forest)        */
#endif
#ifndef LLS_FN_B2
#define LLS_FN_B2  0.8       /* between the breaks (Lyman-limit systems, Prochaska+10)    */
#endif
#ifndef LLS_FN_B3
#define LLS_FN_B3  1.5       /* above 10^LLS_FN_LOGN2 (super-LLS / sub-DLA)               */
#endif
#ifndef LLS_FN_LOGN1
#define LLS_FN_LOGN1 17.5
#endif
#ifndef LLS_FN_LOGN2
#define LLS_FN_LOGN2 19.0
#endif
#ifndef LLS_ZMAX
#define LLS_ZMAX   5.5
#endif

#define C_CGS     2.99792458e10
#define CMPC_CGS  3.0857e24
#define OMEGA_M   0.30964144
#define OMEGA_L   0.6902672
#define OMEGA_B   0.04897
#define H_0_CGS   (0.6766 * 100.0 * 3.241e-20)
#define E_LL_HE   54.418     /* He II Lyman limit [eV] = h * 1.31583e16 Hz */
#define Y_HE      (0.24 / (4.0 * 0.76))
#define NH_MEAN_COM (0.76 * OMEGA_B * 3.0 * H_0_CGS * H_0_CGS / (8.0 * M_PI * 6.674e-8) / 1.67262192e-24)

static const double F_REC[31] = {0, 0, 1.0, 0.0, 0.2609, 0.3078, 0.3259, 0.3353, 0.3410, 0.3448, 0.3476,
    0.3496, 0.3512, 0.3524, 0.3535, 0.3543, 0.3550, 0.3556, 0.3561, 0.3565, 0.3569,
    0.3572, 0.3575, 0.3578, 0.3580, 0.3582, 0.3584, 0.3586, 0.3587, 0.3589, 0.3590};

/* Verner et al. (1996) photoionization cross sections [cm^2] */
static double verner(double E, double E0, double s0, double ya, double P, double yw, double y0, double y1) {
    double x = E / E0 - y0, y = sqrt(x * x + y1 * y1);
    return s0 * 1e-18 * ((x - 1) * (x - 1) + yw * yw) * pow(y, 0.5 * P - 5.5) * pow(1 + sqrt(y / ya), -P);
}
static double sigma_HI(double E)  { return E < 13.598 ? 0.0 : verner(E, 0.4298, 5.475e4, 32.88, 2.963, 0.0, 0.0, 0.0); }
static double sigma_HeI(double E) { return E < 24.587 ? 0.0 : verner(E, 13.61, 949.2, 1.469, 3.188, 2.039, 0.4434, 2.136); }

/* I(sigma) = Int f(N) (1 - exp(-N sigma)) dN for the broken power law, up to a constant:
 * proportional to the effective opacity of Poisson-distributed absorbers */
static double lls_I(double sigma) {
    const double l0 = 11.0, l1 = 22.0; const int n = 4400;           /* 0.0025 dex steps */
    const double dl = (l1 - l0) / n, ln10 = log(10.0);
    double sum = 0.0;
    for (int i = 0; i <= n; i++) {
        double l = l0 + i * dl, b, ln_f;
        if (l < LLS_FN_LOGN1)      { b = LLS_FN_B1; ln_f = -b * (l - LLS_FN_LOGN1) * ln10; }
        else if (l < LLS_FN_LOGN2) { b = LLS_FN_B2; ln_f = -b * (l - LLS_FN_LOGN1) * ln10; }
        else { ln_f = (-LLS_FN_B2 * (LLS_FN_LOGN2 - LLS_FN_LOGN1) - LLS_FN_B3 * (l - LLS_FN_LOGN2)) * ln10; }
        double N = pow(10.0, l);
        double g = exp(ln_f) * N * ln10 * (-expm1(-N * sigma));      /* f(N) dN/dlog10N (1 - e^-N sigma) */
        sum += (i == 0 || i == n ? 0.5 : 1.0) * g;
    }
    return sum * dl;
}
/* lambda_E / lambda_912 for unresolved absorbers, with sigma_E the cross section per H atom */
static double lls_ratio(double sigma_E) { return lls_I(sigma_HI(13.598)) / lls_I(sigma_E); }

static double H_cgs(double z) { return H_0_CGS * sqrt(OMEGA_M * pow(1 + z, 3) + OMEGA_L); }

/* ------------------------------------------------------------------ fields */
typedef struct { int snap; double z; float *nHI, *nHeI; } Field;
static int Nk = 0, FACTOR = 1, SF_MASK = -1;   /* sf_mask of the opacity files (-1 = not yet read) */
static double BOX = 0.0;

static int read_attr_d(hid_t f, const char *n, double *v) {
    if (H5Aexists(f, n) <= 0) return -1;
    hid_t a = H5Aopen(f, n, H5P_DEFAULT); H5Aread(a, H5T_NATIVE_DOUBLE, v); H5Aclose(a); return 0;
}

static float *read_field(hid_t f, const char *name, const char *path) {
    hid_t d = H5Dopen(f, name, H5P_DEFAULT);
    if (d < 0) { fprintf(stderr, "ERROR: no dataset %s in %s\n", name, path); exit(1); }
    hid_t s = H5Dget_space(d); hsize_t dims[3]; H5Sget_simple_extent_dims(s, dims, NULL); H5Sclose(s);
    if (Nk == 0) Nk = (int)dims[0];
    if ((int)dims[0] != Nk || dims[1] != dims[0] || dims[2] != dims[0]) {
        fprintf(stderr, "ERROR: %s in %s has an unexpected shape\n", name, path); exit(1);
    }
    float *a = malloc((size_t)Nk * Nk * Nk * sizeof(float));
    if (!a || H5Dread(d, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, a) < 0) {
        fprintf(stderr, "ERROR: cannot read %s from %s\n", name, path); exit(1);
    }
    H5Dclose(d);
    return a;
}

static void load_field(const char *dir, Field *F) {
    char path[2048];
    snprintf(path, sizeof path, "%s/opacity_%03d.hdf5", dir, F->snap);
    hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }
    double box, fac;
    if (read_attr_d(f, "BoxSize_cMpc", &box) || read_attr_d(f, "factor", &fac)) {
        fprintf(stderr, "ERROR: %s lacks BoxSize_cMpc/factor\n", path); exit(1);
    }
    BOX = box; FACTOR = (int)lround(fac);
    double sfm = 0.0;                                    /* files written before the option: unmasked */
    read_attr_d(f, "sf_mask", &sfm);
    if (SF_MASK >= 0 && (int)sfm != SF_MASK) {
        fprintf(stderr, "ERROR: %s has sf_mask = %d but other opacity files have %d: keep masked and "
                "unmasked fields in separate directories\n", path, (int)sfm, SF_MASK); exit(1);
    }
    SF_MASK = (int)sfm;
    F->nHI = read_field(f, "n_HI", path);
    F->nHeI = read_field(f, "n_HeI", path);
    H5Fclose(f);
}

/* ------------------------------------------------------------- light cone */
#define NZT 8192
static double zt[NZT], rt[NZT];                   /* comoving distance [Mpc] from z_obs */
static void build_lightcone(double z_obs, double z_far) {
    zt[0] = z_obs; rt[0] = 0.0;
    double dz = (z_far - z_obs) / (NZT - 1);
    for (int i = 1; i < NZT; i++) {
        zt[i] = z_obs + i * dz;
        double zm = zt[i] - 0.5 * dz;
        rt[i] = rt[i - 1] + C_CGS / H_cgs(zm) * dz / CMPC_CGS;
    }
}
static double z_of_r(double r) {
    if (r <= 0) return zt[0];
    if (r >= rt[NZT - 1]) return zt[NZT - 1];
    int lo = 0, hi = NZT - 1;
    while (hi - lo > 1) { int m = (lo + hi) / 2; if (rt[m] <= r) lo = m; else hi = m; }
    return zt[lo] + (zt[hi] - zt[lo]) * (r - rt[lo]) / (rt[hi] - rt[lo]);
}

static void fib_dirs(double (*d)[3]) {
    double ga = M_PI * (3.0 - sqrt(5.0));
    for (int i = 0; i < N_DIR; i++) {
        double zc = 1.0 - 2.0 * (i + 0.5) / N_DIR, s = sqrt(1.0 - zc * zc), ph = ga * i;
        d[i][0] = s * cos(ph); d[i][1] = s * sin(ph); d[i][2] = zc;
    }
}

static inline long cell_of(double x, double y, double z, double cell) {
    long i = (long)floor(x / cell), j = (long)floor(y / cell), k = (long)floor(z / cell);
    i %= Nk; if (i < 0) i += Nk;
    j %= Nk; if (j < 0) j += Nk;
    k %= Nk; if (k < 0) k += Nk;
    return (i * Nk + j) * Nk + k;
}

static void write_d(hid_t f, const char *n, double v) {
    hid_t s = H5Screate(H5S_SCALAR), a = H5Acreate(f, n, H5T_NATIVE_DOUBLE, s, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &v); H5Aclose(a); H5Sclose(s);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <snap> <opacity_dir> <output.hdf5 | mfp912>\n", argv[0]);
        return 1;
    }
    int snap = atoi(argv[1]);
    const char *dir = argv[2], *out = argv[3];
    int mfp_mode = strcmp(out, "mfp912") == 0;
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    /* available absorber snapshots and their redshifts */
    static double zs[N_SNAPS]; static int have[N_SNAPS];
    int n_have = 0;
    for (int i = 0; i < N_SNAPS; i++) {
        char path[2048]; snprintf(path, sizeof path, "%s/opacity_%03d.hdf5", dir, i);
        hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        have[i] = 0;
        if (f < 0) continue;
        if (read_attr_d(f, "Redshift", &zs[i]) == 0) { have[i] = 1; n_have++; }
        H5Fclose(f);
    }
    if (snap < 0 || snap >= N_SNAPS || !have[snap]) {
        fprintf(stderr, "ERROR: opacity_%03d.hdf5 not found in %s\n", snap, dir); return 1;
    }
    double z_obs = zs[snap], z_first = -1;
    for (int i = 0; i < N_SNAPS; i++) if (have[i] && zs[i] > z_first) z_first = zs[i];
    printf("=== compute_transmission_He: snap %03d  z_obs = %.4f  (%d absorber snapshots, z <= %.3f) ===\n",
           snap, z_obs, n_have, z_first);
    printf("N_ORIG = %d  N_DIR = %d  N_R = %d  step = %.2f cell  OMP threads = %d\n",
           N_ORIG, N_DIR, N_R, STEP_FRAC, omp_get_max_threads());

    double (*dirs)[3] = malloc(sizeof(double[3]) * N_DIR);
    fib_dirs(dirs);

    /* ---------------------------------------------------------- mfp912 mode */
    if (mfp_mode) {
        Field F = {snap, z_obs, NULL, NULL};
        load_field(dir, &F);
        double cell = BOX / Nk, ds = STEP_FRAC * cell, opz = 1.0 + z_obs;
        double w = 6.3e-18 * opz * opz * ds * CMPC_CGS;          /* sigma_912 (1+z)^2 ds */
        double dlp = ds / opz;                                    /* proper Mpc per step  */
        long nsteps = (long)(BOX / ds);                           /* rays up to one box length */
        double thr = 1e-2 * NH_MEAN_COM;                          /* ionized origin: n_HI < 1% of mean n_H */
        double sum = 0.0; long nray = 0, ncap = 0;
        #pragma omp parallel for collapse(3) reduction(+:sum,nray,ncap) schedule(dynamic, 4)
        for (int a = 0; a < N_ORIG; a++) for (int b = 0; b < N_ORIG; b++) for (int c = 0; c < N_ORIG; c++) {
            double ox = (a + 0.5) * BOX / N_ORIG, oy = (b + 0.5) * BOX / N_ORIG, oz = (c + 0.5) * BOX / N_ORIG;
            if (F.nHI[cell_of(ox, oy, oz, cell)] >= thr) continue;
            for (int d = 0; d < N_DIR; d++) {
                double tau = 0.0, lam = 0.0; long j;
                for (j = 0; j < nsteps; j++) {
                    double r = (j + 0.5) * ds;
                    double t = tau + 0.5 * F.nHI[cell_of(ox + r * dirs[d][0], oy + r * dirs[d][1], oz + r * dirs[d][2], cell)] * w;
                    lam += exp(-t) * dlp;
                    tau = 2 * t - tau;
                    if (tau > TAU_STOP) break;
                }
                if (j == nsteps) ncap++;
                sum += lam; nray++;
            }
        }
        if (nray == 0) { printf("no ionized origins at this snapshot\n"); return 0; }
        printf("lambda_912 (simulation, z = %.3f) = %.4g proper Mpc  from %ld rays; %.2f%% of rays reached one box "
               "length (%.0f cMpc) with tau < %g, so this is then a lower limit; star-forming cells %s\n",
               z_obs, sum / nray, nray, 100.0 * ncap / nray, BOX, TAU_STOP, SF_MASK > 0 ? "masked" : "kept");
        return 0;
    }

    /* --------------------------------------------------- transmission mode */
    double z_max2 = (1.0 + z_obs) * (1.0 - 1.0 / 9.0) / (1.0 - 1.0 / 4.0) - 1.0;   /* Ly-beta horizon */
    if (z_max2 > z_first) z_max2 = z_first;
    double E2 = E_LL_HE * 0.75, E4 = 0.0, wsum = 0.0;
    for (int n = 4; n <= 30; n++) {          /* weight: f_rec x the line's redshift window (its share of J) */
        double win = (1.0 - 1.0 / ((n + 1.0) * (n + 1.0))) / (1.0 - 1.0 / ((double)n * n)) - 1.0;
        E4 += F_REC[n] * win * E_LL_HE * (1.0 - 1.0 / (n * n)); wsum += F_REC[n] * win;
    }
    E4 /= wsum;
    build_lightcone(z_obs, z_max2 + 1e-6);
    double R_max = rt[NZT - 1];

    /* absorber snapshots spanning [z_obs, z_max2], at most MAX_FIELDS */
    int cand[N_SNAPS], nc = 0, beyond = -1;
    for (int i = 0; i < N_SNAPS; i++) {
        if (!have[i]) continue;
        if (zs[i] >= z_obs - 1e-9 && zs[i] <= z_max2 + 1e-9) cand[nc++] = i;
        else if (zs[i] > z_max2 + 1e-9 && (beyond < 0 || zs[i] < zs[beyond])) beyond = i;  /* nearest beyond the far end */
    }
    if (beyond >= 0) cand[nc++] = beyond;
    for (int a = 1; a < nc; a++) {                                    /* order by redshift, high z first */
        int t = cand[a], b = a - 1;
        while (b >= 0 && zs[cand[b]] < zs[t]) { cand[b + 1] = cand[b]; b--; }
        cand[b + 1] = t;
    }
    int nf = nc < MAX_FIELDS ? nc : MAX_FIELDS;
    Field *F = calloc(nf, sizeof(Field));
    for (int k = 0; k < nf; k++) {
        int idx = (nf == 1) ? 0 : (int)lround((double)k * (nc - 1) / (nf - 1));
        F[k].snap = cand[idx]; F[k].z = zs[cand[idx]];
        load_field(dir, &F[k]);
    }
    double cell = BOX / Nk, ds = STEP_FRAC * cell, R_min = BOX / (Nk * FACTOR);
    long nsteps = (long)ceil(R_max / ds) + 1;
    printf("absorber grid %d^3 (%.3f cMpc), %d fields z = %.3f .. %.3f; E_n2 = %.2f eV, E_n4 = %.2f eV\n",
           Nk, cell, nf, F[0].z, F[nf - 1].z, E2, E4);
    printf("R = %.3f .. %.1f cMpc (z' up to %.3f), %ld steps of %.3f cMpc; star-forming cells %s\n",
           R_min, R_max, z_max2, nsteps, ds, SF_MASK > 0 ? "MASKED (transparent)" : "kept");
    printf("LLS term: %s; f(N) slopes %.2f / %.2f / %.2f (breaks 10^%.2f, 10^%.2f): lambda_E/lambda_912 = "
           "%.3f (n=2, at the target), %.3f (n>=4)\n", LLS_A > 0 ? "ON" : "off", LLS_FN_B1, LLS_FN_B2, LLS_FN_B3,
           LLS_FN_LOGN1, LLS_FN_LOGN2, lls_ratio(sigma_HI(E2) + Y_HE * sigma_HeI(E2)),
           lls_ratio(sigma_HI(E4) + Y_HE * sigma_HeI(E4)));

    /* per-step weights (depend only on r) */
    double *w2H = malloc(nsteps * sizeof(double)), *w2He = malloc(nsteps * sizeof(double));
    double *w4H = malloc(nsteps * sizeof(double)), *w4He = malloc(nsteps * sizeof(double));
    double *lls2 = calloc(nsteps + 1, sizeof(double)), *lls4 = calloc(nsteps + 1, sizeof(double));
    int *fid = malloc(nsteps * sizeof(int));
    for (long j = 0; j < nsteps; j++) {
        double zj = z_of_r((j + 0.5) * ds), opz = 1.0 + zj, sc = opz / (1.0 + z_obs);
        double fac = opz * opz * ds * CMPC_CGS;
        w2H[j] = sigma_HI(E2 * sc) * fac;  w2He[j] = sigma_HeI(E2 * sc) * fac;
        w4H[j] = sigma_HI(E4 * sc) * fac;  w4He[j] = sigma_HeI(E4 * sc) * fac;
        int best = 0;
        for (int k = 1; k < nf; k++) if (fabs(F[k].z - zj) < fabs(F[best].z - zj)) best = k;
        fid[j] = best;
        double d2 = 0, d4 = 0;
        if (LLS_A > 0 && zj < LLS_ZMAX) {
            double l912 = LLS_A * pow(opz / 5.0, LLS_SLOPE);
            double l2 = l912 * lls_ratio(sigma_HI(E2 * sc) + Y_HE * sigma_HeI(E2 * sc));
            double l4 = l912 * lls_ratio(sigma_HI(E4 * sc) + Y_HE * sigma_HeI(E4 * sc));
            d2 = ds / opz / l2; d4 = ds / opz / l4;                    /* proper path / lambda */
        }
        lls2[j + 1] = lls2[j] + d2; lls4[j + 1] = lls4[j] + d4;
    }

    double Rk[N_R];
    for (int k = 0; k < N_R; k++) Rk[k] = R_min * pow(R_max / R_min, (double)k / (N_R - 1));
    long No3 = (long)N_ORIG * N_ORIG * N_ORIG;
    float *T2 = malloc((size_t)N_R * No3 * sizeof(float)), *T4 = malloc((size_t)N_R * No3 * sizeof(float));
    if (!T2 || !T4) { fprintf(stderr, "ERROR: output malloc failed\n"); return 1; }

    #pragma omp parallel for schedule(dynamic, 64)
    for (long o = 0; o < No3; o++) {
        int a = (int)(o / ((long)N_ORIG * N_ORIG)), b = (int)((o / N_ORIG) % N_ORIG), c = (int)(o % N_ORIG);
        double ox = (a + 0.5) * BOX / N_ORIG, oy = (b + 0.5) * BOX / N_ORIG, oz = (c + 0.5) * BOX / N_ORIG;
        double s2[N_R], s4[N_R];
        memset(s2, 0, sizeof s2); memset(s4, 0, sizeof s4);
        for (int d = 0; d < N_DIR; d++) {
            double t2 = 0, t4 = 0, p2 = 0, p4 = 0;
            int k = 0;
            for (long j = 0; j < nsteps && k < N_R; j++) {
                double r = (j + 0.5) * ds;
                long ci = cell_of(ox + r * dirs[d][0], oy + r * dirs[d][1], oz + r * dirs[d][2], cell);
                const Field *f = &F[fid[j]];
                double nH = f->nHI[ci], nHe = f->nHeI[ci];
                p2 = t2; p4 = t4;
                t2 += nH * w2H[j] + nHe * w2He[j];
                t4 += nH * w4H[j] + nHe * w4He[j];
                double r1 = (j + 1) * ds;
                while (k < N_R && Rk[k] <= r1) {
                    double q = (Rk[k] - j * ds) / ds;
                    s2[k] += exp(-(p2 + q * (t2 - p2)));
                    s4[k] += exp(-(p4 + q * (t4 - p4)));
                    k++;
                }
                if (t4 > TAU_STOP) break;                  /* the rest of this ray contributes 0 */
            }
        }
        for (int k = 0; k < N_R; k++) {
            double jj = Rk[k] / ds; long j0 = (long)jj; if (j0 >= nsteps) j0 = nsteps - 1;
            double q = jj - j0;
            double L2 = lls2[j0] + q * (lls2[j0 + 1] - lls2[j0]), L4 = lls4[j0] + q * (lls4[j0 + 1] - lls4[j0]);
            T2[(size_t)k * No3 + o] = (float)(s2[k] / N_DIR * exp(-L2));
            T4[(size_t)k * No3 + o] = (float)(s4[k] / N_DIR * exp(-L4));
        }
    }

    double m2[N_R], m4[N_R];
    for (int k = 0; k < N_R; k++) {
        double a2 = 0, a4 = 0;
        for (long o = 0; o < No3; o++) { a2 += T2[(size_t)k * No3 + o]; a4 += T4[(size_t)k * No3 + o]; }
        m2[k] = a2 / No3; m4[k] = a4 / No3;
    }
    printf("box-mean transmission:\n      R [cMpc]     T_n2      T_n4\n");
    for (int k = 0; k < N_R; k += (N_R > 12 ? N_R / 12 : 1))
        printf("  %12.3f  %8.4f  %8.4f\n", Rk[k], m2[k], m4[k]);
    printf("  %12.3f  %8.4f  %8.4f\n", Rk[N_R - 1], m2[N_R - 1], m4[N_R - 1]);

    hid_t fo = H5Fcreate(out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fo < 0) { fprintf(stderr, "ERROR: cannot create %s\n", out); return 1; }
    hsize_t dims[4] = {N_R, N_ORIG, N_ORIG, N_ORIG}, ch[4] = {1, N_ORIG, N_ORIG, N_ORIG}, nr = N_R;
    hid_t sp = H5Screate_simple(4, dims, NULL), pl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(pl, 4, ch); H5Pset_deflate(pl, 4);
    hid_t d2 = H5Dcreate(fo, "T_n2", H5T_NATIVE_FLOAT, sp, H5P_DEFAULT, pl, H5P_DEFAULT);
    H5Dwrite(d2, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, T2); H5Dclose(d2);
    hid_t d4 = H5Dcreate(fo, "T_n4", H5T_NATIVE_FLOAT, sp, H5P_DEFAULT, pl, H5P_DEFAULT);
    H5Dwrite(d4, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, T4); H5Dclose(d4);
    H5Pclose(pl); H5Sclose(sp);
    hid_t s1 = H5Screate_simple(1, &nr, NULL);
    hid_t dr = H5Dcreate(fo, "R_cMpc", H5T_NATIVE_DOUBLE, s1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dr, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, Rk); H5Dclose(dr);
    hid_t dm2 = H5Dcreate(fo, "mean_T_n2", H5T_NATIVE_DOUBLE, s1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dm2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, m2); H5Dclose(dm2);
    hid_t dm4 = H5Dcreate(fo, "mean_T_n4", H5T_NATIVE_DOUBLE, s1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dm4, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, m4); H5Dclose(dm4);
    H5Sclose(s1);
    write_d(fo, "Redshift", z_obs);   write_d(fo, "snap", snap);
    write_d(fo, "BoxSize_cMpc", BOX); write_d(fo, "R_min_cMpc", R_min);
    write_d(fo, "E_n2_eV", E2);       write_d(fo, "E_n4_eV", E4);
    write_d(fo, "N_DIR", N_DIR);      write_d(fo, "LLS_A", LLS_A);
    write_d(fo, "LLS_SLOPE", LLS_SLOPE); write_d(fo, "LLS_ZMAX", LLS_ZMAX);
    write_d(fo, "LLS_FN_B1", LLS_FN_B1); write_d(fo, "LLS_FN_B2", LLS_FN_B2); write_d(fo, "LLS_FN_B3", LLS_FN_B3);
    write_d(fo, "LLS_FN_LOGN1", LLS_FN_LOGN1); write_d(fo, "LLS_FN_LOGN2", LLS_FN_LOGN2);
    write_d(fo, "sf_mask", SF_MASK > 0 ? 1.0 : 0.0);
    H5Fclose(fo);
    printf("wrote %s\n", out);
    return 0;
}
