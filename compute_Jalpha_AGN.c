/* =========================================================================
 * compute_Jalpha_AGN.c
 *
 * Spatially resolved Lyman-alpha background from AGN bolometric luminosity.
 *
 * Source field : AGN bolometric luminosity  [code units / voxel]
 * Spectral weight:
 *   eps_agn(nu) = (L_nu(nu) / L_bol) / (h*nu)   [ph / Hz / erg]
 *
 * Formula (identical structure to stellar code):
 *   J_a(x,z) = c(1+z)^2/4pi  Sum_n  f_rec(n)
 *              Int dz'/H(z')  eps_agn(nu'_n)  rho_L(x',z')
 *
 * where rho_L = box[i] * LUM_UNIT   [erg/s/cm^3]
 *       LUM_UNIT = 6.4478e36 / cell_cm^3   [erg/s/cm^3 per code unit]
 *
 * IMPORTANT (normalization):
 *   The SED file MUST span the full bolometric range (IR to X-ray), because
 *   L_bol is computed internally as Int L_nu dnu over the file's range, and
 *   LuminosityAGN is a *bolometric* luminosity. A band-truncated SED
 *   (e.g. only the Lyman window, renormalized to L_bol=1) silently inflates
 *   eps_agn by 1/f_band -- this was the ~27x (H) / ~110x (He) over-estimate.
 *   A guard in load_agn_sed() now aborts on a suspiciously narrow SED.
 *   Use ONE full-range SED file for both H and He; the species argument
 *   selects the right frequencies via NU_LL_SPECIES in nu_emitted().
 *
 * Key difference from stellar code:
 *   - No age integration (SED is instantaneous, not integrated over lifetime)
 *   - No /MP_CGS  (luminosity is already in erg/s; no mass->baryon conversion)
 *   - LUM_UNIT = 6.4478e36/cell_cm^3   (vs SFR_UNIT = Msun/yr -> g/s per cell)
 *
 * Dimensional check:
 *   [sr^-1] x [ph/Hz/erg] x [cm/s / s^-1] x [erg/s/cm^3] = [ph/s/Hz/cm^2/sr] OK
 *
 * Usage:
 *   mpirun -n NRANKS ./compute_Jalpha_AGN \
 *       <snap> <agn_dir> <output.hdf5> <sed.hdf5> \
 *       [agn_dir2 | none]  [dataset_name]  [species:H|He]
 *
 *   agn_dir2     : second directory for snaps split at z~4.75 ("none" to skip)
 *   dataset_name : HDF5 dataset key inside the AGN files
 *                  (default: "LuminosityAGN")
 *   species      : "H" (default) or "He"
 *
 *   AGN files assumed: {agn_dir}/{dataset}_{snap:03d}.hdf5
 *
 * SED file (HDF5) must contain:
 *   "freq"  [Hz]       shape (Nfreq,)  increasing order, FULL bolometric range
 *   "L_nu"  [erg/s/Hz] shape (Nfreq,)  any normalisation (renormalised internally)
 *
 * Compile (change TARGET and SRC in the existing Makefile):
 *   mpicc -O3 -march=native -fopenmp compute_Jalpha_AGN.c \
 *       -o compute_Jalpha_AGN \
 *       -I$(FFTW_INC) -I$(HDF5_INC) \
 *       -L$(FFTW_LIB) -L$(HDF5_LIB) \
 *       -lfftw3f_omp -lfftw3f -lhdf5 -lm -fopenmp
 * =========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>
#include <fftw3.h>
#include <mpi.h>
#include <omp.h>

/* ---- Physical constants (CGS) ------------------------------------------ */
#define C_CGS      2.9979e10      /* cm/s                  */
#define CMPC_CGS   3.0857e24      /* cm/Mpc                */
/* Lyman limit frequencies (set NU_LL_SPECIES in main from <species> arg):
 *   H    Lyman limit = 13.6 eV  -> 3.289e15  Hz
 *   He II Lyman limit = 54.4 eV -> 1.3156e16 Hz  (= 4 x H limit) */
#define NU_LL_H    3.28984e15     /* Hz, H    Lyman limit */
#define NU_LL_He   1.31594e16     /* Hz, He II Lyman limit */
static double NU_LL_SPECIES = 0.0;
#define PLANCK     6.62607e-27    /* erg.s                 */
#define M_PI_      3.14159265358979323846

/* ---- THESAN-XL cosmology (Planck 2015) ---------------------------------- */
#define OMEGA_M    0.30964144
#define OMEGA_L    0.6902672
#define H_0_CGS   (0.6766 * 100.0 * 3.241e-20)   /* s^-1 */

/* ---- Lyman series / numerical parameters -------------------------------- */
#define N_MAX      30     /* max Lyman series index (converged by n=23)  */
#define N_SHELLS   400    /* log-spaced radial shells (plus one inner shell)
                           * -- more than the stellar 200 because the AGN
                           * field is very sparse (bright point sources) */
#ifndef N_SNAPS
#define N_SNAPS    709    /* total THESAN-XL snapshots                   */
#endif
#define N_OMP      4      /* OpenMP threads per MPI rank                 */

/* ---- Globals set by load_snap_metadata ---------------------------------- */
static int    N_PIX    = 640;
static double BOX_CMPC = 500.0;
static double LUM_UNIT = 0.0;   /* erg/s/cm^3 per code unit
                                  * = 6.4478e36 / cell_cm^3
                                  * converts box[i] [code/vox] -> [erg/s/cm^3]
                                  * stored as double -- no float underflow risk */

static double snap_redshifts[N_SNAPS];

/* =========================================================================
 * f_rec(n): recycling fractions -- Pritchard & Furlanetto 2006, Table 1.
 * Hydrogenic cascade branching ratios are Z-independent, so the SAME table
 * applies to He II (only the line frequencies scale by Z^2=4).
 * ========================================================================= */
static const double F_REC_TABLE[31] = {
    0.0, 0.0,
    1.0000, 0.0000,
    0.2609, 0.3078, 0.3259, 0.3353, 0.3410, 0.3448, 0.3476,
    0.3496, 0.3512, 0.3524, 0.3535, 0.3543, 0.3550, 0.3556,
    0.3561, 0.3565, 0.3569, 0.3572, 0.3575, 0.3578, 0.3580,
    0.3582, 0.3584, 0.3586, 0.3587, 0.3589, 0.3590
};

static double f_rec(int n) {
    if (n >= 2 && n <= 30) return F_REC_TABLE[n];
    return 0.359;
}

/* =========================================================================
 * Cosmology -- identical to stellar code.
 * ========================================================================= */
static double H_cgm(double z) {
    double a = 1.0/(1.0+z);
    return H_0_CGS * sqrt(OMEGA_M/(a*a*a) + OMEGA_L);
}

static double comoving_Mpc(double z1, double z2, double dummy) {
    (void)dummy;
    int NZ = 1000;
    double dz = (z2-z1)/NZ, s = 0.0;
    for (int i = 0; i < NZ; i++)
        s += (C_CGS / H_cgm(z1 + (i+0.5)*dz)) * dz;
    return s / CMPC_CGS;
}

static double R_to_z(double z_obs, double z_max, double R_mpc) {
    double za = z_obs, zb = z_max;
    for (int i = 0; i < 60; i++) {
        double zm = 0.5*(za+zb);
        if (comoving_Mpc(z_obs, zm, 0) < R_mpc) za = zm; else zb = zm;
    }
    return 0.5*(za+zb);
}

/* =========================================================================
 * Lyman series helpers.
 * z_max_lyman() is species-independent (frequency ratios only).
 * nu_emitted() uses NU_LL_SPECIES -> selects H or He II band.
 * ========================================================================= */
static double z_max_lyman(double z_obs, int n) {
    return (1.0+z_obs)*(1.0-pow(n+1.0,-2.0))/(1.0-pow((double)n,-2.0))-1.0;
}
static double nu_emitted(double z_emit, double z_obs, int n) {
    return NU_LL_SPECIES*(1.0-1.0/((double)n*(double)n))*(1.0+z_emit)/(1.0+z_obs);
}

/* =========================================================================
 * Spectral interpolation table.
 * N_nu stores eps_agn(nu) = (L_nu/L_bol)/(h*nu) [ph/Hz/erg].
 * ========================================================================= */
typedef struct { double *freq; double *N_nu; int Nfreq; } SpecTable;

static double eps_b(const SpecTable *st, double nu) {
    int lo = 0, hi = st->Nfreq-1;
    while (hi-lo > 1) {
        int mid = (lo+hi)/2;
        if (st->freq[mid] < nu) lo = mid; else hi = mid;
    }
    if (lo >= st->Nfreq-1) return st->N_nu[st->Nfreq-1];
    double w = (nu-st->freq[lo])/(st->freq[hi]-st->freq[lo]);
    return st->N_nu[lo]*(1.0-w) + st->N_nu[hi]*w;
}

/* =========================================================================
 * HDF5 helpers.
 * ========================================================================= */
static double read_attr_double(hid_t gid, const char *name) {
    double v;
    hid_t a = H5Aopen(gid, name, H5P_DEFAULT);
    H5Aread(a, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(a);
    return v;
}
static int read_attr_int(hid_t gid, const char *name) {
    int v;
    hid_t a = H5Aopen(gid, name, H5P_DEFAULT);
    H5Aread(a, H5T_NATIVE_INT, &v);
    H5Aclose(a);
    return v;
}

/* =========================================================================
 * Snapshot metadata.
 * LUM_UNIT = 6.4478e36/cell_cm^3  (code unit -> erg/s, then per comoving cm^3).
 * cell_cm is COMOVING (boxsize * unitlen / hubble / N_PIX).
 * ========================================================================= */
static void build_snap_path(const char *dir, const char *prefix,
                             int snap, char *out) {
    snprintf(out, 2048, "%s/%s_%03d.hdf5", dir, prefix, snap);
}

static void load_snap_metadata(const char *agn_dir, const char *agn_dir2,
                                const char *prefix, int rank) {
    for (int i = 0; i < N_SNAPS; i++) snap_redshifts[i] = -1.0;

    int geometry_read = 0;
    for (int i = 0; i < N_SNAPS; i++) {
        char path[2048];
        build_snap_path(agn_dir, prefix, i, path);
        FILE *fp = fopen(path, "r");
        if (!fp && agn_dir2) {
            snprintf(path, 2048, "%s/%s_%03d.hdf5", agn_dir2, prefix, i);
            fp = fopen(path, "r");
        }
        if (!fp) continue;
        fclose(fp);

        hid_t fid = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        hid_t gid = H5Gopen(fid, "Header", H5P_DEFAULT);
        snap_redshifts[i] = read_attr_double(gid, "Redshift");

        if (!geometry_read) {
            double boxsize = read_attr_double(gid, "BoxSize");
            double hubble  = read_attr_double(gid, "HubbleParam");
            double unitlen = read_attr_double(gid, "UnitLength_in_cm");
            N_PIX    = read_attr_int(gid, "NumPixels");
            BOX_CMPC = boxsize / hubble / 1000.0;
            double box_cm  = boxsize * unitlen / hubble;   /* comoving cm */
            double cell_cm = box_cm / N_PIX;               /* comoving cm */
            /*
             * LUM_UNIT = 6.4478e36 / cell_cm^3
             * box[i] is in code units; x6.4478e36 -> erg/s; /cell_cm^3 -> erg/s/cm^3.
             * Kept as double -- no float underflow.
             */
            LUM_UNIT = 6.4478e36 / (cell_cm * cell_cm * cell_cm);
            geometry_read = 1;
            if (rank == 0)
                printf("  Box=%.1f cMpc  N=%d  cell_cm=%.4e  LUM_UNIT=%.4e erg/s/cm^3/codeunit\n",
                       BOX_CMPC, N_PIX, cell_cm, LUM_UNIT);
        }
        H5Gclose(gid); H5Fclose(fid);
    }

    int ns = 0; double zlo = 1e30, zhi = -1e30;
    for (int i = 0; i < N_SNAPS; i++) {
        if (snap_redshifts[i] < 0) continue;
        ns++;
        if (snap_redshifts[i] < zlo) zlo = snap_redshifts[i];
        if (snap_redshifts[i] > zhi) zhi = snap_redshifts[i];
    }
    if (rank == 0)
        printf("  Redshifts loaded: z=%.3f to z=%.3f  (%d snaps)\n",
               zlo, zhi, ns);
}

/* =========================================================================
 * AGN box reader.  Reads luminosity [code units/vox] from HDF5.
 * No unit conversion here -- LUM_UNIT folded into prefac as double.
 * ========================================================================= */
static void read_agn_box(const char *agn_dir, const char *agn_dir2,
                          const char *prefix, const char *dataset,
                          int snap, float *box) {
    char path[2048];
    build_snap_path(agn_dir, prefix, snap, path);

    FILE *chk = fopen(path, "r");
    if (!chk) {
        if (agn_dir2)
            snprintf(path, 2048, "%s/%s_%03d.hdf5", agn_dir2, prefix, snap);
        chk = fopen(path, "r");
        if (!chk) {
            fprintf(stderr, "ERROR: snap %03d not found in either directory\n", snap);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    fclose(chk);

    hid_t fid = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t did = H5Dopen(fid, dataset, H5P_DEFAULT);
    H5Dread(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, box);
    H5Dclose(did); H5Fclose(fid);

    /* Non-negativity floor only; no unit conversion (LUM_UNIT stays in prefac) */
    long N3 = (long)N_PIX * N_PIX * N_PIX;
    #pragma omp parallel for num_threads(N_OMP)
    for (long i = 0; i < N3; i++)
        if (box[i] < 0.0f) box[i] = 0.0f;
}

/* =========================================================================
 * Snapshot cache.
 * ========================================================================= */
typedef struct {
    int    snap_lo, snap_hi;
    long   N3;
    float *box_lo, *box_hi, *box_interp;
    const char *agn_dir, *agn_dir2;
    const char *prefix, *dataset;
} SnapCache;

static void find_bracket(double z_prime, int *slo, int *shi) {
    if (z_prime >= snap_redshifts[0])
        { *slo = 0; *shi = 0; return; }
    if (z_prime <= snap_redshifts[N_SNAPS-1])
        { *slo = N_SNAPS-1; *shi = N_SNAPS-1; return; }
    int lo = 0, hi = N_SNAPS-1;
    while (hi-lo > 1) {
        int mid = (lo+hi)/2;
        if (snap_redshifts[mid] >= z_prime) lo = mid; else hi = mid;
    }
    *slo = lo; *shi = hi;
}

static void cache_init(SnapCache *c, long N3,
                        const char *agn_dir, const char *agn_dir2,
                        const char *prefix,  const char *dataset) {
    c->snap_lo = c->snap_hi = -9999;
    c->N3       = N3;
    c->agn_dir  = agn_dir;  c->agn_dir2 = agn_dir2;
    c->prefix   = prefix;   c->dataset  = dataset;
    c->box_lo     = (float*)fftwf_malloc(N3 * sizeof(float));
    c->box_hi     = (float*)fftwf_malloc(N3 * sizeof(float));
    c->box_interp = (float*)fftwf_malloc(N3 * sizeof(float));
}
static void cache_free(SnapCache *c) {
    fftwf_free(c->box_lo); fftwf_free(c->box_hi); fftwf_free(c->box_interp);
}

static float *cache_get(SnapCache *c, double z_prime, int rank) {
    int slo, shi;
    find_bracket(z_prime, &slo, &shi);

    if (slo != c->snap_lo) {
        printf("  [rank %d cache] reading snap %03d  z=%.4f\n",
               rank, slo, snap_redshifts[slo]);
        read_agn_box(c->agn_dir, c->agn_dir2, c->prefix, c->dataset,
                     slo, c->box_lo);
        c->snap_lo = slo;
    }
    if (slo == shi) return c->box_lo;

    if (shi != c->snap_hi) {
        if (snap_redshifts[shi] < 0) {
            fprintf(stderr, "ERROR: snap %d missing\n", shi);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("  [rank %d cache] reading snap %03d  z=%.4f\n",
               rank, shi, snap_redshifts[shi]);
        read_agn_box(c->agn_dir, c->agn_dir2, c->prefix, c->dataset,
                     shi, c->box_hi);
        c->snap_hi = shi;
    }

    double z_lo = snap_redshifts[slo], z_hi = snap_redshifts[shi];
    float  w    = (float)((z_prime - z_lo) / (z_hi - z_lo));
    long   N3   = c->N3;
    #pragma omp parallel for num_threads(N_OMP)
    for (long i = 0; i < N3; i++) {
        float v = (1.0f-w)*c->box_lo[i] + w*c->box_hi[i];
        c->box_interp[i] = (v > 0.0f) ? v : 0.0f;
    }
    return c->box_interp;
}

/* =========================================================================
 * FFT spherical-shell averaging
 *
 * At fixed z_obs, the emission redshift z' maps to one comoving distance
 * R(z_obs, z'), so a slice [z'_lo, z'_hi] of the J_alpha integral collects
 * the emissivity on the spherical SHELL R_lo < r < R_hi, not inside the
 * full sphere of radius R. Averaging over the full sphere would weight
 * nearby sources 3/2 too heavily for the 1/(4 pi r^2) kernel.
 *
 * Shell window (volume-weighted average over R_lo < r < R_hi):
 *   W(k) = [R_hi^3 W_TH(k R_hi) - R_lo^3 W_TH(k R_lo)] / (R_hi^3 - R_lo^3)
 * Inner shell (R_lo = 0): the radial weight is constant per unit r, so the
 * exact window of the 1/(4 pi r^2) kernel is used instead:
 *   W(k) = Si(k R_hi) / (k R_hi)
 * Both have W(0) = 1, so the mean is preserved. Shell windows ring in real
 * space; individual shells may go negative and are NOT clipped here, since
 * clipping each shell biases the mean. Only the final J is clipped.
 * ========================================================================= */

/* Top-hat sphere window 3 j1(x)/x */
static double W_tophat(double x) {
    if (x < 1e-3) return 1.0 - x*x/10.0;
    return 3.0*(sin(x) - x*cos(x))/(x*x*x);
}

/* Si(x)/x by its power series. Used only for the inner shell, where
 * x <= sqrt(3)*pi*R_hi/dx <= 5.5; the series is accurate to ~1e-13 there. */
static double Si_over_x(double x) {
    double x2 = x*x, u = 1.0, s = 1.0;   /* u_n = (-1)^n x^2n / (2n+1)! */
    for (int n = 1; n < 60; n++) {
        u *= -x2 / ((2.0*n) * (2.0*n + 1.0));
        double t = u / (2.0*n + 1.0);
        s += t;
        if (fabs(t) < 1e-17*fabs(s)) break;
    }
    return s;
}

static void smooth_shell(const float *sfr_in, float *sfr_out,
                         int N, double box_mpc, double R_lo, double R_hi,
                         fftwf_plan plan_r2c, fftwf_plan plan_c2r,
                         fftwf_complex *work_k, float *work_r) {
    long   N3 = (long)N*N*N;
    int    Nk = N/2+1;
    double R3_lo = R_lo*R_lo*R_lo, R3_hi = R_hi*R_hi*R_hi;
    double k0 = 2.0*M_PI_/box_mpc;
    double norm = 1.0/(double)N3;

    /* W depends only on |k|^2 = k0^2 (fx^2+fy^2+fz^2): tabulate it once
     * over the integer m = fx^2+fy^2+fz^2 instead of per Fourier mode. */
    int    Nm  = 3*(N/2)*(N/2) + 1;
    float *Wm  = malloc(Nm * sizeof(float));
    if (!Wm) {
        fprintf(stderr, "ERROR: shell window table malloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    #pragma omp parallel for num_threads(N_OMP)
    for (int m = 0; m < Nm; m++) {
        double k = k0*sqrt((double)m);
        double W = (R_lo <= 0.0)
                 ? Si_over_x(k*R_hi)
                 : (R3_hi*W_tophat(k*R_hi) - R3_lo*W_tophat(k*R_lo)) / (R3_hi - R3_lo);
        Wm[m] = (float)(W*norm);
    }

    memcpy(work_r, sfr_in, N3*sizeof(float));
    fftwf_execute(plan_r2c);

    #pragma omp parallel for collapse(3) num_threads(N_OMP)
    for (int ix=0; ix<N; ix++)
    for (int iy=0; iy<N; iy++)
    for (int iz=0; iz<Nk; iz++) {
        int fx = (ix<=N/2) ? ix : ix-N;
        int fy = (iy<=N/2) ? iy : iy-N;
        float W  = Wm[fx*fx + fy*fy + iz*iz];
        long idx = ((long)ix*N+iy)*Nk+iz;
        work_k[idx][0] *= W;
        work_k[idx][1] *= W;
    }
    free(Wm);

    fftwf_execute(plan_c2r);
    memcpy(sfr_out, work_r, N3*sizeof(float));
}

/* =========================================================================
 * AGN SED loader.
 *
 * Loads freq [Hz] and L_nu [erg/s/Hz], then builds:
 *   eps_agn[i] = (L_nu[i] / L_bol) / (PLANCK * freq[i])   [ph/Hz/erg]
 * with L_bol = Int L_nu dnu over the FULL file range (trapezoidal).
 *
 * The absolute normalisation of L_nu cancels; only the RANGE matters.
 * The SED MUST span the full bolometric range -- a band-truncated file
 * gives too-small L_bol and inflates eps_agn. The guard below aborts on
 * a suspiciously narrow SED.
 * ========================================================================= */
static void load_agn_sed(const char *sed_file, SpecTable *st, int rank) {
    hid_t fid = H5Fopen(sed_file, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "ERROR: cannot open SED file %s\n", sed_file);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Read frequency array */
    hid_t  did_f = H5Dopen(fid, "freq", H5P_DEFAULT);
    hid_t  sid_f = H5Dget_space(did_f);
    hsize_t nfreq;
    H5Sget_simple_extent_dims(sid_f, &nfreq, NULL);
    H5Sclose(sid_f);

    double *freq_d = (double*)malloc(nfreq * sizeof(double));
    double *lnu_d  = (double*)malloc(nfreq * sizeof(double));
    H5Dread(did_f, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, freq_d);
    H5Dclose(did_f);

    /* Read L_nu */
    hid_t did_l = H5Dopen(fid, "L_nu", H5P_DEFAULT);
    H5Dread(did_l, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, lnu_d);
    H5Dclose(did_l);
    H5Fclose(fid);

    /* L_bol = Int L_nu dnu  (trapezoidal, freq must be increasing) */
    double L_bol = 0.0;
    for (hsize_t i = 0; i < nfreq-1; i++)
        L_bol += 0.5*(lnu_d[i]+lnu_d[i+1]) * (freq_d[i+1]-freq_d[i]);

    if (L_bol <= 0.0) {
        fprintf(stderr, "ERROR: AGN SED gives L_bol <= 0. "
                        "Check that freq is increasing and L_nu > 0.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* -------------------------------------------------------------------
     * GUARD: catch a band-truncated / pre-normalized SED.
     *
     * A real bolometric SED spans many decades (IR ~0.01 eV to X-ray ~MeV,
     * i.e. ~8 decades in frequency). A file covering only a Lyman window
     * (renormalized to L_bol ~ 1) spans < 1 decade and silently inflates
     * eps_agn by 1/f_band -- exactly the bug that produced J too large by
     * ~27x (H) and ~110x (He). Abort rather than write wrong results.
     * ------------------------------------------------------------------- */
    double decades = log10(freq_d[nfreq-1] / freq_d[0]);
    if (decades < 3.0) {
        if (rank == 0) {
            fprintf(stderr,
                "FATAL: SED spans only %.2f decades in frequency "
                "[%.3e, %.3e] Hz (L_bol=%.3e).\n"
                "       Expected a FULL bolometric SED (>~6 decades, IR to X-ray).\n"
                "       This file looks band-truncated/renormalized -- it would\n"
                "       inflate eps_agn by 1/f_band. Provide the full-range SED.\n",
                decades, freq_d[0], freq_d[nfreq-1], L_bol);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Check SED coverage of Lyman-series windows */
    double nu_lya = NU_LL_SPECIES * (1.0 - 1.0/4.0);   /* species Ly-alpha */
    if (rank == 0) {
        printf("  AGN SED: Nfreq=%llu  freq=[%.3e, %.3e] Hz  (%.2f decades)\n",
               (unsigned long long)nfreq, freq_d[0], freq_d[nfreq-1], decades);
        printf("  L_bol (full-range normalisation) = %.4e erg/s\n", L_bol);
        if (freq_d[0] > nu_lya)
            printf("  WARNING: SED lower edge %.3e Hz is above Ly-alpha %.3e Hz -- "
                   "n=2 shells near z_obs will use boundary eps_agn value\n",
                   freq_d[0], nu_lya);
        if (freq_d[nfreq-1] < NU_LL_SPECIES)
            printf("  WARNING: SED upper edge %.3e Hz is below Lyman limit "
                   "%.3e Hz -- high-n shells will use boundary eps_agn value\n",
                   freq_d[nfreq-1], NU_LL_SPECIES);
    }

    /* Build eps_agn table */
    st->Nfreq = (int)nfreq;
    st->freq  = (double*)malloc(nfreq * sizeof(double));
    st->N_nu  = (double*)malloc(nfreq * sizeof(double));
    for (hsize_t i = 0; i < nfreq; i++) {
        st->freq[i] = freq_d[i];
        /*
         * eps_agn(nu) = (L_nu / L_bol) / (h*nu)   [ph/Hz/erg]
         *
         * Dimensional check:
         *   [erg/s/Hz] / [erg/s] / [erg] = [1/(Hz.erg)] = [ph/Hz/erg]
         * Multiplied by rho_L [erg/s/cm^3]:
         *   [ph/Hz/erg] x [erg/s/cm^3] = [ph/s/Hz/cm^3]  <- emissivity OK
         */
        st->N_nu[i] = (lnu_d[i] / L_bol) / (PLANCK * freq_d[i]);
    }

    if (rank == 0) {
        double eps_lya = eps_b(st, nu_lya);
        double eps_ll  = eps_b(st, NU_LL_SPECIES);
        printf("  eps_agn at Ly-alpha (%.3e Hz) = %.4e ph/Hz/erg\n",
               nu_lya, eps_lya);
        printf("  eps_agn at Lyman limit       = %.4e ph/Hz/erg\n", eps_ll);
    }
    free(freq_d); free(lnu_d);
}

/* =========================================================================
 * HDF5 output.
 * ========================================================================= */
static void write_output(const char *path, const float *J,
                          int N, double z_obs, double box_mpc,
                          double J_mean, double J_scalar,
                          const char *dset_name) {
    hid_t fid = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[3] = {(hsize_t)N, (hsize_t)N, (hsize_t)N};
    hid_t sid = H5Screate_simple(3, dims, NULL);
    hid_t did = H5Dcreate(fid, dset_name, H5T_NATIVE_FLOAT, sid,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, J);
    H5Dclose(did); H5Sclose(sid);

    hid_t gid = H5Gcreate(fid, "Header",
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#define WA(n,v) { double _v=(v); \
    hid_t _s=H5Screate(H5S_SCALAR); \
    hid_t _a=H5Acreate(gid,n,H5T_NATIVE_DOUBLE,_s,H5P_DEFAULT,H5P_DEFAULT); \
    H5Awrite(_a,H5T_NATIVE_DOUBLE,&_v); H5Aclose(_a); H5Sclose(_s); }
    WA("Redshift",     z_obs)
    WA("BoxSize_cMpc", box_mpc)
    WA("J_mean",        J_mean)
    WA("J_scalar",      J_scalar)
    WA("NU_LL_species", NU_LL_SPECIES)
#undef WA
    H5Gclose(gid); H5Fclose(fid);
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, N_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &N_ranks);

    if (argc < 5) {
        if (rank == 0)
            fprintf(stderr,
                "Usage: %s <snap> <agn_dir> <output.hdf5> <sed.hdf5>"
                " [agn_dir2|none] [dataset_name] [species:H|He]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int         snap_target = atoi(argv[1]);
    const char *agn_dir     = argv[2];
    const char *outfile     = argv[3];
    const char *sed_file    = argv[4];
    const char *agn_dir2    = (argc >= 6 && strcmp(argv[5],"none") != 0)
                              ? argv[5] : NULL;
    const char *dataset     = (argc >= 7) ? argv[6]
                              : "LuminosityAGN";
    const char *prefix      = dataset;   /* file: {prefix}_{snap:03d}.hdf5 */
    const char *species     = (argc >= 8) ? argv[7] : "H";

    /* Set Lyman limit for H or He II.
     * z_max_lyman() is species-independent (uses only frequency ratios).
     * nu_emitted() uses NU_LL_SPECIES and is called for every shell. */
    if (strcmp(species, "He") == 0) {
        NU_LL_SPECIES = NU_LL_He;
    } else {
        NU_LL_SPECIES = NU_LL_H;
        species = "H";
    }

    char dset_out[32];
    snprintf(dset_out, sizeof(dset_out), "Jalpha_AGN_%s", species);

    if (rank == 0) {
        printf("==============================\n");
        printf("Array task : (from SLURM)\n");
        printf("Snapshot   : %d\n", snap_target);
        printf("Species    : %s  (NU_LL = %.4e Hz)\n", species, NU_LL_SPECIES);
        printf("AGN dir    : %s\n", agn_dir);
        if (agn_dir2) printf("AGN dir2   : %s\n", agn_dir2);
        printf("Dataset    : %s\n", dataset);
        printf("SED file   : %s\n", sed_file);
        printf("Output     : %s  [dataset: %s]\n", outfile, dset_out);
        printf("MPI ranks  : %d\n", N_ranks);
        printf("OMP threads: %d\n", N_OMP);
        printf("==============================\n\n");
    }

    fftwf_init_threads();
    fftwf_plan_with_nthreads(N_OMP);
    double t_start = MPI_Wtime();

    /* ---- Step 1: snapshot metadata ---- */
    if (rank == 0) printf("Loading snapshot metadata...\n");
    load_snap_metadata(agn_dir, agn_dir2, prefix, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    if (snap_redshifts[snap_target] < 0) {
        fprintf(stderr, "ERROR: snap %03d not found\n", snap_target);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double z_obs   = snap_redshifts[snap_target];
    double box_mpc = BOX_CMPC;
    int    N       = N_PIX;
    long   N3      = (long)N * N * N;
    if (rank == 0) {
        double cell_mpc  = box_mpc / N;
        double cell_ckpc = cell_mpc * 1000.0;
        printf("z_obs = %.6f  (snap %03d)\n", z_obs, snap_target);
        printf("\n--- Geometry ---\n");
        printf("  BoxSize     = %.3f cMpc\n",                  box_mpc);
        printf("  N_pix       = %d  (%ld voxels total)\n",    N, (long)N*N*N);
        printf("  cell size   = %.4f cMpc  = %.2f ckpc\n",    cell_mpc, cell_ckpc);
        printf("  LUM_UNIT    = %.4e erg/s/cm^3 per code unit\n", LUM_UNIT);
        printf("  NU_LL       = %.4e Hz  = %.2f eV  (species %s)\n",
               NU_LL_SPECIES, NU_LL_SPECIES * PLANCK / 1.602e-12, species);
        printf("  H(z_obs)    = %.4e s^-1\n",                  H_cgm(z_obs));
        printf("  c/H(z_obs)  = %.4e cm  = %.2f cMpc\n\n",
               C_CGS/H_cgm(z_obs), C_CGS/H_cgm(z_obs)/CMPC_CGS);
    }

    /* ---- Step 2: AGN SED ---- */
    if (rank == 0) printf("Loading AGN SED...\n");
    SpecTable st; memset(&st, 0, sizeof(st));
    load_agn_sed(sed_file, &st, rank);
    if (rank == 0) {
        /* Print eps_agn at each Lyman-n frequency at z_obs (nu = nu_n exactly)
         * to confirm the SED covers the relevant windows. */
        printf("  eps_agn at Lyman-n frequencies [ph/Hz/erg]:\n");
        printf("  %4s  %14s  %8s  %14s\n",
               "n", "nu_n [Hz]", "E_n [eV]", "eps_agn(nu_n)");
        for (int nn = 2; nn <= 10; nn++) {
            if (f_rec(nn) == 0.0) continue;
            double nu_n = NU_LL_SPECIES * (1.0 - 1.0/((double)nn*nn));
            double E_eV = nu_n * PLANCK / 1.602e-12;
            printf("  %4d  %.4e  %8.2f  %.4e\n",
                   nn, nu_n, E_eV, eps_b(&st, nu_n));
        }
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Step 3: FFTW buffers and plans ---- */
    if (rank == 0) printf("Planning FFTs...\n");
    float         *work_r = (float*)fftwf_malloc(N3 * sizeof(float));
    fftwf_complex *work_k = (fftwf_complex*)fftwf_malloc(
                                (long)N*N*(N/2+1) * sizeof(fftwf_complex));
    int n_arr[3] = {N, N, N};
    fftwf_plan plan_r2c = fftwf_plan_dft_r2c(3, n_arr, work_r, work_k, FFTW_MEASURE);
    fftwf_plan plan_c2r = fftwf_plan_dft_c2r(3, n_arr, work_k, work_r, FFTW_MEASURE);
    if (rank == 0) printf("FFT plans ready.\n\n");

    /* ---- Step 4: cache and accumulation buffers ---- */
    SnapCache cache;
    cache_init(&cache, N3, agn_dir, agn_dir2, prefix, dataset);

    float *agn_smooth = (float*)fftwf_malloc(N3 * sizeof(float));
    float *J_partial  = (float*)fftwf_malloc(N3 * sizeof(float));
    memset(J_partial, 0, N3 * sizeof(float));

    /* ---- Step 5: main Lyman-series loop ---- */
    if (rank == 0) printf("Computing J_alpha_AGN partial boxes...\n");
    int n_shells_done = 0;

    for (int n = 2; n <= N_MAX; n++) {
        if ((n-2) % N_ranks != rank) continue;  /* round-robin over n */

        double frec = f_rec(n);
        if (frec == 0.0) continue;               /* n=3: skip (f_rec=0) */

        double z_max = z_max_lyman(z_obs, n);
        double R_max = comoving_Mpc(z_obs, z_max, 0);
        double R_min = box_mpc / N;              /* 1 cell = first log shell */

        /* Lyman-n photon frequency window at z_obs:
         * nu_lo = nu_n (photon emitted at z_obs itself)
         * nu_hi = nu_{n+1} (photon emitted at z_max, just below next limit) */
        double nu_lo = NU_LL_SPECIES * (1.0 - 1.0/((double)n*n));
        double nu_hi = NU_LL_SPECIES * (1.0 - 1.0/((double)(n+1)*(n+1)));
        double E_lo  = nu_lo * PLANCK / 1.602e-12;   /* eV */
        double E_hi  = nu_hi * PLANCK / 1.602e-12;
        double dz_total = z_max - z_obs;
        printf("  rank %d: n=%2d  f_rec=%.4f  z_max=%.3f  R_max=%7.1f Mpc"
               "  dz=%.3f\n",
               rank, n, frec, z_max, R_max, dz_total);
        printf("           nu window = [%.3e, %.3e] Hz"
               "  =  [%5.2f, %5.2f] eV\n",
               nu_lo, nu_hi, E_lo, E_hi);
        fflush(stdout);

        /* Shells: an inner shell [0, R_in] covering the cell itself, then
         * N_SHELLS log-spaced shells from R_min to R_max. A line whose
         * R_max is below one cell gets only the inner shell [0, R_max]. */
        double R_in     = (R_max < R_min) ? R_max : R_min;
        int    n_log    = (R_max > R_min) ? N_SHELLS : 0;
        double log_Rmin = log(R_min);
        double dlogR    = (n_log > 0) ? (log(R_max) - log_Rmin) / n_log : 0.0;

        for (int is = -1; is < n_log; is++) {
            double R_lo  = (is < 0) ? 0.0  : exp(log_Rmin +  is      * dlogR);
            double R_hi  = (is < 0) ? R_in : exp(log_Rmin + (is+1.0) * dlogR);
            double R_mid = 0.5*(R_lo + R_hi);

            double z_prime = R_to_z(z_obs, z_max, R_mid);
            /* Shell dz (the outermost shell ends exactly at z_max) */
            double z_r_lo  = (is < 0)          ? z_obs : R_to_z(z_obs, z_max, R_lo);
            double z_r_hi  = (is == n_log - 1) ? z_max : R_to_z(z_obs, z_max, R_hi);
            double dz      = z_r_hi - z_r_lo;

            double nu_p = nu_emitted(z_prime, z_obs, n);
            double eb   = eps_b(&st, nu_p);          /* [ph/Hz/erg]  */
            double c_H  = C_CGS / H_cgm(z_prime);   /* [cm]         */

            /*
             * prefac = (1+z_obs)^2/4pi x f_rec x eps_agn(nu') x c/H x LUM_UNIT x dz
             *
             * x box[i] [code units/vox]:
             *   [sr^-1] x [ph/Hz/erg] x [cm] x [erg/s/cm^3/codeunit] x [code/vox]
             *   = [ph/s/Hz/cm^2/sr]  OK
             *
             * NO /MP_CGS: stellar code divided by m_p to convert mass flux
             * [g/s/cm^3] to baryon rate [bar/s/cm^3]. AGN luminosity is already
             * in erg/s; eps_agn already carries the /erg from 1/(h*nu).
             */
            double prefac = (1.0+z_obs)*(1.0+z_obs) / (4.0*M_PI_)
                            * frec
                            * eb
                            * c_H
                            * LUM_UNIT
                            * dz;

            /* Interpolate AGN box at z_prime, average over the shell R_lo < r < R_hi */
            float *agn_z = cache_get(&cache, z_prime, rank);
            smooth_shell(agn_z, agn_smooth,
                         N, box_mpc, R_lo, R_hi,
                         plan_r2c, plan_c2r, work_k, work_r);

            #pragma omp parallel for num_threads(N_OMP)
            for (long i = 0; i < N3; i++)
                J_partial[i] += (float)(prefac * (double)agn_smooth[i]);

            n_shells_done++;

            /* Progress: inner shell, first log shell, then every N_SHELLS/4 */
            if (is <= 0 || (is+1) % (N_SHELLS/4) == 0) {
                printf("    [rank %d n=%2d] shell %3d/%d"
                       "  R_mid=%8.2f Mpc  z'=%.4f"
                       "  eps_agn=%.3e [ph/Hz/erg]"
                       "  prefac=%.3e\n",
                       rank, n, is+1, N_SHELLS,
                       R_mid, z_prime, eb, prefac);
                fflush(stdout);
            }
        }

        /* Summary for this n-line */
        {
            double jp_mean = 0.0;
            for (long i = 0; i < N3; i++) jp_mean += J_partial[i];
            jp_mean /= N3;
            printf("  rank %d: n=%2d done -- running J_partial mean = %.4e"
                   " [ph/s/cm^2/Hz/sr]\n", rank, n, jp_mean);
            fflush(stdout);
        }
    }
    printf("  rank %d: %d shells done (all n)\n", rank, n_shells_done);
    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Step 6: MPI reduce ---- */
    float *J_total = NULL;
    if (rank == 0) {
        J_total = (float*)fftwf_malloc(N3 * sizeof(float));
        memset(J_total, 0, N3 * sizeof(float));
    }
    MPI_Reduce(J_partial, J_total, (int)N3,
               MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* ---- Step 7: rank 0 -- stats and scalar sanity check ---- */
    if (rank == 0) {
        /* Shell windows ring, so the sum can dip slightly below zero right
         * next to bright sources. Clip only here, once, and report it. */
        long   n_neg = 0;
        double J_min = 0.0, neg_sum = 0.0, J_raw = 0.0;
        for (long i = 0; i < N3; i++) {
            J_raw += J_total[i];
            if (J_total[i] < 0.0f) {
                n_neg++;  neg_sum += J_total[i];
                if (J_total[i] < J_min) J_min = J_total[i];
                J_total[i] = 0.0f;
            }
        }
        printf("  clipped %ld negative voxels (%.3e of box), min J = %.3e, "
               "mean shift = %.3e of J_mean\n",
               n_neg, (double)n_neg/N3, J_min, (J_raw != 0.0) ? -neg_sum/J_raw : 0.0);

        double J_sum = 0.0, J_sq = 0.0, J_max = 0.0, J_min_nz = 1e30;
        long   n_pos = 0;
        for (long i = 0; i < N3; i++) {
            double v = (double)J_total[i];
            J_sum += v;
            J_sq  += v * v;
            if (v > J_max) J_max = v;
            if (v > 0.0  && v < J_min_nz) J_min_nz = v;
            if (v > 0.0) n_pos++;
        }
        double J_mean = J_sum / N3;
        double J_var  = J_sq / N3 - J_mean * J_mean;
        double J_std  = (J_var > 0.0) ? sqrt(J_var) : 0.0;

        /*
         * Scalar sanity check: compute J_alpha using box-mean AGN luminosity
         * instead of the full 3D field. Result should equal J_mean.
         * Uses the same cache (already populated) -- no extra disk I/O.
         */
        double J_scalar = 0.0;
        for (int n = 2; n <= N_MAX; n++) {
            double frec = f_rec(n);
            if (frec == 0.0) continue;
            double z_max = z_max_lyman(z_obs, n);

            double z_pts[N_SNAPS+2]; int np2 = 0;
            z_pts[np2++] = z_obs;
            for (int i = N_SNAPS-1; i >= 0; i--) {
                double z = snap_redshifts[i]; if (z < 0) continue;
                if (z > z_obs && z <= z_max) z_pts[np2++] = z;
            }
            z_pts[np2++] = z_max;
            for (int a = 1; a < np2; a++) {       /* insertion sort */
                double tmp = z_pts[a]; int b = a-1;
                while (b >= 0 && z_pts[b] > tmp) { z_pts[b+1]=z_pts[b]; b--; }
                z_pts[b+1] = tmp;
            }
            for (int iz = 0; iz < np2-1; iz++) {
                double zm  = 0.5*(z_pts[iz]+z_pts[iz+1]);
                double dz  = z_pts[iz+1] - z_pts[iz];
                float *box = cache_get(&cache, zm, 0);
                double agn_mean = 0.0;
                for (long k = 0; k < N3; k++) agn_mean += box[k];
                agn_mean /= N3;
                double nu_p = nu_emitted(zm, z_obs, n);
                double c_H  = C_CGS / H_cgm(zm);
                J_scalar += (1.0+z_obs)*(1.0+z_obs) / (4.0*M_PI_)
                             * frec * eps_b(&st, nu_p) * c_H
                             * LUM_UNIT * agn_mean * dz;
            }
        }

        double t_end = MPI_Wtime();
        printf("\n=== Results (species %s, snap %03d, z=%.4f) ===\n",
               species, snap_target, z_obs);
        printf("  All intensities in [ph / s / cm^2 / Hz / sr]\n\n");
        printf("  J_mean      = %+.6e\n",  J_mean);
        printf("  J_std       = %+.6e\n",  J_std);
        printf("  J_std/J_mean= %.4f       (spatial fluctuation amplitude)\n",
               (J_mean > 0) ? J_std/J_mean : 0.0);
        printf("  J_max       = %+.6e\n",  J_max);
        printf("  J_min(>0)   = %+.6e\n",
               (J_min_nz < 1e29) ? J_min_nz : 0.0);
        printf("  n_positive  = %ld / %ld  (%.2f%% of voxels)\n",
               n_pos, (long)N*N*N, 100.0*n_pos/((long)N*N*N));
        printf("\n  Scalar sanity check (mean-field, no FFT):\n");
        printf("  J_scalar    = %+.6e\n",  J_scalar);
        printf("  ratio       = %.6f       (J_mean/J_scalar, should be ~1.0)\n",
               (J_scalar > 0) ? J_mean/J_scalar : 0.0);
        if (fabs(J_mean/J_scalar - 1.0) > 0.05 && J_scalar > 0)
            printf("  WARNING: ratio deviates >5%% -- check FFT normalisation\n");
        printf("\n  Timing:\n");
        printf("  Elapsed     = %.1f s  (%.2f min)\n",
               t_end-t_start, (t_end-t_start)/60.0);

        printf("\nWriting %s...\n", outfile);
        write_output(outfile, J_total, N, z_obs, box_mpc, J_mean, J_scalar, dset_out);
        fftwf_free(J_total);
    }

    /* ---- Cleanup ---- */
    cache_free(&cache);
    fftwf_free(agn_smooth); fftwf_free(J_partial);
    fftwf_free(work_r);     fftwf_free(work_k);
    fftwf_destroy_plan(plan_r2c);
    fftwf_destroy_plan(plan_c2r);
    fftwf_cleanup_threads();
    free(st.freq); free(st.N_nu);
    MPI_Finalize();
    if (rank == 0) printf("Done.\n");
    return 0;
}