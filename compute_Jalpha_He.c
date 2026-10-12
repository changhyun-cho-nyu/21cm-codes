/*
 * compute_Jalpha_He.c
 * ===================
 * Compute spatially-resolved He II Lyman-alpha intensity J_alpha_He(x, z_obs)
 * using hybrid MPI + OpenMP parallelism.
 *
 * Identical in structure to compute_Jalpha.c (hydrogen version).
 * The only physical change is the Lyman limit frequency:
 *   He II is hydrogen-like with Z=2, so NU_LL(HeII) = 4 * NU_LL(HI)
 *   = 1.31583e16 Hz. All Lyman-n frequencies and redshift windows
 *   shift accordingly. The BPASS spectrum is evaluated at the
 *   He II Lyman-n frequencies (~4x higher than H), which are well
 *   within the BPASS frequency grid (max ~3e18 Hz).
 *
 * The result J_alpha_He feeds the WF coupling for the
 * He+ 3.46 cm hyperfine line (analogous to H 21 cm).
 *
 * Parallelism strategy
 * --------------------
 * MPI  : each rank handles a disjoint subset of Lyman-n lines.
 *        At the end, MPI_Allreduce sums all partial J_alpha_He boxes.
 * OMP  : each rank uses N_OMP threads for FFT and voxel loops.
 *
 * SFR data location
 * -----------------
 * Two fixed storage locations hold the StarFormationRate snapshots,
 * both using the identical nested layout:
 *   <dir>/StarFormationRate/StarFormationRate_NNN.hdf5
 * SFR_DIR_OLD is tried first for every snapshot index; SFR_DIR_NEW
 * is the fallback. These are hardcoded rather than passed as
 * arguments, since there are only ever two locations and routing
 * this through argv/sbatch logic has repeatedly been a source of
 * mismatched-path bugs.
 *
 * Calling convention
 * ------------------
 * mpirun -np 16 ./compute_Jalpha_He  <snap>  <output_hdf5>
 *
 *   snap        : integer snapshot number (e.g. 177)
 *   output_hdf5 : full path for output file (e.g. /path/Jalpha_He_177.hdf5)
 *
 * BPASS spectrum file is hardcoded below — edit SPECTRUM_FILE.
 * The stellar emissivity includes the host escape fraction F_ESC_STAR_HE
 * (default 0.18, LUMINA's value at all z); -DF_ESC_STAR_HE=1.0 for no host absorption.
 *
 * Compile
 * -------
 * mpicc -O3 -march=native -fopenmp compute_Jalpha_He.c -o compute_Jalpha_He \
 *     -lfftw3f_omp -lfftw3f -lhdf5 -lm
 */

#include <fftw3.h>
#include <hdf5.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* =========================================================================
 * Configuration
 * ========================================================================= */

#ifndef SPECTRUM_FILE
#define SPECTRUM_FILE \
    "/orcd/data/mvogelsb/004/ozier/ZOOM_XL/sharedData/" \
    "bpass_spectra_bin_chab100_v2.2.1.hdf5"
#endif

/* Hardcoded SFR storage locations. Both use the same nested layout:
 *   <dir>/StarFormationRate/StarFormationRate_NNN.hdf5
 * SFR_DIR_OLD is tried first for every snapshot index; SFR_DIR_NEW
 * is the fallback used for later-era snapshots stored separately. */
#ifndef SFR_DIR_OLD
#define SFR_DIR_OLD \
    "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/" \
    "Flagship_new/postprocessing/ren_640"
#endif

#ifndef SFR_DIR_NEW
#define SFR_DIR_NEW \
    "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/" \
    "Flagship_below_z_4p75_new/postprocessing/ren_640"
#endif

#ifndef N_SNAPS
#define N_SNAPS      709     /* total number of snapshots                  */
#endif
#define N_MAX        30      /* max Lyman series index                     */
#define N_SHELLS     200     /* radial shells per Lyman-n line             */
/* Young-star metallicity weights w_j(z) per BPASS metallicity, written by
 * make_Z_snapshots.py (Z_weights.txt). EDIT this path. */
#ifndef Z_WEIGHTS_FILE
#define Z_WEIGHTS_FILE \
    "/orcd/data/mvogelsb/004/chcho/cell_by_cell/helium/Z_weights.txt"
#endif
#define N_OMP        4       /* OpenMP threads per MPI rank (= cpus-per-task) */
/* Escape fraction of stellar He II Lyman-band photons (40.8-54.4 eV) from the
 * host galaxy: the frequency-independent f_esc of LUMINA's radiative transfer,
 * so stars are treated like the AGN, whose rendered luminosity already carries
 * LUMINA's f_esc(L). -DF_ESC_STAR_HE=1.0 gives the no-host-absorption upper limit. */
#ifndef F_ESC_STAR_HE
#define F_ESC_STAR_HE 0.18
#endif

/* =========================================================================
 * Physical constants (CGS)
 * ========================================================================= */

#define C_CGS      2.9979e10
#define MSUN_CGS   1.989e33
#define YR_CGS     3.15576e7
#define MP_CGS     1.6738e-24
#define CMPC_CGS   3.0857e24
#define M_PI_      3.14159265358979323846

/* He II Lyman limit: Z=2 hydrogen-like, so nu_LL(HeII) = 4 * nu_LL(HI)  */
#define NU_LL      1.31583e16   /* He II Lyman limit [Hz]                  */

/* Cosmology (THESAN-XL / Planck 2015) */
#define OMEGA_M    0.30964144
#define OMEGA_L    0.6902672
#define H_0_CGS    (0.6766 * 100.0 * 3.241e-20)   /* s^-1 */

/* =========================================================================
 * Global geometry (filled from HDF5 header of snap 0 in sfr_dir)
 * ========================================================================= */

static int    N_PIX    = 640;
static double BOX_CMPC = 500.0;
static double SFR_UNIT = 0.0;   /* g/s/cm^3 per Msun/yr/voxel */

/* Per-snapshot redshift table */
static double snap_redshifts[N_SNAPS];

/* =========================================================================
 * Recycling fractions  [Pritchard & Furlanetto 2006]
 * He II is hydrogen-like (Z=2), so the same cascade fractions apply.
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
 * Cosmological utilities
 * ========================================================================= */

static double H_cgm(double z) {
    return H_0_CGS * sqrt(OMEGA_M * pow(1.0+z, 3.0) + OMEGA_L);
}

/*
 * z_max_lyman: redshift of the He II Lyman-(n+1) edge as seen from z_obs.
 * A photon emitted at Lyman-(n+1) frequency at z_max redshifts into the
 * Lyman-n frequency at z_obs.  Formula is identical to the H version;
 * only NU_LL (now He II) affects nu_emitted — not this function directly.
 */
static double z_max_lyman(double z_obs, int n) {
    return (1.0+z_obs) * (1.0 - pow(n+1.0,-2.0))
                       / (1.0 - pow((double)n,-2.0)) - 1.0;
}

/*
 * nu_emitted: He II Lyman-n emission frequency at z_emit that redshifts
 * to Lyman-alpha at z_obs.  Uses He II NU_LL = 1.31583e16 Hz.
 */
static double nu_emitted(double z_emit, double z_obs, int n) {
    return NU_LL * (1.0 - 1.0/((double)n*(double)n))
                 * (1.0+z_emit) / (1.0+z_obs);
}

/* Comoving distance from z1 to z2 [Mpc], simple trapezoid */
static double comoving_Mpc(double z1, double z2, int nsteps) {
    if (z2 <= z1) return 0.0;
    double dz = (z2 - z1) / nsteps;
    double s  = 0.0;
    for (int i = 0; i <= nsteps; i++) {
        double z = z1 + i * dz;
        double w = (i==0 || i==nsteps) ? 0.5 : 1.0;
        s += w * C_CGS / H_cgm(z);
    }
    return s * dz / CMPC_CGS;
}

/* Invert comoving distance: find z such that comoving_Mpc(z_obs,z) = R_mpc */
static double R_to_z(double z_obs, double z_max, double R_mpc) {
    double za = z_obs, zb = z_max;
    for (int i = 0; i < 60; i++) {
        double zm = 0.5*(za+zb);
        if (comoving_Mpc(z_obs, zm, 60) < R_mpc) za = zm;
        else                                       zb = zm;
    }
    return 0.5*(za+zb);
}

/* =========================================================================
 * HDF5 helpers
 * ========================================================================= */

static double read_attr_double(hid_t loc, const char *name) {
    double val = 0.0;
    hid_t a = H5Aopen(loc, name, H5P_DEFAULT);
    H5Aread(a, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(a);
    return val;
}

static int read_attr_int(hid_t loc, const char *name) {
    int val = 0;
    hid_t a = H5Aopen(loc, name, H5P_DEFAULT);
    H5Aread(a, H5T_NATIVE_INT, &val);
    H5Aclose(a);
    return val;
}

/* =========================================================================
 * Load snapshot redshifts + geometry
 * ========================================================================= */

static void build_snap_path(const char *sfr_dir, int snap, char *out) {
    snprintf(out, 2048, "%s/StarFormationRate/StarFormationRate_%03d.hdf5", sfr_dir, snap);
}

static double snap_z_max(void) {
    for (int i = 0; i < N_SNAPS; i++)
        if (snap_redshifts[i] >= 0.0) return snap_redshifts[i];
    return -1.0;
}

/* Tries SFR_DIR_OLD first, then SFR_DIR_NEW, for every snapshot index.
 * Both directories use the identical nested StarFormationRate/ layout. */
static int snap_src[N_SNAPS];   /* 1 = first directory, 2 = second, 0 = missing */

/* Print which directory every snapshot range was found in, flag missing
 * snapshots, and abort if redshift does not decrease with snapshot number
 * (that would mean the two directories do not share one numbering). */
static void report_snapshots(int rank, const char *dir1, const char *dir2) {
    if (rank == 0) {
        printf("  Snapshot sources:\n");
        int i = 0;
        while (i < N_SNAPS) {
            int j = i;
            while (j+1 < N_SNAPS && snap_src[j+1] == snap_src[i]) j++;
            if (snap_src[i] == 0)
                printf("    snaps %03d-%03d  MISSING in both directories\n", i, j);
            else
                printf("    snaps %03d-%03d  z = %7.3f .. %7.3f  <- %s\n", i, j,
                       snap_redshifts[i], snap_redshifts[j],
                       snap_src[i] == 1 ? dir1 : dir2);
            i = j+1;
        }
    }
    int prev = -1;
    for (int i = 0; i < N_SNAPS; i++) {
        if (snap_src[i] == 0) continue;
        if (prev >= 0 && snap_redshifts[i] > snap_redshifts[prev]) {
            if (rank == 0)
                fprintf(stderr, "ERROR: redshift increases from snap %03d (z=%.4f) to snap %03d "
                        "(z=%.4f): the directories do not share one snapshot numbering\n",
                        prev, snap_redshifts[prev], i, snap_redshifts[i]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        prev = i;
    }
}

static void load_snap_metadata(int rank) {
    char path[2048];
    for (int i = 0; i < N_SNAPS; i++) {
        build_snap_path(SFR_DIR_OLD, i, path);
        hid_t fid = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        snap_src[i] = 1;
        if (fid < 0) {
            build_snap_path(SFR_DIR_NEW, i, path);
            fid = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
            snap_src[i] = 2;
        }
        if (fid < 0) {
            snap_redshifts[i] = -1.0;
            snap_src[i] = 0;
            continue;
        }
        hid_t gid = H5Gopen(fid, "Header", H5P_DEFAULT);
        snap_redshifts[i] = read_attr_double(gid, "Redshift");

        if (SFR_UNIT == 0.0) {
            double boxsize  = read_attr_double(gid, "BoxSize");
            double hubble   = read_attr_double(gid, "HubbleParam");
            double unitlen  = read_attr_double(gid, "UnitLength_in_cm");
            N_PIX    = read_attr_int(gid, "NumPixels");
            BOX_CMPC = boxsize / hubble / 1000.0;
            double box_cm  = boxsize * unitlen / hubble;
            double cell_cm = box_cm / N_PIX;
            SFR_UNIT = (MSUN_CGS / YR_CGS) / (cell_cm*cell_cm*cell_cm);
            if (rank == 0)
                printf("  Box=%.1f cMpc  N=%d  SFR_UNIT=%.4e g/s/cm3/vox\n",
                       BOX_CMPC, N_PIX, SFR_UNIT);
        }
        H5Gclose(gid);
        H5Fclose(fid);
    }
    report_snapshots(rank, SFR_DIR_OLD, SFR_DIR_NEW);
}

/* =========================================================================
 * Read one SFR box from disk
 * ========================================================================= */

static char last_sfr_path[2048];   /* file actually opened by the last read_sfr_box() */

static void read_sfr_box(int snap, float *box) {
    char path[2048];
    build_snap_path(SFR_DIR_OLD, snap, path);
    hid_t fid = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) {
        build_snap_path(SFR_DIR_NEW, snap, path);
        fid = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fid < 0) {
            fprintf(stderr, "ERROR: snap %03d not found in either directory\n", snap);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    hid_t did = H5Dopen(fid, "StarFormationRate", H5P_DEFAULT);
    H5Dread(did, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, box);
    snprintf(last_sfr_path, sizeof last_sfr_path, "%s", path);
    H5Dclose(did);
    H5Fclose(fid);

    long N3 = (long)N_PIX * N_PIX * N_PIX;
    #pragma omp parallel for num_threads(N_OMP)
    for (long i = 0; i < N3; i++) {
        if (box[i] < 0.0f) box[i] = 0.0f;
    }
}

/* =========================================================================
 * Snapshot cache
 * ========================================================================= */

typedef struct {
    int    snap_lo;
    int    snap_hi;
    float *box_lo;
    float *box_hi;
    float *box_interp;
    long   N3;
} SnapCache;

static void cache_init(SnapCache *c, long N3) {
    c->snap_lo    = -9999;
    c->snap_hi    = -9999;
    c->N3         = N3;
    c->box_lo     = fftwf_malloc(N3 * sizeof(float));
    c->box_hi     = fftwf_malloc(N3 * sizeof(float));
    c->box_interp = fftwf_malloc(N3 * sizeof(float));
    if (!c->box_lo || !c->box_hi || !c->box_interp) {
        fprintf(stderr, "ERROR: cache fftwf_malloc failed (N3=%ld, %.2f GB each)\n",
                N3, N3*4.0/1e9);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void cache_free(SnapCache *c) {
    fftwf_free(c->box_lo);
    fftwf_free(c->box_hi);
    fftwf_free(c->box_interp);
}

static void find_bracket(double z_prime, int *slo, int *shi) {
    if (z_prime >= snap_redshifts[0]) {
        *slo = 0; *shi = 0; return;
    }
    if (z_prime <= snap_redshifts[N_SNAPS-1]) {
        *slo = N_SNAPS-1; *shi = N_SNAPS-1; return;
    }
    int lo = 0, hi = N_SNAPS-1;
    while (hi - lo > 1) {
        int mid = (lo+hi)/2;
        if (snap_redshifts[mid] >= z_prime) lo = mid;
        else                                hi = mid;
    }
    *slo = lo;
    *shi = hi;
}

static float *cache_get(SnapCache *c, double z_prime, int rank) {
    int slo, shi;
    find_bracket(z_prime, &slo, &shi);

    if (slo != c->snap_lo) {
        if (snap_redshifts[slo] < 0) {
            fprintf(stderr, "ERROR: snap %d not found in any directory\n", slo);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        read_sfr_box(slo, c->box_lo);
        printf("  [rank %d cache] read snap %03d  z=%.4f  <- %s\n",
               rank, slo, snap_redshifts[slo], last_sfr_path);
        c->snap_lo = slo;
    }
    if (shi != c->snap_hi) {
        if (snap_redshifts[shi] < 0) {
            fprintf(stderr, "ERROR: snap %d not found in any directory\n", shi);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        read_sfr_box(shi, c->box_hi);
        printf("  [rank %d cache] read snap %03d  z=%.4f  <- %s\n",
               rank, shi, snap_redshifts[shi], last_sfr_path);
        c->snap_hi = shi;
    }

    if (slo == shi) return c->box_lo;

    double z_lo = snap_redshifts[slo];
    double z_hi = snap_redshifts[shi];
    float  w    = (float)((z_prime - z_lo) / (z_hi - z_lo));

    #pragma omp parallel for num_threads(N_OMP)
    for (long i = 0; i < c->N3; i++) {
        float v = (1.0f-w)*c->box_lo[i] + w*c->box_hi[i];
        c->box_interp[i] = (v > 0.0f) ? v : 0.0f;
    }
    return c->box_interp;
}

/* =========================================================================
 * Spectrum table (BPASS) -- ALL metallicities
 *   N_nu[iZ*Nfreq + ifreq] : photon yield integrated over stellar age
 *                            [photons / Hz / baryon], one row per BPASS Z
 * Age bins are centred in log(age) (BPASS grid: uniform 0.1 dex), the first
 * bin running from t = 0. Identical to the Python notebook.
 * ========================================================================= */

typedef struct {
    int     Nfreq;
    int     N_met;
    double *freq;   /* [Nfreq]          Hz            */
    double *logZ;   /* [N_met]          log10(Z)      */
    double *N_nu;   /* [N_met * Nfreq]  ph/Hz/baryon  */
} SpecTable;

static SpecTable load_spectrum(int rank) {
    hid_t fid = H5Fopen(SPECTRUM_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    int Nfreq, N_age, N_met;
    {
        /* Number_of_freq, N_age, N_metallicity are scalar DATASETS */
        hid_t d2;
        d2=H5Dopen(fid,"Number_of_freq",H5P_DEFAULT); H5Dread(d2,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,&Nfreq); H5Dclose(d2);
        d2=H5Dopen(fid,"N_age",         H5P_DEFAULT); H5Dread(d2,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,&N_age); H5Dclose(d2);
        d2=H5Dopen(fid,"N_metallicity", H5P_DEFAULT); H5Dread(d2,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,&N_met); H5Dclose(d2);
        if (rank==0)
            printf("  Spectrum dims: Nfreq=%d  N_age=%d  N_met=%d\n",
                   Nfreq, N_age, N_met);
    }

    double *freq_arr = malloc(Nfreq * sizeof(double));
    double *age_arr  = malloc(N_age * sizeof(double));
    double *met_arr  = malloc(N_met * sizeof(double));
    double *Lnu      = malloc((long)N_met*N_age*Nfreq * sizeof(double));
    double *log_age  = malloc(N_age * sizeof(double));
    double *logZ     = malloc(N_met * sizeof(double));
    double *N_nu     = calloc((long)N_met*Nfreq, sizeof(double));
    if (!freq_arr || !age_arr || !met_arr || !Lnu || !log_age || !logZ || !N_nu) {
        fprintf(stderr, "ERROR: spectrum malloc failed (Nfreq=%d N_age=%d N_met=%d)\n",
                Nfreq, N_age, N_met);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    hid_t d;
    d=H5Dopen(fid,"freq",H5P_DEFAULT);        H5Dread(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,freq_arr); H5Dclose(d);
    d=H5Dopen(fid,"age",H5P_DEFAULT);         H5Dread(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,age_arr);  H5Dclose(d);
    d=H5Dopen(fid,"metallicity",H5P_DEFAULT); H5Dread(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,met_arr);  H5Dclose(d);
    d=H5Dopen(fid,"Lnu",H5P_DEFAULT);         H5Dread(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,Lnu);      H5Dclose(d);
    H5Fclose(fid);

    const double PLANCK    = 6.62607015e-27;
    const double MSOL2GRAM = 1.989e33;
    const double GYR2SEC   = 3.15576e16;

    for (int ia = 0; ia < N_age; ia++) {
        double a = (age_arr[ia] < 1e-6) ? 1e-6 : age_arr[ia];
        log_age[ia] = log10(a);
    }
    double dlog = (N_age > 1) ? (log_age[N_age/2] - log_age[N_age/2 - 1]) : 0.1;

    for (int iz = 0; iz < N_met; iz++) {
        logZ[iz] = log10(met_arr[iz]);
        for (int ia = 0; ia < N_age; ia++) {
            double edge_hi = pow(10.0, log_age[ia] + 0.5*dlog);
            double edge_lo = (ia == 0) ? 0.0 : pow(10.0, log_age[ia] - 0.5*dlog);
            double dt      = (edge_hi - edge_lo) * GYR2SEC;
            long   off     = ((long)iz*N_age + ia)*Nfreq;
            for (int ifreq = 0; ifreq < Nfreq; ifreq++) {
                double lp = Lnu[off+ifreq] / (PLANCK*freq_arr[ifreq])
                            * (MP_CGS/MSOL2GRAM);
                N_nu[(long)iz*Nfreq + ifreq] += lp * dt;
            }
        }
    }
    if (rank==0) {
        printf("  Spectrum: Nfreq=%d  %d metallicities, Z = %.1e .. %.1e\n",
               Nfreq, N_met, met_arr[0], met_arr[N_met-1]);
        for (int iz = 1; iz < N_met; iz++)
            if (!(logZ[iz] > logZ[iz-1])) {
                fprintf(stderr, "ERROR: BPASS metallicity grid not increasing\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
    }

    free(log_age); free(age_arr); free(met_arr); free(Lnu);

    SpecTable st = {Nfreq, N_met, freq_arr, logZ, N_nu};
    return st;
}

/* =========================================================================
 * Young-star metallicity distribution w_j(z)  (BPASS yield averaged over the
 * metallicities of the stars forming at z, no single "mean" metallicity)
 *
 * Text file from make_Z_snapshots.py, rows "redshift  w_00 ... w_{N-1}":
 * the fraction of the initial mass of young stars at each BPASS metallicity
 * grid point (lines starting with # are ignored, except that a
 * "# Z_grid = ..." line is checked against the BPASS table). The yield is
 *     eps(nu, z) = sum_j w_j(z) eps_b(nu, Z_j),
 * i.e. the young-star average of eps_b. A metallicity Z, which make_Z_snapshots.py
 * splits between its two neighbouring grid points with log-Z weights, gives
 * exactly the old single-Z eps_b(nu, Z). w_j is interpolated linearly in z
 * between rows and held at the end rows.
 * ========================================================================= */

#define NMET_MAX 64

static int     NZW  = 0;
static double *zw_z = NULL;   /* ascending redshift   [NZW]          */
static double *zw_w = NULL;   /* weights, row-major   [NZW * N_met]  */

static void load_Z_weights(const SpecTable *st, int rank) {
    int Nz = st->N_met;
    if (Nz > NMET_MAX) {
        fprintf(stderr, "ERROR: BPASS table has %d metallicities (max %d)\n", Nz, NMET_MAX);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    FILE *fp = fopen(Z_WEIGHTS_FILE, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open Z weights file %s\n", Z_WEIGHTS_FILE);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int  cap = 64, grid_checked = 0;
    zw_z = malloc(cap * sizeof(double));
    zw_w = malloc((long)cap * Nz * sizeof(double));
    char line[8192];
    while (fgets(line, sizeof line, fp)) {
        if (line[0] == '#') {
            char *p = strstr(line, "Z_grid =");
            if (p) {
                p += strlen("Z_grid =");
                for (int j = 0; j < Nz; j++) {
                    char *q;
                    double Zf = strtod(p, &q);
                    if (q == p || !(Zf > 0.0) ||
                        fabs(log10(Zf) - st->logZ[j]) > 1e-3 * (fabs(st->logZ[j]) + 1.0)) {
                        fprintf(stderr, "ERROR: %s: Z_grid entry %d does not match the BPASS "
                                "metallicity %.4g\n", Z_WEIGHTS_FILE, j, pow(10.0, st->logZ[j]));
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    p = q;
                }
                grid_checked = 1;
            }
            continue;
        }
        char *p = line, *q;
        double z = strtod(p, &q);
        if (q == p) continue;                       /* blank line */
        p = q;
        if (NZW == cap) {
            cap *= 2;
            zw_z = realloc(zw_z, cap * sizeof(double));
            zw_w = realloc(zw_w, (long)cap * Nz * sizeof(double));
        }
        double *w = zw_w + (long)NZW * Nz, sum = 0.0;
        for (int j = 0; j < Nz; j++) {
            w[j] = strtod(p, &q);
            if (q == p || !(w[j] >= 0.0)) {
                fprintf(stderr, "ERROR: %s: row at z = %g needs %d non-negative weights\n",
                        Z_WEIGHTS_FILE, z, Nz);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            sum += w[j];  p = q;
        }
        if (!(sum > 0.0)) continue;                 /* no young stars: skip */
        for (int j = 0; j < Nz; j++) w[j] /= sum;   /* guard against rounding */
        zw_z[NZW++] = z;
    }
    fclose(fp);
    if (NZW < 1) {
        fprintf(stderr, "ERROR: %s has no usable rows\n", Z_WEIGHTS_FILE);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int a = 1; a < NZW; a++) {                 /* sort rows by redshift */
        double tz = zw_z[a], tw[NMET_MAX];
        memcpy(tw, zw_w + (long)a*Nz, Nz*sizeof(double));
        int b = a-1;
        while (b >= 0 && zw_z[b] > tz) {
            zw_z[b+1] = zw_z[b];
            memcpy(zw_w + (long)(b+1)*Nz, zw_w + (long)b*Nz, Nz*sizeof(double));
            b--;
        }
        zw_z[b+1] = tz;
        memcpy(zw_w + (long)(b+1)*Nz, tw, Nz*sizeof(double));
    }
    if (rank == 0)
        printf("  Z weights: %d rows from %s, z = %.3f .. %.3f, %d metallicities (%s)\n",
               NZW, Z_WEIGHTS_FILE, zw_z[0], zw_z[NZW-1], Nz,
               grid_checked ? "Z_grid matches BPASS" : "WARNING: no Z_grid line to check");
}

/* Weights at redshift z: linear in z between rows, end rows held outside. */
static void Z_weights_at(const SpecTable *st, double z, double *w) {
    int Nz = st->N_met;
    if (NZW == 1 || z <= zw_z[0]) { memcpy(w, zw_w, Nz*sizeof(double)); return; }
    if (z >= zw_z[NZW-1]) { memcpy(w, zw_w + (long)(NZW-1)*Nz, Nz*sizeof(double)); return; }
    int lo = 0, hi = NZW-1;
    while (hi-lo > 1) {
        int mid = (lo+hi)/2;
        if (zw_z[mid] <= z) lo = mid; else hi = mid;
    }
    double t = (z - zw_z[lo]) / (zw_z[hi] - zw_z[lo]);
    const double *a = zw_w + (long)lo*Nz, *b = zw_w + (long)hi*Nz;
    for (int j = 0; j < Nz; j++) w[j] = (1.0-t)*a[j] + t*b[j];
}

/*
 * eps_mix: escaping photon yield [photons/Hz/baryon] at frequency nu for the
 * young stars forming at redshift z: F_ESC_STAR_HE x sum_j w_j(z) N_nu(nu, Z_j).
 * Linear in nu, edge value outside the frequency table.
 */
static double eps_mix(const SpecTable *st, double nu, double z) {
    int    Nf = st->Nfreq, Nz = st->N_met;
    int    lo = 0, hi = Nf-1;
    double wf;
    if      (nu <= st->freq[0])    { hi = 0;    wf = 0.0; }
    else if (nu >= st->freq[Nf-1]) { lo = Nf-1; wf = 0.0; }
    else {
        while (hi-lo > 1) {
            int mid=(lo+hi)/2;
            if (st->freq[mid]<nu) lo=mid; else hi=mid;
        }
        wf = (nu-st->freq[lo])/(st->freq[hi]-st->freq[lo]);
    }
    double w[NMET_MAX], s = 0.0;
    Z_weights_at(st, z, w);
    for (int j = 0; j < Nz; j++) {
        if (w[j] == 0.0) continue;
        const double *row = st->N_nu + (long)j*Nf;
        s += w[j] * (row[lo]*(1.0-wf) + row[hi]*wf);
    }
    return F_ESC_STAR_HE * s;
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
 * HDF5 output
 * Dataset name: J_alpha_He  (distinguishes from hydrogen J_alpha)
 * Units: photons s^-1 cm^-2 Hz^-1 sr^-1
 * ========================================================================= */

static void write_output(const char *outfile, const float *J,
                          int N, double z_obs, double box_mpc,
                          double J_mean, double J_scalar,
                          const SpecTable *st) {
    hid_t fid = H5Fcreate(outfile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[3] = {(hsize_t)N,(hsize_t)N,(hsize_t)N};
    hid_t sid = H5Screate_simple(3,dims,NULL);

    hid_t pid = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t c64 = (N < 64) ? (hsize_t)N : 64;          /* chunk must not exceed the grid */
    hsize_t chunk[3]={c64,c64,c64};
    H5Pset_chunk(pid,3,chunk);
    H5Pset_deflate(pid,4);

    hid_t did = H5Dcreate(fid,"J_alpha_He",H5T_NATIVE_FLOAT,
                           sid,H5P_DEFAULT,pid,H5P_DEFAULT);
    H5Dwrite(did,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,J);

    hid_t sc = H5Screate(H5S_SCALAR);
    #define WA(nm,val) { \
        double v=(val); \
        hid_t a=H5Acreate(did,nm,H5T_NATIVE_DOUBLE,sc,H5P_DEFAULT,H5P_DEFAULT); \
        H5Awrite(a,H5T_NATIVE_DOUBLE,&v); H5Aclose(a); }
    WA("Redshift",    z_obs)
    WA("BoxSize_cMpc",box_mpc)
    WA("J_mean",      J_mean)
    WA("J_scalar",    J_scalar)
    WA("NU_LL_HeII",  1.31583e16)
    WA("f_esc_star",  F_ESC_STAR_HE)
    #undef WA

    {
        double w[NMET_MAX], Zg[NMET_MAX];
        hsize_t nz = (hsize_t)st->N_met;
        Z_weights_at(st, z_obs, w);
        for (int j = 0; j < st->N_met; j++) Zg[j] = pow(10.0, st->logZ[j]);
        hid_t vs = H5Screate_simple(1, &nz, NULL);
        hid_t aw = H5Acreate(did, "Z_weights_obs", H5T_NATIVE_DOUBLE, vs, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(aw, H5T_NATIVE_DOUBLE, w);  H5Aclose(aw);
        hid_t ag = H5Acreate(did, "Z_grid", H5T_NATIVE_DOUBLE, vs, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(ag, H5T_NATIVE_DOUBLE, Zg); H5Aclose(ag);
        H5Sclose(vs);
    }

    hid_t str_t = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_t,64);
    hid_t a = H5Acreate(did,"Units",str_t,sc,H5P_DEFAULT,H5P_DEFAULT);
    H5Awrite(a,str_t,"photons s^-1 cm^-2 Hz^-1 sr^-1");
    H5Aclose(a);

    H5Sclose(sc); H5Tclose(str_t);
    H5Pclose(pid); H5Dclose(did);
    H5Sclose(sid); H5Fclose(fid);
}

/* =========================================================================
 * Main computation
 * ========================================================================= */

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (argc < 3) {
        if (rank==0)
            fprintf(stderr,
                "Usage: %s <snap> <output.hdf5>\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    int         snap_target = atoi(argv[1]);
    const char *outfile     = argv[2];

    time_t t0 = time(NULL);
    if (rank==0) {
        printf("=== compute_Jalpha_He: snap %03d ===\n", snap_target);
        printf("MPI ranks=%d  OMP threads/rank=%d\n", nranks, N_OMP);
        printf("He II Lyman limit: NU_LL = %.6e Hz  (4 x H Lyman limit)\n", (double)NU_LL);
        printf("SFR dir (primary) : %s\n", SFR_DIR_OLD);
        printf("SFR dir (fallback): %s\n", SFR_DIR_NEW);
        printf("Stellar escape fraction (He II band): f_esc = %.3f\n", (double)F_ESC_STAR_HE);
        printf("Output  : %s\n", outfile);
    }

    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    if (rank==0) printf("\nLoading snapshot metadata...\n");
    load_snap_metadata(rank);
    MPI_Barrier(MPI_COMM_WORLD);

    double z_obs = snap_redshifts[snap_target];
    if (z_obs < 0.0) {
        if (rank==0)
            fprintf(stderr,
                "ERROR: snap %03d has no SFR file in either directory "
                "(z_obs = sentinel -1). Aborting.\n", snap_target);
        MPI_Finalize();
        return 1;
    }
    if (rank==0) printf("z_obs = %.6f\n", z_obs);

    int    N      = N_PIX;
    long   N3     = (long)N*N*N;
    double box_mpc = BOX_CMPC;

    if (rank==0) printf("\nLoading BPASS spectrum...\n");
    SpecTable st = load_spectrum(rank);
    load_Z_weights(&st, rank);
    if (rank==0) {
        double w[NMET_MAX];
        Z_weights_at(&st, z_obs, w);
        printf("  young-star Z weights at z_obs=%.3f:", z_obs);
        for (int j = 0; j < st.N_met; j++) printf(" %.2g:%.3f", pow(10.0, st.logZ[j]), w[j]);
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    fftwf_init_threads();
    fftwf_plan_with_nthreads(N_OMP);

    float         *work_r     = fftwf_malloc(N3 * sizeof(float));
    fftwf_complex *work_k     = fftwf_malloc((long)N*N*(N/2+1)*sizeof(fftwf_complex));
    float         *sfr_smooth = fftwf_malloc(N3 * sizeof(float));
    if (!work_r || !work_k || !sfr_smooth) {
        fprintf(stderr, "ERROR: FFT workspace fftwf_malloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n_arr[3] = {N, N, N};
    if (rank==0) printf("Planning FFTs...\n");
    fftwf_plan plan_r2c = fftwf_plan_dft_r2c(3,n_arr,work_r,work_k,FFTW_MEASURE);
    fftwf_plan plan_c2r = fftwf_plan_dft_c2r(3,n_arr,work_k,work_r,FFTW_MEASURE);
    if (rank==0) printf("FFT plans ready.\n");

    SnapCache cache;
    cache_init(&cache, N3);

    float *J_partial = fftwf_malloc(N3 * sizeof(float));
    if (!J_partial) {
        fprintf(stderr, "ERROR: J_partial fftwf_malloc failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(J_partial, 0, N3*sizeof(float));

    if (rank==0) printf("\nComputing J_alpha_He partial boxes...\n");

    int n_shells_done = 0;

    for (int n = 2; n <= N_MAX; n++) {

        if ((n - 2) % nranks != rank) continue;

        double frec = f_rec(n);
        if (frec == 0.0) continue;

        double z_max = z_max_lyman(z_obs, n);
        if (z_max > snap_z_max()) z_max = snap_z_max();
        if (z_max <= z_obs)       continue;

        double R_max = comoving_Mpc(z_obs, z_max, 500);
        double R_min = box_mpc / N;

        printf("  rank %d: n=%2d  f_rec=%.4f  z_max=%.3f  R_max=%.1f Mpc\n",
               rank, n, frec, z_max, R_max);

        /* Shells: an inner shell [0, R_in] covering the cell itself, then
         * N_SHELLS log-spaced shells from R_min to R_max. A line whose
         * R_max is below one cell gets only the inner shell [0, R_max]. */
        double R_in     = (R_max < R_min) ? R_max : R_min;
        int    n_log    = (R_max > R_min) ? N_SHELLS : 0;
        double log_Rmin = log(R_min);
        double dlogR    = (n_log > 0) ? (log(R_max) - log_Rmin) / n_log : 0.0;

        for (int is = -1; is < n_log; is++) {
            double R_lo  = (is < 0) ? 0.0  : exp(log_Rmin + is       * dlogR);
            double R_hi  = (is < 0) ? R_in : exp(log_Rmin + (is+1.0) * dlogR);
            double R_mid = 0.5*(R_lo + R_hi);

            /* Emission redshift at R_mid */
            double z_prime = R_to_z(z_obs, z_max, R_mid);

            /* Shell dz (the outermost shell ends exactly at z_max) */
            double z_r_lo = (is < 0)          ? z_obs : R_to_z(z_obs, z_max, R_lo);
            double z_r_hi = (is == n_log - 1) ? z_max : R_to_z(z_obs, z_max, R_hi);
            double dz     = z_r_hi - z_r_lo;

            /* nu_emitted now uses He II NU_LL = 1.31583e16 Hz */
            double nu_p  = nu_emitted(z_prime, z_obs, n);
            double eb    = eps_mix(&st, nu_p, z_prime);   /* young stars at the EMISSION redshift */
            double c_H   = C_CGS / H_cgm(z_prime);
            double prefac = (1.0+z_obs)*(1.0+z_obs) / (4.0*M_PI_)
                            * frec * eb * c_H / MP_CGS * SFR_UNIT * dz;

            float *sfr_z = cache_get(&cache, z_prime, rank);

            /* Average over the shell R_lo < r < R_hi */
            smooth_shell(sfr_z, sfr_smooth,
                         N, box_mpc, R_lo, R_hi,
                         plan_r2c, plan_c2r, work_k, work_r);

            #pragma omp parallel for num_threads(N_OMP)
            for (long i=0; i<N3; i++)
                J_partial[i] += (float)(prefac * (double)sfr_smooth[i]);

            n_shells_done++;
        }
    }

    printf("  rank %d: %d shells done\n", rank, n_shells_done);
    MPI_Barrier(MPI_COMM_WORLD);

    float *J_total = NULL;
    if (rank==0) {
        J_total = fftwf_malloc(N3 * sizeof(float));
        if (!J_total) {
            fprintf(stderr, "ERROR: J_total fftwf_malloc failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memset(J_total, 0, N3*sizeof(float));
    }

    MPI_Reduce(J_partial, J_total, (int)N3,
               MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank==0) {
        /* Shell windows ring, so the sum can dip slightly below zero right
         * next to bright sources. Clip only here, once, and report it. */
        long   n_neg = 0;
        double J_min = 0.0, neg_sum = 0.0, J_sum = 0.0;
        for (long i=0; i<N3; i++) {
            J_sum += J_total[i];
            if (J_total[i] < 0.0f) {
                n_neg++;  neg_sum += J_total[i];
                if (J_total[i] < J_min) J_min = J_total[i];
                J_total[i] = 0.0f;
            }
        }
        printf("  clipped %ld negative voxels (%.3e of box), min J = %.3e, "
               "mean shift = %.3e of J_mean\n",
               n_neg, (double)n_neg/N3, J_min, (J_sum != 0.0) ? -neg_sum/J_sum : 0.0);

        double J_mean=0.0, J_max=0.0;
        for (long i=0; i<N3; i++) {
            J_mean += J_total[i];
            if (J_total[i] > J_max) J_max = J_total[i];
        }
        J_mean /= N3;

        /* Scalar sanity check — same trapezoid approach as hydrogen version */
        double J_scalar = 0.0;
        for (int n=2; n<=N_MAX; n++) {
            double frec = f_rec(n);
            if (frec==0.0) continue;
            double z_max = z_max_lyman(z_obs, n);
            if (z_max > snap_z_max()) z_max = snap_z_max();
            if (z_max <= z_obs) continue;

            double z_pts[N_SNAPS+2];
            int    np2 = 0;
            z_pts[np2++] = z_obs;
            for (int i=N_SNAPS-1; i>=0; i--) {
                double z = snap_redshifts[i];
                if (z < 0) continue;
                if (z > z_obs && z <= z_max)
                    z_pts[np2++] = z;
            }
            z_pts[np2++] = z_max;
            for (int a=1; a<np2; a++) {
                double tmp = z_pts[a];
                int b = a-1;
                while (b>=0 && z_pts[b]>tmp) { z_pts[b+1]=z_pts[b]; b--; }
                z_pts[b+1] = tmp;
            }
            for (int iz2=0; iz2<np2-1; iz2++) {
                double z1  = z_pts[iz2];
                double z2  = z_pts[iz2+1];
                double zm  = 0.5*(z1+z2);
                double dz2 = z2 - z1;
                float *box = cache_get(&cache, zm, 0);
                double sfr_mean=0.0;
                for (long k=0; k<N3; k++) sfr_mean+=box[k];
                sfr_mean /= N3;
                double nu_p = nu_emitted(zm, z_obs, n);
                double c_H  = C_CGS / H_cgm(zm);
                J_scalar += (1.0+z_obs)*(1.0+z_obs)/(4.0*M_PI_)
                             * frec * eps_mix(&st,nu_p,zm) * c_H
                             * sfr_mean * SFR_UNIT / MP_CGS * dz2;
            }
        }

        printf("\n=== Results (He II Lyman-alpha) ===\n");
        printf("  J_mean   = %.6e  [ph/s/cm^2/Hz/sr]\n", J_mean);
        printf("  J_max    = %.6e\n", J_max);
        printf("  J_scalar = %.6e\n", J_scalar);
        printf("  ratio    = %.6f  (should be ~1.0)\n",
               (J_scalar>0) ? J_mean/J_scalar : 0.0);

        printf("\nWriting %s...\n", outfile);
        write_output(outfile, J_total, N, z_obs, box_mpc, J_mean, J_scalar, &st);

        fftwf_free(J_total);
        time_t t1 = time(NULL);
        printf("Done. Elapsed: %ld s\n", (long)(t1-t0));
    }

    cache_free(&cache);
    fftwf_free(J_partial);
    fftwf_free(sfr_smooth);
    fftwf_free(work_r);
    fftwf_free(work_k);
    fftwf_destroy_plan(plan_r2c);
    fftwf_destroy_plan(plan_c2r);
    fftwf_cleanup_threads();
    free(st.freq); free(st.N_nu); free(st.logZ);
    free(zw_z); free(zw_w);

    MPI_Finalize();
    return 0;
}