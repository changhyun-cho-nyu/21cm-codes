/*
 * compute_Jalpha.c
 * ================
 * Compute spatially-resolved Lyman-alpha intensity J_alpha(x, z_obs)
 * using hybrid MPI + OpenMP parallelism.
 *
 * Parallelism strategy
 * --------------------
 * MPI  : each rank handles a disjoint subset of Lyman-n lines.
 *         At the end, MPI_Allreduce sums all partial J_alpha boxes.
 * OMP  : each rank uses N_OMP threads for FFT and voxel loops.
 *
 * Calling convention (matches your SLURM script)
 * -----------------------------------------------
 * mpirun -np 16 ./compute_Jalpha_H  <snap>  <sfr_dir>  <output_hdf5>
 *
 *   snap        : integer snapshot number (e.g. 177)
 *   sfr_dir     : directory containing StarFormationRate_NNN.hdf5 files
 *   output_hdf5 : full path for output file (e.g. /path/Jalpha_H_177.hdf5)
 *
 * BPASS spectrum file is hardcoded below — edit SPECTRUM_FILE.
 *
 * Compile
 * -------
 * mpicc -O3 -march=native -fopenmp compute_Jalpha.c -o compute_Jalpha_H \
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

#define SPECTRUM_FILE \
    "/orcd/data/mvogelsb/004/ozier/ZOOM_XL/sharedData/" \
    "bpass_spectra_bin_chab100_v2.2.1.hdf5"

#define N_SNAPS      709     /* total number of snapshots                  */
#define N_MAX        30      /* max Lyman series index                     */
#define N_SHELLS     200     /* radial shells per Lyman-n line             */
/* Stellar metallicity history Z(z) (replaces the fixed METALLICITY).
 * Two columns "redshift  Z"; make it from Z_above4p75.csv + Z_below4p75.csv.
 * EDIT this path. */
#define Z_HISTORY_FILE \
    "/orcd/data/mvogelsb/004/chcho/cell_by_cell/hydrogen/Z_history.txt"
#define N_OMP        4       /* OpenMP threads per MPI rank (= cpus-per-task) */

/* =========================================================================
 * Physical constants (CGS)
 * ========================================================================= */

#define C_CGS      2.9979e10
#define MSUN_CGS   1.989e33
#define YR_CGS     3.15576e7
#define MP_CGS     1.6738e-24
#define CMPC_CGS   3.0857e24
#define NU_LL      3.289e15
#define M_PI_      3.14159265358979323846

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
    return 0.3590;
}

/* =========================================================================
 * Cosmological utilities
 * ========================================================================= */

static double H_cgm(double z) {
    return H_0_CGS * sqrt(OMEGA_M * pow(1.0+z, 3.0) + OMEGA_L);
}

static double z_max_lyman(double z_obs, int n) {
    return (1.0+z_obs) * (1.0 - pow(n+1.0,-2.0))
                       / (1.0 - pow((double)n,-2.0)) - 1.0;
}

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
 * HDF5 helper: read a scalar double attribute
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
 * Load snapshot redshifts from HDF5 headers
 * Also reads geometry from the first file found.
 * ========================================================================= */

static void build_snap_path(const char *sfr_dir, int snap, char *out) {
    snprintf(out, 2048, "%s/All/All_%03d.hdf5", sfr_dir, snap);
}

/* Return the highest valid redshift: first non-sentinel entry (array is
 * ordered highest-z first, so snap 0 is normally z~29). Falls back to
 * subsequent snaps if snap 0 is missing. */
static double snap_z_max(void) {
    for (int i = 0; i < N_SNAPS; i++)
        if (snap_redshifts[i] >= 0.0) return snap_redshifts[i];
    return -1.0;
}

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

static void load_snap_metadata(const char *sfr_dir, const char *sfr_dir2, int rank) {
    char path[2048];
    for (int i = 0; i < N_SNAPS; i++) {
        build_snap_path(sfr_dir, i, path);

        hid_t fid = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        snap_src[i] = 1;
        if (fid < 0) {
            /* Try second directory */
            if (sfr_dir2 != NULL) {
                build_snap_path(sfr_dir2, i, path);
                fid = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
                snap_src[i] = 2;
            }
            if (fid < 0) {
                snap_redshifts[i] = -1.0;   /* mark missing */
                snap_src[i] = 0;
                continue;
            }
        }
        hid_t gid = H5Gopen(fid, "Header", H5P_DEFAULT);
        snap_redshifts[i] = read_attr_double(gid, "Redshift");

        /* Read geometry once from first valid file found */
        if (SFR_UNIT == 0.0) {
            double boxsize  = read_attr_double(gid, "BoxSize");
            double hubble   = read_attr_double(gid, "HubbleParam");
            double unitlen  = read_attr_double(gid, "UnitLength_in_cm");
            N_PIX    = read_attr_int(gid, "NumPixels");
            BOX_CMPC = boxsize / hubble / 1000.0;
            double box_cm = boxsize * unitlen / hubble;
            double cell_cm = box_cm / N_PIX;
            SFR_UNIT = (MSUN_CGS / YR_CGS) / (cell_cm*cell_cm*cell_cm);
            if (rank == 0)
                printf("  Box=%.1f cMpc  N=%d  SFR_UNIT=%.4e g/s/cm3/vox\n",
                       BOX_CMPC, N_PIX, SFR_UNIT);
        }
        H5Gclose(gid);
        H5Fclose(fid);
    }
    report_snapshots(rank, sfr_dir, sfr_dir2);
}

/* =========================================================================
 * Read one SFR box from disk, convert units in-place
 * box must be pre-allocated: float[N_PIX^3]
 * ========================================================================= */

static char last_sfr_path[2048];   /* file actually opened by the last read_sfr_box() */

static void read_sfr_box(const char *sfr_dir, const char *sfr_dir2,
                          int snap, float *box) {
    char path[2048];
    build_snap_path(sfr_dir, snap, path);

    hid_t fid = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) {
        if (sfr_dir2 != NULL)
            build_snap_path(sfr_dir2, snap, path);
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

    /*
     * Leave values in raw simulation units [Msun/yr/voxel].
     * SFR_UNIT (approx 4.5e-48) is folded into the prefactor as a double
     * to avoid float underflow (SFR_UNIT < float_subnormal approx 1.4e-45).
     * Only apply a non-negativity floor here.
     */
    long N3 = (long)N_PIX * N_PIX * N_PIX;
    #pragma omp parallel for num_threads(N_OMP)
    for (long i = 0; i < N3; i++) {
        if (box[i] < 0.0f) box[i] = 0.0f;
    }
}

/* =========================================================================
 * Snapshot cache
 * Holds the two bracketing snapshots (lo=higher z, hi=lower z).
 * Re-reads from disk only when z_prime crosses a snapshot boundary.
 * ========================================================================= */

typedef struct {
    int    snap_lo;          /* higher-z snapshot index */
    int    snap_hi;          /* lower-z  snapshot index */
    float *box_lo;
    float *box_hi;
    float *box_interp;
    long   N3;
    const char *sfr_dir;
    const char *sfr_dir2;
} SnapCache;

static void cache_init(SnapCache *c, long N3, const char *sfr_dir,
                        const char *sfr_dir2) {
    c->snap_lo   = -9999;
    c->snap_hi   = -9999;
    c->N3        = N3;
    c->sfr_dir   = sfr_dir;
    c->sfr_dir2  = sfr_dir2;
    c->box_lo    = fftwf_malloc(N3 * sizeof(float));
    c->box_hi    = fftwf_malloc(N3 * sizeof(float));
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

/*
 * Find bracketing snapshots for z_prime.
 * snap_redshifts is DECREASING: index 0 = highest z.
 * We want: snap_redshifts[slo] >= z_prime > snap_redshifts[shi]
 */
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
        read_sfr_box(c->sfr_dir, c->sfr_dir2, slo, c->box_lo);
        printf("  [rank %d cache] read snap %03d  z=%.4f  <- %s\n",
               rank, slo, snap_redshifts[slo], last_sfr_path);
        c->snap_lo = slo;
    }
    if (shi != c->snap_hi) {
        if (snap_redshifts[shi] < 0) {
            fprintf(stderr, "ERROR: snap %d not found in any directory\n", shi);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        read_sfr_box(c->sfr_dir, c->sfr_dir2, shi, c->box_hi);
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

/*
 * eps_b: photon yield at frequency nu and metallicity Z.
 * Linear in nu, linear in log Z (bilinear). Outside the tables the edge value
 * is used: Z < Z_min -> Z_min, Z > Z_max -> Z_max, and likewise in nu.
 * Same scheme as N_nu_at() in the Python notebook.
 */
static double eps_b(const SpecTable *st, double nu, double Z) {
    int    Nf = st->Nfreq;
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

    int    Nz = st->N_met;
    double lz = (Z > 0.0) ? log10(Z) : st->logZ[0];
    if (lz < st->logZ[0])    lz = st->logZ[0];
    if (lz > st->logZ[Nz-1]) lz = st->logZ[Nz-1];
    int j = 0;
    while (j < Nz-2 && lz > st->logZ[j+1]) j++;
    double wz = (Nz > 1) ? (lz - st->logZ[j]) / (st->logZ[j+1] - st->logZ[j]) : 0.0;

    const double *a = st->N_nu + (long)j*Nf;
    const double *b = (Nz > 1) ? a + Nf : a;
    double va = a[lo]*(1.0-wf) + a[hi]*wf;
    double vb = b[lo]*(1.0-wf) + b[hi]*wf;
    return (1.0-wz)*va + wz*vb;
}

/* =========================================================================
 * Stellar metallicity history Z(z)
 * Text file, two columns "redshift  Z" (lines starting with # ignored).
 * Z is clamped to the BPASS grid, then interpolated linearly in log Z;
 * outside the tabulated redshift range the end value is held.
 * Same as Z_of_z() in the Python notebook.
 * ========================================================================= */

static int     NZH     = 0;
static double *zh_z    = NULL;   /* ascending redshift */
static double *zh_logZ = NULL;

static void load_Z_history(const SpecTable *st, int rank) {
    FILE *fp = fopen(Z_HISTORY_FILE, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open Z history file %s\n", Z_HISTORY_FILE);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int    cap  = 256;
    double Zmin = pow(10.0, st->logZ[0]);
    double Zmax = pow(10.0, st->logZ[st->N_met-1]);
    zh_z    = malloc(cap * sizeof(double));
    zh_logZ = malloc(cap * sizeof(double));
    char line[1024];
    while (fgets(line, sizeof line, fp)) {
        if (line[0] == '#') continue;
        double z, Z;
        if (sscanf(line, "%lf %lf", &z, &Z) != 2) continue;
        if (NZH == cap) {
            cap *= 2;
            zh_z    = realloc(zh_z,    cap * sizeof(double));
            zh_logZ = realloc(zh_logZ, cap * sizeof(double));
        }
        if (!(Z > Zmin)) Z = Zmin;          /* also catches Z = 0 and NaN */
        if (Z > Zmax)    Z = Zmax;
        zh_z[NZH] = z;  zh_logZ[NZH] = log10(Z);  NZH++;
    }
    fclose(fp);
    if (NZH < 2) {
        fprintf(stderr, "ERROR: %s has %d usable rows (need >= 2)\n", Z_HISTORY_FILE, NZH);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int a = 1; a < NZH; a++) {                    /* sort by redshift */
        double tz = zh_z[a], tl = zh_logZ[a];
        int b = a-1;
        while (b >= 0 && zh_z[b] > tz) { zh_z[b+1] = zh_z[b]; zh_logZ[b+1] = zh_logZ[b]; b--; }
        zh_z[b+1] = tz;  zh_logZ[b+1] = tl;
    }
    if (rank == 0)
        printf("  Z(z): %d rows from %s, z = %.3f .. %.3f, Z = %.3e .. %.3e (after clamping)\n",
               NZH, Z_HISTORY_FILE, zh_z[0], zh_z[NZH-1],
               pow(10.0, zh_logZ[NZH-1]), pow(10.0, zh_logZ[0]));
}

static double Z_of_z(double z) {
    if (z <= zh_z[0])     return pow(10.0, zh_logZ[0]);
    if (z >= zh_z[NZH-1]) return pow(10.0, zh_logZ[NZH-1]);
    int lo = 0, hi = NZH-1;
    while (hi-lo > 1) {
        int mid = (lo+hi)/2;
        if (zh_z[mid] <= z) lo = mid; else hi = mid;
    }
    double w = (z - zh_z[lo]) / (zh_z[hi] - zh_z[lo]);
    return pow(10.0, (1.0-w)*zh_logZ[lo] + w*zh_logZ[hi]);
}

/* =========================================================================
 * FFT top-hat smoothing
 * ========================================================================= */

static void smooth_tophat(const float *sfr_in, float *sfr_out,
                           int N, double box_mpc, double R_mpc,
                           fftwf_plan plan_r2c, fftwf_plan plan_c2r,
                           fftwf_complex *work_k, float *work_r) {
    double dx = box_mpc / N;
    long   N3 = (long)N*N*N;
    int    Nk = N/2+1;

    if (R_mpc <= sqrt(3.0)/2.0*dx) {
        memcpy(sfr_out, sfr_in, N3*sizeof(float));
        return;
    }

    memcpy(work_r, sfr_in, N3*sizeof(float));
    fftwf_execute(plan_r2c);

    float norm = 1.0f/(float)N3;

    #pragma omp parallel for collapse(3) num_threads(N_OMP)
    for (int ix=0; ix<N; ix++)
    for (int iy=0; iy<N; iy++)
    for (int iz=0; iz<Nk; iz++) {
        int fx = (ix<=N/2) ? ix : ix-N;
        int fy = (iy<=N/2) ? iy : iy-N;
        double kx = 2.0*M_PI_*fx/box_mpc;
        double ky = 2.0*M_PI_*fy/box_mpc;
        double kz = 2.0*M_PI_*iz/box_mpc;
        double kR = sqrt(kx*kx+ky*ky+kz*kz) * R_mpc;
        double W  = (kR<1e-6) ? 1.0 :
                    3.0*(sin(kR)-kR*cos(kR))/(kR*kR*kR);
        long idx = ((long)ix*N+iy)*Nk+iz;
        work_k[idx][0] *= (float)(W*norm);
        work_k[idx][1] *= (float)(W*norm);
    }

    fftwf_execute(plan_c2r);

    #pragma omp parallel for num_threads(N_OMP)
    for (long i=0; i<N3; i++)
        sfr_out[i] = (work_r[i]>0.0f) ? work_r[i] : 0.0f;
}

/* =========================================================================
 * HDF5 output
 * ========================================================================= */

static void write_output(const char *outfile, const float *J,
                          int N, double z_obs, double box_mpc,
                          double J_mean, double J_scalar) {
    hid_t fid = H5Fcreate(outfile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[3] = {(hsize_t)N,(hsize_t)N,(hsize_t)N};
    hid_t sid = H5Screate_simple(3,dims,NULL);

    hid_t pid = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk[3]={64,64,64};
    H5Pset_chunk(pid,3,chunk);
    H5Pset_deflate(pid,4);

    hid_t did = H5Dcreate(fid,"J_alpha",H5T_NATIVE_FLOAT,
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
    WA("Z_star_obs",  Z_of_z(z_obs))
    #undef WA

    /* Units string attribute */
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
 * Each MPI rank handles a subset of Lyman-n lines.
 * Result is MPI_Reduced (summed) onto rank 0.
 * ========================================================================= */

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (argc < 5) {
        if (rank==0)
            fprintf(stderr,
                "Usage: %s <snap> <sfr_dir> <sfr_dir2> <output.hdf5>\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    int         snap_target = atoi(argv[1]);
    const char *sfr_dir     = argv[2];
    const char *sfr_dir2    = argv[3];
    const char *outfile     = argv[4];

    time_t t0 = time(NULL);
    if (rank==0) {
        printf("=== compute_Jalpha: snap %03d ===\n", snap_target);
        printf("MPI ranks=%d  OMP threads/rank=%d\n", nranks, N_OMP);
        printf("SFR dir : %s\n", sfr_dir);
        printf("SFR dir2: %s\n", sfr_dir2);
        printf("Output  : %s\n", outfile);
    }

    /* ------------------------------------------------------------------
     * Step 1: Load snapshot metadata (all ranks read, cheap)
     * ------------------------------------------------------------------ */
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    if (rank==0) printf("\nLoading snapshot metadata...\n");
    load_snap_metadata(sfr_dir, sfr_dir2, rank);
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

    int    N  = N_PIX;
    long   N3 = (long)N*N*N;
    double box_mpc = BOX_CMPC;

    /* ------------------------------------------------------------------
     * Step 2: Load spectrum table (all ranks, cheap)
     * ------------------------------------------------------------------ */
    if (rank==0) printf("\nLoading BPASS spectrum...\n");
    SpecTable st = load_spectrum(rank);
    load_Z_history(&st, rank);
    if (rank==0) printf("  Z(z_obs=%.3f) = %.4e\n", z_obs, Z_of_z(z_obs));
    MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------------
     * Step 3: FFTW setup (per-rank, each rank has own plans + buffers)
     * ------------------------------------------------------------------ */
    fftwf_init_threads();
    fftwf_plan_with_nthreads(N_OMP);

    float         *work_r   = fftwf_malloc(N3 * sizeof(float));
    fftwf_complex *work_k   = fftwf_malloc((long)N*N*(N/2+1)*sizeof(fftwf_complex));
    float         *sfr_smooth = fftwf_malloc(N3 * sizeof(float));
    if (!work_r || !work_k || !sfr_smooth) {
        fprintf(stderr, "ERROR: FFT workspace fftwf_malloc failed (N=%d, %.2f GB per buffer)\n",
                N, N3*4.0/1e9);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n_arr[3] = {N, N, N};
    if (rank==0) printf("Planning FFTs...\n");
    fftwf_plan plan_r2c = fftwf_plan_dft_r2c(3,n_arr,work_r,work_k,FFTW_MEASURE);
    fftwf_plan plan_c2r = fftwf_plan_dft_c2r(3,n_arr,work_k,work_r,FFTW_MEASURE);
    if (rank==0) printf("FFT plans ready.\n");

    /* ------------------------------------------------------------------
     * Step 4: Snapshot cache (per-rank)
     * ------------------------------------------------------------------ */
    SnapCache cache;
    cache_init(&cache, N3, sfr_dir, sfr_dir2);

    /* ------------------------------------------------------------------
     * Step 5: Partial J_alpha box for this rank
     * ------------------------------------------------------------------ */
    float *J_partial = fftwf_malloc(N3 * sizeof(float));
    if (!J_partial) {
        fprintf(stderr, "ERROR: J_partial fftwf_malloc failed (%.2f GB)\n", N3*4.0/1e9);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(J_partial, 0, N3*sizeof(float));

    /* ------------------------------------------------------------------
     * Step 6: Loop over Lyman-n lines assigned to this rank
     * Distribute n = 2..N_MAX round-robin across ranks.
     * n=3 (Lyb, f_rec=0) is skipped but still counted for load balance.
     * ------------------------------------------------------------------ */
    if (rank==0) printf("\nComputing J_alpha partial boxes...\n");

    int n_shells_done = 0;

    for (int n = 2; n <= N_MAX; n++) {

        /* Round-robin assignment: rank 0 gets n=2,2+nranks,...
           rank 1 gets n=3,3+nranks,...  etc.                  */
        if ((n - 2) % nranks != rank) continue;

        double frec = f_rec(n);
        if (frec == 0.0) continue;

        double z_max = z_max_lyman(z_obs, n);
        if (z_max > snap_z_max()) z_max = snap_z_max();
        if (z_max <= z_obs)       continue;

        double R_max = comoving_Mpc(z_obs, z_max, 500);
        double R_min = box_mpc / N;
        if (R_max <= R_min) continue;

        printf("  rank %d: n=%2d  f_rec=%.4f  z_max=%.3f  R_max=%.1f Mpc\n",
               rank, n, frec, z_max, R_max);

        /* Log-spaced shell edges */
        double log_Rmin = log(R_min);
        double log_Rmax = log(R_max);
        double dlogR    = (log_Rmax - log_Rmin) / N_SHELLS;

        for (int is = 0; is < N_SHELLS; is++) {
            double R_lo  = exp(log_Rmin + is       * dlogR);
            double R_hi  = exp(log_Rmin + (is+1.0) * dlogR);
            double R_mid = 0.5*(R_lo + R_hi);

            /* Emission redshift at R_mid */
            double z_prime = R_to_z(z_obs, z_max, R_mid);

            /* Shell dz */
            double z_r_lo = R_to_z(z_obs, z_max, R_lo);
            double z_r_hi = R_to_z(z_obs, z_max, R_hi);
            double dz     = z_r_hi - z_r_lo;

            /* Scalar prefactor */
            double nu_p  = nu_emitted(z_prime, z_obs, n);
            double eb    = eps_b(&st, nu_p, Z_of_z(z_prime));   /* Z at the EMISSION redshift */
            double c_H   = C_CGS / H_cgm(z_prime);
            /* SFR_UNIT (≈4.5e-48) is kept as double here to avoid
             * float underflow (SFR_UNIT < float subnormal ≈ 1.4e-45).
             * sfr_smooth is in raw [Msun/yr/vox], so multiply by SFR_UNIT
             * inside the prefactor to keep everything in double. */
            double prefac = (1.0+z_obs)*(1.0+z_obs) / (4.0*M_PI_)
                            * frec * eb * c_H / MP_CGS * SFR_UNIT * dz;

            /* Get interpolated SFR box at z_prime */
            float *sfr_z = cache_get(&cache, z_prime, rank);

            /* FFT-smooth at R_mid */
            smooth_tophat(sfr_z, sfr_smooth,
                          N, box_mpc, R_mid,
                          plan_r2c, plan_c2r, work_k, work_r);

            /* Accumulate — prefac is double, cast product to float.
             * With SFR_UNIT folded in, prefac ≈ 1e-10, sfr_smooth ≈ 1-1000
             * [Msun/yr/vox], so each term ≈ 1e-10 to 1e-7: safely in float range. */
            #pragma omp parallel for num_threads(N_OMP)
            for (long i=0; i<N3; i++)
                J_partial[i] += (float)(prefac * (double)sfr_smooth[i]);

            n_shells_done++;
        }
    }

    printf("  rank %d: %d shells done\n", rank, n_shells_done);
    MPI_Barrier(MPI_COMM_WORLD);

    /* ------------------------------------------------------------------
     * Step 7: MPI_Reduce — sum partial boxes onto rank 0
     * ------------------------------------------------------------------ */
    float *J_total = NULL;
    if (rank==0) {
        J_total = fftwf_malloc(N3 * sizeof(float));
        if (!J_total) {
            fprintf(stderr, "ERROR: J_total fftwf_malloc failed (%.2f GB)\n", N3*4.0/1e9);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memset(J_total, 0, N3*sizeof(float));
    }

    MPI_Reduce(J_partial, J_total, (int)N3,
               MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* ------------------------------------------------------------------
     * Step 8: Stats + scalar sanity check (rank 0 only)
     * ------------------------------------------------------------------ */
    if (rank==0) {
        double J_mean=0.0, J_max=0.0;
        for (long i=0; i<N3; i++) {
            J_mean += J_total[i];
            if (J_total[i] > J_max) J_max = J_total[i];
        }
        J_mean /= N3;

        /* Scalar J_alpha from box mean SFR (quick, no FFT needed) */
        double J_scalar = 0.0;
        for (int n=2; n<=N_MAX; n++) {
            double frec = f_rec(n);
            if (frec==0.0) continue;
            double z_max = z_max_lyman(z_obs, n);
            if (z_max > snap_z_max()) z_max = snap_z_max();
            if (z_max <= z_obs) continue;
            /* Use snapshot grid as integration points */
            /* Build sorted list of z points in [z_obs, z_max] */
            /* Use same approach as Python notebook: trapezoid over snapshot grid */
            double z_pts[N_SNAPS+2];
            int    np2 = 0;
            z_pts[np2++] = z_obs;
            /* snap_redshifts is DECREASING, so iterate forward = increasing z */
            /* We want z in (z_obs, z_max], increasing order */
            /* Collect valid snaps, then sort */
            for (int i=N_SNAPS-1; i>=0; i--) {
                double z = snap_redshifts[i];
                if (z < 0) continue;  /* missing snap sentinel */
                if (z > z_obs && z <= z_max)
                    z_pts[np2++] = z;
            }
            z_pts[np2++] = z_max;
            /* Sort z_pts ascending (simple insertion sort, small array) */
            for (int a=1; a<np2; a++) {
                double tmp = z_pts[a];
                int b = a-1;
                while (b>=0 && z_pts[b]>tmp) { z_pts[b+1]=z_pts[b]; b--; }
                z_pts[b+1] = tmp;
            }
            /* Trapezoid integration */
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
                /* sfr_mean is in raw [Msun/yr/vox]; multiply by SFR_UNIT
                 * to convert to g/s/cm^3 before dividing by MP_CGS. */
                J_scalar += (1.0+z_obs)*(1.0+z_obs)/(4.0*M_PI_)
                             * frec * eps_b(&st,nu_p,Z_of_z(zm)) * c_H
                             * sfr_mean * SFR_UNIT / MP_CGS * dz2;
            }
        }

        printf("\n=== Results ===\n");
        printf("  J_mean   = %.6e  [ph/s/cm^2/Hz/sr]\n", J_mean);
        printf("  J_max    = %.6e\n", J_max);
        printf("  J_scalar = %.6e  (expected ~2e-9)\n", J_scalar);
        printf("  ratio    = %.6f  (should be ~1.0)\n",
               (J_scalar>0) ? J_mean/J_scalar : 0.0);

        /* Write output */
        printf("\nWriting %s...\n", outfile);
        write_output(outfile, J_total, N, z_obs, box_mpc, J_mean, J_scalar);

        fftwf_free(J_total);
        time_t t1 = time(NULL);
        printf("Done. Elapsed: %ld s\n", (long)(t1-t0));
    }

    /* Cleanup */
    cache_free(&cache);
    fftwf_free(J_partial);
    fftwf_free(sfr_smooth);
    fftwf_free(work_r);
    fftwf_free(work_k);
    fftwf_destroy_plan(plan_r2c);
    fftwf_destroy_plan(plan_c2r);
    fftwf_cleanup_threads();
    free(st.freq); free(st.N_nu); free(st.logZ);
    free(zh_z); free(zh_logZ);

    MPI_Finalize();
    return 0;
}