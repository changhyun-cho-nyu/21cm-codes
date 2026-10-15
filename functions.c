#include "collisions.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "constants.h"
#include <stddef.h>
#include <stdlib.h>
#include <hdf5.h>

#define MAXN 200000
#define NSNAPS 709
#define M_PI 3.14159265358979323846
#define T_STAR   0.0681687        // K, h nu21 / k_B
#define F_ALPHA  0.4162           // oscillator strength
#define A10      2.86888e-15      // s^-1

#define E_CHARGE 4.80320425e-10   // statC (cgs)
#define M_ELEC      9.10938356e-28   // g
#define C_LIGHT  2.99792458e10    // cm/s

/* Function prototypes */

double J_alpha(double z, double sfrd_cell[NSNAPS]);
static inline double S_alpha(double z, double T_K, double HI_frac);
double integrate_lyman(int n, double z, double sfrd_cell[NSNAPS]);
double epsilon(double z, double sfrd, double nu);
double epsilon_b(double nu);

static double freq[MAXN];
static double Nnu[MAXN];
static int N = 0;
static int loaded = 0;
double redshift[NSNAPS];

hid_t file_id;
hid_t dset_id;
hid_t filespace;

static inline double T_gamma(double z)
{
    return 2.7255 * (1.0 + z);
}

double T_spin(double z,
              double T_K,
              double x_c,
              double x_alpha)
{
    double Tcmb = T_gamma(z);

    return (1.0 + x_c + x_alpha) /
           (1.0/Tcmb + (x_c + x_alpha)/T_K);
}

double x_c(double z,
           double T_K,
           double n_H,
           double n_e,
           double n_p)
{
    double Tcmb = T_gamma(z);

    double kHH = kappa_HH(T_K);
    double keH = kappa_eH(T_K);
    double kpH = kappa_pH(T_K);

    double coupling =
        kHH * n_H +
        keH * n_e +
        kpH * n_p;

    return (T_STAR / (A10 * Tcmb)) * coupling;
}

double x_alpha(double z,
               double T_K,
               double HI_frac,
               double sfrd_cell[NSNAPS])
{
    /* Ly-alpha flux */
    double J = J_alpha(z, sfrd_cell);

    /* Gunn-Peterson suppression */
    double S = S_alpha(z, T_K, HI_frac);

    /* CMB temperature */
    double Tcmb = T_gamma(z);

    /* prefactor */
    double prefactor =
        (16.0 * M_PI * M_PI * T_STAR * E_CHARGE * E_CHARGE * F_ALPHA) /
        (27.0 * A10 * Tcmb * M_ELEC * C_LIGHT);

    return prefactor * S * J;
}

/*
void load_sfh_txt(const char *filename,
                  double redshift[NSNAPS],
                  double sfrd_cell[NSNAPS])
{
    FILE *f = fopen(filename, "r");

    if(!f)
    {
        printf("Cannot open SFH file: %s\n", filename);
        exit(1);
    }

    int i = 0;

    while(i < NSNAPS && fscanf(f,"%lf %lf",
                               &redshift[i],
                               &sfrd_cell[i]) == 2)
    {
        i++;
    }

    fclose(f);

    if(i != NSNAPS)
    {
        printf("Warning: expected %d rows but read %d\n", NSNAPS, i);
    }
}
*/
/* ============================= */
/* load redshift                 */
/* ============================= */

void load_redshift()
{
    FILE *f = fopen("redshift.txt","r");

    if(!f)
    {
        printf("Cannot open redshift.txt\n");
        exit(1);
    }

    for(int i=0;i<NSNAPS;i++)
        fscanf(f,"%lf",&redshift[i]);

    fclose(f);
}

/* ============================= */
/* open SFRD dataset once        */
/* ============================= */

void open_sfrd_file()
{
    file_id = H5Fopen("SFRD.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);

    if(file_id < 0)
    {
        printf("Cannot open SFRD.hdf5\n");
        exit(1);
    }

    dset_id = H5Dopen2(file_id, "/SFRD", H5P_DEFAULT);
    filespace = H5Dget_space(dset_id);
}

/* ============================= */
/* close file                    */
/* ============================= */

void close_sfrd_file()
{
    H5Sclose(filespace);
    H5Dclose(dset_id);
    H5Fclose(file_id);
}

/* ============================= */
/* read SFH of a cell            */
/* ============================= */

void read_cell_sfh(int i, int j, int k, double sfrd[NSNAPS])
{
    hsize_t start[4] = {0, i, j, k};
    hsize_t count[4] = {NSNAPS, 1, 1, 1};

    hid_t fspace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);

    hsize_t memdims[1] = {NSNAPS};
    hid_t memspace = H5Screate_simple(1, memdims, NULL);

    H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, fspace, H5P_DEFAULT, sfrd);

    H5Sclose(memspace);
    H5Sclose(fspace);
}

/* ============================= */
/* j-row read (opt #2, low-mem) */
/*                               */
/* Reads one (i,j) row = N_grid  */
/* cells in one HDF5 call.       */
/* Peak memory: 2 × N × NSNAPS  */
/* × 8 bytes ≈ 7 MB (640 grid). */
/*                               */
/* Output layout:                */
/*   jrow[k * NSNAPS + s]        */
/* so &jrow[k*NSNAPS] is the     */
/* snap array for cell k.        */
/* ============================= */

void read_sfrd_jrow(int i_global, int j, int N_grid, double *jrow)
{
    hsize_t start[4] = {0,               (hsize_t)i_global,
                        (hsize_t)j,      0};
    hsize_t count[4] = {(hsize_t)NSNAPS, 1,
                        1,               (hsize_t)N_grid};

    hid_t fspace   = H5Dget_space(dset_id);
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t memspace = H5Screate_simple(4, count, NULL);

    /* buf layout: [NSNAPS][1][1][N_grid]  →  buf[s * N_grid + k] */
    double *buf = malloc((size_t)NSNAPS * N_grid * sizeof(double));

    H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, fspace, H5P_DEFAULT, buf);

    H5Sclose(memspace);
    H5Sclose(fspace);

    /* Transpose to jrow[k * NSNAPS + s] — snap-contiguous per cell */
    for (int s = 0; s < NSNAPS; s++)
        for (int k = 0; k < N_grid; k++)
            jrow[(size_t)k * NSNAPS + s] = buf[(size_t)s * N_grid + k];

    free(buf);
}

/* ============================= */
/* i-plane slab read             */
/*                               */
/* Reads one full x-plane:       */
/*   [NSNAPS, 1, N, N]           */
/* in ONE HDF5 call instead of   */
/* N separate j-row calls.       */
/*                               */
/* Output layout:                */
/*   slab[(j*N+k)*NSNAPS + s]    */
/* (spatial-first, snap last)    */
/* ============================= */

/* buf must be pre-allocated by caller: N_grid*N_grid*NSNAPS doubles.
   Passing it in avoids a 2.3 GB malloc/free on every call.          */
void read_sfrd_islab(int i_global, int N_grid, double *slab, double *buf)
{
    hsize_t start[4] = {0,               (hsize_t)i_global,
                        0,               0};
    hsize_t count[4] = {(hsize_t)NSNAPS, 1,
                        (hsize_t)N_grid, (hsize_t)N_grid};

    hid_t fspace   = H5Dget_space(dset_id);
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t memspace = H5Screate_simple(4, count, NULL);

    /* Use collective MPI-IO if file was opened with parallel access  */
    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, fspace, xfer, buf);
    H5Pclose(xfer);

    H5Sclose(memspace);
    H5Sclose(fspace);

    /* Transpose [NSNAPS][1][N][N] -> [j][k][NSNAPS]
       so slab[(j*N+k)*NSNAPS + s] is snap-contiguous per cell.      */
    size_t ncells = (size_t)N_grid * N_grid;
    for (int s = 0; s < NSNAPS; s++)
        for (size_t c = 0; c < ncells; c++)
            slab[c * NSNAPS + s] = buf[(size_t)s * ncells + c];
}

/* Open SFRD.hdf5 with parallel MPI-IO access for collective reads   */
void open_sfrd_file_parallel(MPI_Comm comm)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);
    file_id = H5Fopen("SFRD.hdf5", H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);

    if (file_id < 0) {
        printf("Cannot open SFRD.hdf5 (parallel)\n");
        exit(1);
    }

    dset_id   = H5Dopen2(file_id, "/SFRD", H5P_DEFAULT);
    filespace = H5Dget_space(dset_id);
}

/* ============================= */
/* interpolate sfrd(z)           */
/* ============================= */


double get_sfrd_from_z(double z, double sfrd[NSNAPS])
{
    int left = 0;
    int right = NSNAPS - 1;

    /* boundary checks */

    if(z >= redshift[0])
        return sfrd[0];

    if(z <= redshift[NSNAPS-1])
        return sfrd[NSNAPS-1];

    /* binary search */

    while(right - left > 1)
    {
        int mid = (left + right)/2;

        if(redshift[mid] < z)
            right = mid;
        else
            left = mid;
    }

    double z1 = redshift[left];
    double z2 = redshift[right];

    double s1 = sfrd[left];
    double s2 = sfrd[right];

    double t = (z - z1)/(z2 - z1);

    return s1 + t*(s2 - s1);
}


/*
double get_sfrd_from_z(double z, double sfrd[NSNAPS])
{
    for(int i=0;i<NSNAPS-1;i++)
    {
        if(z >= redshift[i] && z <= redshift[i+1])
        {
            double z1 = redshift[i];
            double z2 = redshift[i+1];

            double s1 = sfrd[i];
            double s2 = sfrd[i+1];

            double t = (z - z1)/(z2 - z1);

            return s1 + t*(s2 - s1);
        }
    }

    printf("Warning: z=%f outside redshift range\n", z);
    return 0.0;
}
*/

/**
 * Log-Log interpolation:
 * Interpolates log10(y) vs log10(x)
 *
 * Args:
 *   x      : frequency to evaluate
 *   x_arr  : frequency array (must be > 0 and sorted ascending)
 *   y_arr  : luminosity array (must be > 0)
 *   N      : array length
 *
 * Returns:
 *   interpolated y(x)
 */
double interp_loglog(double x,
                     const double *x_arr,
                     const double *y_arr,
                     int N)
{
    if (x <= 0.0)
        return 0.0;

    // Boundary handling
    if (x <= x_arr[0])
        return y_arr[0];

    if (x >= x_arr[N-1])
        return y_arr[N-1];

    double logx = log10(x);

    // Find interval (can optimize later)
    int i = 0;
    while (i < N-1 && x_arr[i+1] < x)
        i++;

    double logx0 = log10(x_arr[i]);
    double logx1 = log10(x_arr[i+1]);

    double logy0 = log10(y_arr[i]);
    double logy1 = log10(y_arr[i+1]);

    double t = (logx - logx0) / (logx1 - logx0);

    double logy = logy0 + t * (logy1 - logy0);

    return pow(10.0, logy);
}


/**
 * Calculates the Hubble parameter H(z) at a given redshift z.
 * Assumes a flat Lambda-CDM cosmology.
 * Uses the global 'h' parameter.
 * 
 * Args:
 *     z: Redshift
 *     
 * Returns:
 *     double: Hubble parameter H(z) in cgs units (cm^{-1})
 */
static inline double H_cgm(double z)
{
    double H_0_cgs = 100.0 * h * 3.241e-20;  // km/s/Mpc to cm^{-1}
    return H_0_cgs * sqrt(OMEGA_m * pow(1.0 + z, 3.0) + OMEGA_Lambda);
}

/**
 * Returns the recycling fraction f_rec(n) for a given Lyman series level n.
 * Values based on Pritchard & Furlanetto (2006).
 * 
 * Args:
 *     n (int): Principal quantum number (n >= 2).
 *     
 * Returns:
 *     double: The recycling fraction f_rec.
 */
static double f_rec(int n)
{
    // Array for n=0 to n=30
    static const double f_rec_values[31] = {
        0.0,      // n=0 (Invalid)
        0.0,      // n=1 (Ground state, Invalid)
        1.0000,   // n=2 (Ly-alpha)
        0.0000,   // n=3 (Ly-beta)
        0.2609,   // n=4
        0.3078,   // n=5
        0.3266,   // n=6
        0.3353,   // n=7
        0.3400,   // n=8
        0.3429,   // n=9
        0.3448,   // n=10
        0.3462,   // n=11
        0.3472,   // n=12
        0.3480,   // n=13
        0.3486,   // n=14
        0.3491,   // n=15
        0.3495,   // n=16
        0.3499,   // n=17
        0.3502,   // n=18
        0.3505,   // n=19
        0.3508,   // n=20
        0.3510,   // n=21
        0.3512,   // n=22
        0.3514,   // n=23
        0.3516,   // n=24
        0.3517,   // n=25
        0.3519,   // n=26
        0.3520,   // n=27
        0.3521,   // n=28
        0.3522,   // n=29
        0.3523    // n=30
    };
    
    // Return value based on n
    if (n >= 0 && n <= 30) {
        return f_rec_values[n];
    } else {
        // For n > 30, the value approaches 0.359 asymptotically
        return 0.359;
    }
}

/**
 * Gunn-Peterson optical depth (dimensionless)
 * 
 * Args:
 *     z: Redshift
 *     HI_frac: Neutral hydrogen fraction at redshift z
 *     
 * Returns:
 *     double: Gunn-Peterson optical depth
 */
static inline double tau_GP(double z, double HI_frac)
{
    return 3.0e5 * HI_frac * pow((1.0 + z) / 7.0, 1.5);
}

/**
 * Suppression factor due to Gunn-Peterson damping
 * 
 * Args:
 *     z: Redshift
 *     T_K: Gas temperature in Kelvin
 *     HI_frac: Neutral hydrogen fraction at redshift z
 *     
 * Returns:
 *     double: Suppression factor S_alpha
 */
static inline double S_alpha(double z, double T_K, double HI_frac)
{
    double tau = tau_GP(z, HI_frac);
    return exp(-0.803 * pow(T_K, -2.0/3.0) * pow(1.0e-6 * tau, 1.0/3.0));
}

// ============================================================
// Lyman series constants
// ============================================================
#define NU_LYMAN_LIMIT 3.289e15  // Hz (n=infinity, ionization potential)

/**
 * Calculate the maximum redshift z_max for Lyman series scattering.
 * 
 * Args:
 *     z: Observer redshift
 *     n: Principal quantum number (Lyman series level)
 *     
 * Returns:
 *     double: Maximum redshift z_max
 */
static inline double z_max(double z, int n)
{
    double n_sq_inv = 1.0 / (n * n);
    double n_plus_1_sq_inv = 1.0 / ((n + 1.0) * (n + 1.0));
    
    return (1.0 + z) * (1.0 - n_plus_1_sq_inv) / (1.0 - n_sq_inv) - 1.0;
}

/**
 * Calculate the emitted frequency for a Lyman transition.
 * 
 * Args:
 *     z_emit: Emission redshift
 *     z_base: Base/observer redshift
 *     n: Principal quantum number (Lyman series level)
 *     
 * Returns:
 *     double: Emitted frequency in Hz
 */
static inline double emitted_freq(double z_emit, double z_base, int n)
{
    double n_sq_inv = 1.0 / (n * n);
    
    return NU_LYMAN_LIMIT * (1.0 - n_sq_inv) * (1.0 + z_emit) / (1.0 + z_base);
}


void load_spectrum(const char *filename)
{
    FILE *fp = fopen(filename,"r");

    if(!fp){
        printf("Cannot open file\n");
        exit(1);
    }

    fscanf(fp,"%*[^\n]\n"); // skip header

    while(fscanf(fp,"%lf,%lf",&freq[N],&Nnu[N])==2){
        N++;
    }

    fclose(fp);

    loaded = 1;
}

double epsilon_b(double nu)
{
    if(!loaded)
        load_spectrum("spectrum_0.0001.csv");

    if(nu <= freq[0]) return Nnu[0];
    if(nu >= freq[N-1]) return Nnu[N-1];

    int left = 0;
    int right = N-1;

    while(right - left > 1)
    {
        int mid = (left + right)/2;

        if(freq[mid] > nu)
            right = mid;
        else
            left = mid;
    }

    double t = (nu - freq[left])/(freq[right]-freq[left]);

    return Nnu[left] + t*(Nnu[right]-Nnu[left]);
}

double rho_crit_0()
{
    return 3.0 * H_0_cgs * H_0_cgs / (8.0 * M_PI * G_cgs);
}

double rho_b_0()
{
    return OMEGA_b * rho_crit_0();
}

double epsilon(double z,
               double sfrd,      // g s^-1 cm^-3 (comoving)
               double nu)
{
    double m_p = 1.6726219e-24;  // g

    double Lnu = epsilon_b(nu);  // photons/Hz per baryon

    return Lnu * sfrd / m_p;
}

double integrate_lyman(int n,
                       double z,
                       double sfrd_cell[NSNAPS])
{
    double zmax = z_max(z, n);

    int Nz = 200;   // integration resolution
    double dz = (zmax - z) / Nz;

    double sum = 0.0;

    for(int i=0;i<Nz;i++)
    {
        double zp = z + (i + 0.5) * dz;   // midpoint integration

        double sfrd = get_sfrd_from_z(zp, sfrd_cell);

        double nu_emit = emitted_freq(zp, z, n);

        double eps = epsilon(zp, sfrd, nu_emit);

        double integrand = C_LIGHT * eps / H_cgm(zp);

        sum += integrand * dz;
    }

    return sum;
}

/* ============================================================
 * compute_S_alpha — expose the per-cell suppression factor
 * ============================================================ */
double compute_S_alpha(double z, double T_K, double HI_frac)
{
    return S_alpha(z, T_K, HI_frac);
}

/* ============================================================
 * precompute_Jalpha_weights
 *
 * Computes weights[NSNAPS] once per snapshot such that:
 *
 *   x_alpha(cell) = compute_S_alpha(z, T_K, HI_frac)
 *                   * dot(weights, sfrd_cell)
 *
 * Everything that depends only on z is folded in:
 *   - Lyman series loop (n=2..30) and f_rec(n)
 *   - integrate_lyman midpoint rule (Nz points per level)
 *   - epsilon_b(nu_emit) spectrum lookup
 *   - H(z') Hubble factor
 *   - redshift interpolation -> distributes onto weights[left/right]
 *   - (1+z)^2 / (4*pi) prefactor for J_alpha
 *   - x_alpha constant prefactor
 *
 * Per-cell cost drops from 5800 iterations to NSNAPS dot-product.
 * ============================================================ */
void precompute_Jalpha_weights(double z, double *weights)
{
    memset(weights, 0, NSNAPS * sizeof(double));

    const double m_p_loc = 1.6726219e-24;
    const double H_0_cgs = 100.0 * h * 3.241e-20;  /* s^-1 */
    const double OMEGA_L = 1.0 - OMEGA_m;

    double Tcmb        = T_gamma(z);
    double prefac_J    = pow(1.0 + z, 2.0) / (4.0 * M_PI);
    double xa_const    = (16.0 * M_PI * M_PI * T_STAR
                          * E_CHARGE * E_CHARGE * F_ALPHA)
                       / (27.0 * A10 * Tcmb * M_ELEC * C_LIGHT);
    double full_prefac = prefac_J * xa_const;

    for (int n = 2; n <= 30; n++)
    {
        double zmax_n = z_max(z, n);
        double frec_n = f_rec(n);

        /* Find the snapshot index range where z <= redshift[s] <= zmax_n.
           redshift[] is in descending order, so:
             s_lo = first index (smallest) where redshift[s] <= zmax_n
             s_hi = last  index (largest)  where redshift[s] >= z       */
        int s_lo = -1, s_hi = -1;
        for (int s = 0; s < NSNAPS; s++) {
            if (redshift[s] <= zmax_n && s_lo < 0) s_lo = s;
            if (redshift[s] >= z)                   s_hi = s;
        }
        if (s_lo < 0 || s_hi < 0 || s_lo > s_hi) continue;

        /* Trapezoidal rule using actual snapshot redshifts as quadrature
           points — matches the Python integration over df["redshift"].
           dz = redshift[s] - redshift[s+1] > 0  (array is descending)  */
        for (int s = s_lo; s < s_hi; s++)
        {
            double zp0 = redshift[s];      /* higher z (earlier time) */
            double zp1 = redshift[s + 1];  /* lower  z (later  time) */
            double dz  = zp0 - zp1;        /* always > 0             */

            double nu0 = emitted_freq(zp0, z, n);
            double nu1 = emitted_freq(zp1, z, n);

            double Hz0 = H_0_cgs * sqrt(OMEGA_m * pow(1.0+zp0, 3.0) + OMEGA_L);
            double Hz1 = H_0_cgs * sqrt(OMEGA_m * pow(1.0+zp1, 3.0) + OMEGA_L);

            double g0 = C_LIGHT * epsilon_b(nu0) / (m_p_loc * Hz0);
            double g1 = C_LIGHT * epsilon_b(nu1) / (m_p_loc * Hz1);

            /* Trapezoidal: 0.5*(g0*sfrd[s] + g1*sfrd[s+1])*dz
               Split by snapshot index for the dot product.        */
            weights[s]     += full_prefac * frec_n * 0.5 * g0 * dz;
            weights[s + 1] += full_prefac * frec_n * 0.5 * g1 * dz;
        }
    }
}

double J_alpha(double z, double sfrd_cell[NSNAPS])
{
    double prefactor = pow(1.0 + z, 2) / (4.0 * M_PI);

    double sum = 0.0;

    for(int n=2;n<=30;n++)
    {
        double integral = integrate_lyman(n, z, sfrd_cell);

        sum += f_rec(n) * integral;
    }

    return prefactor * sum;
}
