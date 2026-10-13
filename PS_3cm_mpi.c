/*
  PS_3cm_mpi.c

  Power spectrum of the HeII 3.46 cm brightness temperature signal.
  Based on PS_21cm_mpi.c; differences:
    - Constants : T_STAR_HE, A_10_HE, F_ALPHA_HE
    - xa_const  : denominator 9 (HeII) instead of 27 (HI)
    - x_c       : electron-HeII collisions (x_c_He)
    - T_spin    : T_spin_He
    - Signal    : f_HeII * (rho_spec / global_gas)
    - Jalpha    : /J_alpha_He dataset (stars) + /Jalpha_AGN_He dataset (AGN)
    - Prefactor : 0.53 * (Omega_b h^2 / 0.023) * sqrt(0.15/(Omega_m h^2) * (1+z))

  Author: Changhyun
*/

#include <fftw3-mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include <hdf5.h>
#include <time.h>
#include <omp.h>
#include "functions.h"

typedef long long myint;

#define M_PI      3.14159265358979323846
#define T_STAR_HE 0.41569        /* hc / (k_B * 3.46 cm)   [K]    */
#define A_10_HE   1.959e-12      /* spontaneous emission    [s^-1] */
#define F_ALPHA_HE 0.4162        /* oscillator strength             */

#define E_CHARGE  4.80320425e-10 /* statC (cgs) */
#define M_E       9.10938356e-28 /* g           */
#define C_LIGHT   2.99792458e10  /* cm/s        */

#define BINS_PS 2000

static void divide_galaxies(int numfiles, int *first, int *last);
static void read_header(int snapnum);
static void read_data(int snapnum, int first, int last,
                      ptrdiff_t local_n0, ptrdiff_t local_0_start);
static void get_offsets_and_data(void);
static double input_data(int i, int j, int k, int line);
static double input_data0(int i, int j, int k);
static double input_data1(int i, int j, int k);
static double input_data2(int i, int j, int k);
static double input_data3(int i, int j, int k);
static double input_data4(int i, int j, int k);
static double input_data5(int i, int j, int k);
static double input_data6(int i, int j, int k);
static double input_data7(int i, int j, int k);
static double get_Hz(double afac);
static void allocate_shared(void);
static void free_shared(void);
static void output_PS(int local_n0, int local_0_start,
                      fftw_complex *cout, fftw_complex *cout1,
                      char *outputfilename);
static void read_Jalpha_He_slab(const char *filename, const char *dsetname,
                                ptrdiff_t local_n0,
                                ptrdiff_t local_0_start, double *Jalpha);

static double sum_gal   = 0.0;
static double sum_diffuse = 0.0;

static double hubble, MASS, LENGTH, VEL;
static int    N, Numfiles;

static int    ThisTask, NTask;
static double totmass, totmass2, totvolume;
static double tot1mass, tot1mass2, tot1volume;
static char   inputfilename[500], inputfilename1[500],
              outputfilename[500], debugfilename[500];
static double L;
static double intensity_constant;
static double volume_cell;

static double BOXSIZE, afac;
static int    NumSubhalos;
static myint *offsets;

double *sfr, *SFRD, *T_gas, *xpos, *ypos, *zpos, *intensity, *Lum,
       *f_HII, *f_HeII, *f_HeIII,
       *rho_HI_spec, *rho_spec, *rho_gas,
       *mass, *VZ, *diffuse;
static MPI_Win win_sfr, win_SFRD, win_T_gas, win_xpos, win_ypos, win_zpos,
               win_intensity, win_Lum, win_rho_HI_spec, win_rho_spec,
               win_f_HII, win_f_HeII, win_f_HeIII, win_rho_gas,
               win_mass, win_VZ, win_diffuse;

static double *dset, *dset2, *shared_dset,
              *dset3, *dsetT, *dset4,
              *dset5, *dset6, *dset7;
static myint  dset_length_task, total_length;
static double global_gas;

static double constK  = 47229.765;
static double omega_m = 0.3089;
static double omega_b = 0.0486;
static double h       = 0.6774;

static inline double T_CMB(double z) { return 2.7255 * (1.0 + z); }

/* ==================================================================
 * HeII Lyman-series helpers
 * ================================================================== */

static inline double z_max_He(double z, int n)
{
    double ninv2   = 1.0 / ((double)n * (double)n);
    double np1inv2 = 1.0 / ((n + 1.0) * (n + 1.0));
    return (1.0 + z) * (1.0 - np1inv2) / (1.0 - ninv2) - 1.0;
}

static inline double emitted_freq_He(double z_emit, double z_base, int n)
{
    const double NU_LYM_HE = 1.31583e16;
    return NU_LYM_HE * (1.0 - 1.0 / ((double)n * (double)n))
                     * (1.0 + z_emit) / (1.0 + z_base);
}

/* ==================================================================
 * x_c_He: collisional coupling of HeII spin to kinetic temperature
 * ================================================================== */
static inline double x_c_He(double T_K, double n_e_cgs, double Tcmb)
{
    const double k_B    = 1.381e-16;
    const double m_e    = 9.10938356e-28;
    const double C      = 2.99792458e10;
    const double a_0    = 5.29177e-9;
    const double E_14eV = 2.291112e-11;

    double bar_sig = (E_14eV / (k_B * T_K)) * a_0 * a_0;
    double v_frac  = sqrt(8.0 * k_B * T_K / (M_PI * m_e * C * C));
    double kappa   = n_e_cgs * v_frac * C * bar_sig;

    return (T_STAR_HE / (A_10_HE * Tcmb)) * kappa;
}

/* ==================================================================
 * T_spin_He
 * ================================================================== */
static inline double T_spin_He(double Tcmb, double T_K,
                                double xc,   double xa)
{
    return (1.0 + xc + xa)
         / (1.0 / Tcmb + (xc + xa) / T_K);
}

/* ==================================================================
 * main
 * ================================================================== */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);

    clock_t start, end;
    double  cpu_time_used;
    start = clock();

    fftw_plan     plan, plan1;
    double       *rin,  *rin1;
    fftw_complex *cout, *cout1;
    ptrdiff_t     alloc_local, local_n0, local_0_start;

    int snapnum = atoi(argv[1]);
    strcpy(inputfilename,  argv[2]);
    strcpy(outputfilename, argv[3]);
    strcpy(debugfilename,  argv[4]);
    /* argv[5] = line (unused) */
    char jalpha_he_filename[500];
    strcpy(jalpha_he_filename, argv[6]);     /* precomputed J_alpha_He HDF5 file (stars) */
    /* [CHANGE 1] AGN Jalpha_He file */
    char jalpha_agn_he_filename[500];
    strcpy(jalpha_agn_he_filename, argv[7]); /* precomputed Jalpha_AGN_He HDF5 file */

    int line = atoi(argv[5]);

    read_header(snapnum);
    if (ThisTask == 0) {
        printf("Doing HeII 3.46 cm PS\n");
        printf("Read header successfully\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    allocate_shared();
    MPI_Barrier(MPI_COMM_WORLD);

    volume_cell = pow((BOXSIZE) / ((double)N), 3);
    L = BOXSIZE;

    int firstfile = 0;
    int lastfile  = 0;

    fftw_mpi_init();
    alloc_local = fftw_mpi_local_size_3d(N, N, N/2+1,
                                         MPI_COMM_WORLD,
                                         &local_n0, &local_0_start);

    read_data(snapnum, firstfile, lastfile, local_n0, local_0_start);

    MPI_Barrier(MPI_COMM_WORLD);
    get_offsets_and_data();

    if (ThisTask == 0)
        printf("Read data\nBOXSIZE = %g \t volume_cell = %g \t N = %d\n",
               BOXSIZE, volume_cell, N);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    rin   = fftw_alloc_real(2 * alloc_local);
    cout  = fftw_alloc_complex(alloc_local);
    rin1  = fftw_alloc_real(2 * alloc_local);
    cout1 = fftw_alloc_complex(alloc_local);

    plan  = fftw_mpi_plan_dft_r2c_3d(N, N, N, rin,  cout,
                                      MPI_COMM_WORLD, FFTW_MEASURE);
    plan1 = fftw_mpi_plan_dft_r2c_3d(N, N, N, rin1, cout1,
                                      MPI_COMM_WORLD, FFTW_MEASURE);

    /* -------------------------------------------------------------- */
    /* Precompute loop-invariant scalars                               */
    /* -------------------------------------------------------------- */
    double z         = 1.0 / afac - 1.0;
    double tcmb      = T_CMB(z);
    double oneplusz3 = pow(1.0 + z, 3);

    double length_to_cgs  = LENGTH / h;
    double volume_to_cgs  = pow(length_to_cgs, 3);
    double mass_to_cgs    = MASS / h;
    double density_to_cgs = mass_to_cgs / volume_to_cgs;
    double volume_cell_cgs= pow(length_to_cgs * BOXSIZE / (double)N, 3);

    double HYDROGEN_MASSFRAC = 0.76;
    double y_frac            = (1.0 - HYDROGEN_MASSFRAC) / (4.0 * HYDROGEN_MASSFRAC);
    double m_p               = 1.6738e-24;
    double msol_cgs          = 1.98847e33;
    double yr_sec            = 3.15576e7;

    double mass_to_nH = HYDROGEN_MASSFRAC * density_to_cgs
                      / volume_cell * oneplusz3 / m_p;

    /* xa_const for HeII: denominator 9 (not 27 as in HI) */
    double xa_const = (16.0 * M_PI * M_PI * T_STAR_HE
                       * E_CHARGE * E_CHARGE * F_ALPHA_HE)
                    / (9.0 * A_10_HE * tcmb * M_E * C_LIGHT);

    /* HeII 3.46 cm brightness temperature prefactor [mK] */
    double prefactor_He = 0.53
                        * (omega_b * h * h / 0.023)
                        * sqrt((0.15 / (omega_m * h * h)) * (1.0 + z));

    /* -------------------------------------------------------------- */
    /* Load precomputed J_alpha_He field from HDF5 (stars).           */
    /* Stored value = J_alpha_He [ph/s/cm^2/Hz/sr].                   */
    /* -------------------------------------------------------------- */
    size_t ncells_local = (size_t)local_n0 * N * N;
    double *Jalpha_field = malloc(ncells_local * sizeof(double));
    if (!Jalpha_field) {
        fprintf(stderr, "Task %d: failed to alloc Jalpha_field\n", ThisTask);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    read_Jalpha_He_slab(jalpha_he_filename, "/J_alpha_He",
                        local_n0, local_0_start, Jalpha_field);

    /* [CHANGE 2] Load AGN Jalpha_He field */
    double *Jalpha_AGN_field = malloc(ncells_local * sizeof(double));
    if (!Jalpha_AGN_field) {
        fprintf(stderr, "Task %d: failed to alloc Jalpha_AGN_field\n", ThisTask);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    read_Jalpha_He_slab(jalpha_agn_he_filename, "/Jalpha_AGN_He",
                        local_n0, local_0_start, Jalpha_AGN_field);

    /* Warm up collision tables before OpenMP region */
    (void)x_c(z, 1000.0, 1.0, 0.0, 0.0);

    if (ThisTask == 0) {
        printf("Jalpha_He (stars) loaded from %s\n", jalpha_he_filename);
        printf("Jalpha_AGN_He     loaded from %s\n", jalpha_agn_he_filename);
    }

    /* -------------------------------------------------------------- */
    /* Main loop                                                       */
    /* -------------------------------------------------------------- */
    double local_totmass  = 0.0;
    double local_totmass2 = 0.0;
    double local_volume   = 0.0;
    int    nzerocells     = 0;

    /* diagnostic accumulators */
    double d_nH        = 0.0, d_nH_min     =  1e300, d_nH_max     = -1e300;
    double d_Tgas      = 0.0, d_Tgas_min   =  1e300, d_Tgas_max   = -1e300;
    double d_fHeII     = 0.0;
    double d_Jalpha    = 0.0, d_Jalpha_min =  1e300, d_Jalpha_max = -1e300;
    double d_xa        = 0.0, d_xa_min     =  1e300, d_xa_max     = -1e300;
    double d_xc        = 0.0, d_xc_min     =  1e300, d_xc_max     = -1e300;
    double d_Tspin     = 0.0, d_Tspin_min  =  1e300, d_Tspin_max  = -1e300;
    double d_sf        = 0.0, d_sf_min     =  1e300, d_sf_max     = -1e300;
    double d_rin       = 0.0, d_rin_min    =  1e300, d_rin_max    = -1e300;
    double d_overdense = 0.0;
    long   d_ncells    = 0, d_nHeII_zero = 0, d_Ts_neg = 0;

    for (int i = 0; i < local_n0; ++i)
      for (int j = 0; j < N; ++j)
      {
        #pragma omp parallel for schedule(static) \
            reduction(+:local_totmass,local_totmass2,local_volume,nzerocells, \
                        d_nH,d_Tgas,d_fHeII,d_Jalpha,d_xa,d_xc,d_Tspin, \
                        d_sf,d_rin,d_overdense,d_ncells,d_nHeII_zero,d_Ts_neg) \
            reduction(min:d_nH_min,d_Tgas_min,d_Jalpha_min,d_xa_min, \
                          d_xc_min,d_Tspin_min,d_sf_min,d_rin_min) \
            reduction(max:d_nH_max,d_Tgas_max,d_Jalpha_max,d_xa_max, \
                          d_xc_max,d_Tspin_max,d_sf_max,d_rin_max)
        for (int k = 0; k < N; ++k)
        {
            double value1    = input_data0(local_0_start+i, j, k); /* gas mass / cell  */
            double spec_rho  = input_data5(local_0_start+i, j, k); /* SpectralDensity  */
            double T_K       = input_data2(local_0_start+i, j, k); /* gas temperature  */
            double f_HII_val = input_data4(local_0_start+i, j, k); /* HII fraction     */
            double f_HeII_val= input_data6(local_0_start+i, j, k); /* HeII fraction    */
            double f_HeIII_val=input_data7(local_0_start+i, j, k); /* HeIII fraction   */

            double num_H_cgs     = value1 * mass_to_nH;
            double num_HII_cgs   = f_HII_val  * num_H_cgs;

            double num_He_cgs    = y_frac * num_H_cgs;
            double num_HeII_cgs  = f_HeII_val * num_H_cgs;
            double num_HeIII_cgs = f_HeIII_val* num_H_cgs;
            double num_e_cgs     = num_HII_cgs + num_HeII_cgs + 2.0 * num_HeIII_cgs;
            double x_HeII        = num_HeII_cgs / num_He_cgs;

            double f_HI = 1.0 - f_HII_val;

            /* [CHANGE 3] sum stellar + AGN J_alpha_He before computing xa */
            double Jalpha = Jalpha_field[(size_t)(i*N + j)*N + k]
                          + Jalpha_AGN_field[(size_t)(i*N + j)*N + k];
            double xa     = xa_const * compute_S_alpha(z, T_K, f_HI) * Jalpha;
            double xc     = x_c_He(T_K, num_e_cgs, tcmb);
            double Tspin  = T_spin_He(tcmb, T_K, xc, xa);

            double sf      = 1.0 - tcmb / Tspin;
            double rin_val = prefactor_He * x_HeII * (spec_rho / global_gas) * sf;

            rin[(i*N + j) * (2*(N/2+1)) + k] = rin_val;

            local_totmass  += value1 * volume_cell;
            local_totmass2 += value1 * value1 * volume_cell * volume_cell;
            local_volume   += volume_cell;
            if (f_HeII_val * spec_rho <= 0.0) nzerocells++;

            /* diagnostic accumulation */
            d_nH        += num_H_cgs;
            d_Tgas      += T_K;
            d_fHeII     += f_HeII_val;
            d_Jalpha    += Jalpha;
            d_xa        += xa;
            d_xc        += xc;
            d_Tspin     += Tspin;
            d_sf        += sf;
            d_rin       += rin_val;
            d_overdense += spec_rho / global_gas;
            d_ncells++;

            if (num_H_cgs   < d_nH_min)     d_nH_min     = num_H_cgs;
            if (num_H_cgs   > d_nH_max)     d_nH_max     = num_H_cgs;
            if (T_K         < d_Tgas_min)   d_Tgas_min   = T_K;
            if (T_K         > d_Tgas_max)   d_Tgas_max   = T_K;
            if (Jalpha      < d_Jalpha_min) d_Jalpha_min = Jalpha;
            if (Jalpha      > d_Jalpha_max) d_Jalpha_max = Jalpha;
            if (xa          < d_xa_min)     d_xa_min     = xa;
            if (xa          > d_xa_max)     d_xa_max     = xa;
            if (xc          < d_xc_min)     d_xc_min     = xc;
            if (xc          > d_xc_max)     d_xc_max     = xc;
            if (Tspin       < d_Tspin_min)  d_Tspin_min  = Tspin;
            if (Tspin       > d_Tspin_max)  d_Tspin_max  = Tspin;
            if (sf          < d_sf_min)     d_sf_min     = sf;
            if (sf          > d_sf_max)     d_sf_max     = sf;
            if (rin_val     < d_rin_min)    d_rin_min    = rin_val;
            if (rin_val     > d_rin_max)    d_rin_max    = rin_val;

            if (f_HeII_val <= 0.0) d_nHeII_zero++;
            if (Tspin      <= 0.0) d_Ts_neg++;
        }
      }

    /* --- MPI reductions for diagnostics --- */
    {
        double ls[10] = { d_nH, d_Tgas, d_fHeII, d_Jalpha, d_xa,
                          d_xc, d_Tspin, d_sf, d_rin, d_overdense };
        double gs[10];
        MPI_Allreduce(ls, gs, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double g_nH_min,  g_nH_max,  g_Tgas_min, g_Tgas_max;
        double g_Ja_min,  g_Ja_max,  g_xa_min,   g_xa_max;
        double g_xc_min,  g_xc_max,  g_Ts_min,   g_Ts_max;
        double g_sf_min,  g_sf_max,  g_rin_min,  g_rin_max;
        long   g_ncells,  g_nHeII_zero, g_Ts_neg;

        MPI_Allreduce(&d_nH_min,    &g_nH_min,   1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&d_nH_max,    &g_nH_max,   1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&d_Tgas_min,  &g_Tgas_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&d_Tgas_max,  &g_Tgas_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&d_Jalpha_min,&g_Ja_min,   1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&d_Jalpha_max,&g_Ja_max,   1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&d_xa_min,    &g_xa_min,   1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&d_xa_max,    &g_xa_max,   1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&d_xc_min,    &g_xc_min,   1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&d_xc_max,    &g_xc_max,   1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&d_Tspin_min, &g_Ts_min,   1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&d_Tspin_max, &g_Ts_max,   1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&d_sf_min,    &g_sf_min,   1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&d_sf_max,    &g_sf_max,   1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&d_rin_min,   &g_rin_min,  1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&d_rin_max,   &g_rin_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&d_ncells,    &g_ncells,   1, MPI_LONG,   MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&d_nHeII_zero,&g_nHeII_zero,1,MPI_LONG,   MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&d_Ts_neg,    &g_Ts_neg,   1, MPI_LONG,   MPI_SUM, MPI_COMM_WORLD);

        if (ThisTask == 0) {
            double nc = (double)g_ncells;
            printf("\n");
            printf("========================================================\n");
            printf("  DIAGNOSTIC: rin variable audit   z = %.4f\n", z);
            printf("  Total cells = %ld  (expected %d^3 = %ld)\n",
                   g_ncells, N, (long)N*N*N);
            printf("  T_CMB            = %.6f K\n", tcmb);
            printf("  prefactor_He     = %.6e mK\n", prefactor_He);
            printf("  xa_const (HeII)  = %.6e\n", xa_const);
            printf("  global_gas       = %.6e (sim units)\n", global_gas);
            printf("--------------------------------------------------------\n");
            printf("  %-22s  %12s  %12s  %12s\n",
                   "Variable", "Mean", "Min", "Max");
            printf("  %-22s  %12s  %12s  %12s\n",
                   "----------------------","------------","------------","------------");
            printf("  %-22s  %12.4e  %12.4e  %12.4e   [cm^-3]\n",
                   "n_H (proper)",      gs[0]/nc, g_nH_min,  g_nH_max);
            printf("  %-22s  %12.4e  %12.4e  %12.4e   [K]\n",
                   "T_gas",             gs[1]/nc, g_Tgas_min,g_Tgas_max);
            printf("  %-22s  %12.6f\n",
                   "f_HeII",            gs[2]/nc);
            printf("  %-22s  %12.4e  %12.4e  %12.4e   [ph/s/cm^2/Hz/sr]\n",
                   "J_alpha_He (s+AGN)",gs[3]/nc, g_Ja_min,  g_Ja_max);
            printf("  %-22s  %12.4e  %12.4e  %12.4e\n",
                   "x_alpha (WF)",      gs[4]/nc, g_xa_min,  g_xa_max);
            printf("  %-22s  %12.4e  %12.4e  %12.4e\n",
                   "x_c (coll.)",       gs[5]/nc, g_xc_min,  g_xc_max);
            printf("  %-22s  %12.4e  %12.4e  %12.4e   [K]\n",
                   "T_spin_He",         gs[6]/nc, g_Ts_min,  g_Ts_max);
            printf("  %-22s  %12.6f  %12.6f  %12.6f\n",
                   "sf = 1-Tcmb/Ts",   gs[7]/nc, g_sf_min,  g_sf_max);
            printf("  %-22s  %12.6f\n",
                   "spec_rho/global_gas", gs[9]/nc);
            printf("  %-22s  %12.4e  %12.4e  %12.4e   [mK]\n",
                   "rin (T_b_He)",      gs[8]/nc, g_rin_min, g_rin_max);
            printf("--------------------------------------------------------\n");
            printf("  Cells with f_HeII <= 0 : %ld  (%.2f%%)\n",
                   g_nHeII_zero, 100.0*g_nHeII_zero/nc);
            printf("  Cells with T_spin <= 0 : %ld  (%.2f%%)\n",
                   g_Ts_neg,     100.0*g_Ts_neg/nc);
            printf("========================================================\n\n");
        }
    }

    /* [CHANGE 4] free both Jalpha fields */
    free(Jalpha_field);
    free(Jalpha_AGN_field);

    MPI_Allreduce(&local_totmass,  &totmass,   1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_totmass2, &totmass2,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_volume,   &totvolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int Nzerocells = 0;
    MPI_Allreduce(&nzerocells, &Nzerocells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    memcpy(rin1, rin, 2 * alloc_local * sizeof(double));

    tot1mass   = totmass;
    tot1mass2  = totmass2;
    tot1volume = totvolume;

    if (ThisTask == 0)
        printf("Volume weighted mean = %g | Old way = %g | Zero cells = %d | non-zero vol = %g | vol = %g\n",
               totmass/totvolume, totmass/pow(BOXSIZE,3),
               Nzerocells, totvolume, pow(BOXSIZE,3));

    /* -------------------------------------------------------------- */
    /* Debug HDF5 output                                               */
    /* -------------------------------------------------------------- */
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file = H5Fcreate(debugfilename, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    hsize_t dims[3]  = {N, N, N};
    hid_t   dspace   = H5Screate_simple(3, dims, NULL);
    hid_t   dset_out = H5Dcreate(file, "/rin", H5T_NATIVE_FLOAT,
                                  dspace, H5P_DEFAULT);

    hsize_t hstart[3] = {local_0_start, 0, 0};
    hsize_t count[3]  = {local_n0, N, N};
    hid_t   filespace = H5Dget_space(dset_out);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, hstart, NULL, count, NULL);
    hid_t memspace = H5Screate_simple(3, count, NULL);

    size_t total_cells = (size_t)local_n0 * N * N;
    float *buffer = malloc(total_cells * sizeof(float));
    for (int i = 0; i < local_n0; i++)
      for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
          buffer[(i*N+j)*N+k] =
              (float)(rin[(i*N+j)*(2*(N/2+1))+k] / volume_cell);

    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    H5Dwrite(dset_out, H5T_NATIVE_FLOAT, memspace, filespace, xfer, buffer);
    free(buffer);

    H5Pclose(xfer);
    H5Sclose(memspace); H5Sclose(filespace);
    H5Dclose(dset_out); H5Sclose(dspace);
    H5Fclose(file);

    /* -------------------------------------------------------------- */
    /* FFT + power spectrum                                            */
    /* -------------------------------------------------------------- */
    fftw_execute(plan);
    fftw_execute(plan1);

    if (ThisTask == 0) printf("Finished fftw\n");

    output_PS(local_n0, local_0_start, cout, cout1, outputfilename);

    if (ThisTask == 0) printf("Finished power spectrum\n");

    fftw_free(rin1); fftw_free(cout1);
    fftw_free(rin);  fftw_free(cout);
    fftw_destroy_plan(plan);
    fftw_destroy_plan(plan1);

    MPI_Barrier(MPI_COMM_WORLD);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    if (ThisTask == 0)
        printf("Time = %g seconds\n", cpu_time_used);

    MPI_Finalize();
    return 0;
}


/*#############################################################*/

static void get_offsets_and_data(void)
{
    offsets = (myint *)malloc(NTask * sizeof(myint));
    MPI_Allgather(&dset_length_task, sizeof(myint), MPI_BYTE,
                  offsets, sizeof(myint), MPI_BYTE, MPI_COMM_WORLD);
    myint temp = 0;
    for (int i = 0; i < NTask; i++) {
        myint temp1 = offsets[i];
        offsets[i] += temp;
        temp += temp1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double local_gas = 0.0;
    global_gas = 0.0;
    for (myint i = 0; i < dset_length_task; i++) {
        dset2[i] *= volume_cell;  /* density -> mass (sim units) */
        local_gas += dset2[i];
        dset[i]   /= 0.76;
    }
    MPI_Allreduce(&local_gas, &global_gas, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    global_gas /= (L * L * L);

    if (ThisTask == 0)
        printf("global_gas = %g\n", global_gas);

    for (myint j = 0; j < dset_length_task; j++) {
        myint index = (ThisTask == 0) ? j : offsets[ThisTask-1] + j;
        rho_HI_spec[index] = dset[j]  * volume_cell;
        rho_gas[index]     = dset2[j];
        f_HII[index]       = dset3[j];
        T_gas[index]       = dsetT[j];
        SFRD[index]        = dset4[j];
        rho_spec[index]    = dset5[j] * volume_cell;
        f_HeII[index]      = dset6[j];
        f_HeIII[index]     = dset7[j];
    }

    free(dset);  free(dset2); free(dset3); free(dset4);
    free(dsetT); free(dset5); free(dset6); free(dset7);
}

/*#############################################################*/

static void grid_data(void)
{
    double dx = BOXSIZE / ((double)N);
    if (ThisTask == 0) {
        memset(intensity, 0, N*N*N*sizeof(double));
        memset(SFRD,      0, N*N*N*sizeof(double));
        for (int index = 0; index < NumSubhalos; index++) {
            zpos[index] += VZ[index] / (get_Hz(afac) * afac);
            if (zpos[index] >= BOXSIZE) zpos[index] -= BOXSIZE;
            if (zpos[index] < 0.0)     zpos[index] += BOXSIZE;
            int i = (int)(xpos[index]/dx);
            int j = (int)(ypos[index]/dx);
            int k = (int)(zpos[index]/dx);
            intensity[i*N*N + j*N + k] += Lum[index];
            sum_gal += Lum[index];
        }
    }
    MPI_Bcast(&sum_gal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

static double get_Hz(double afac)
{
    double redz = 1.0/afac - 1.0;
    return 67.74 * sqrt(0.3089*pow((1.0+redz),3) + (1.0-0.3089));
}

/*#############################################################*/

static void divide_galaxies(int numfiles, int *first, int *last)
{
    int rat = numfiles / NTask;
    int rem = numfiles % NTask;
    if (ThisTask < NTask - rem) {
        *first = ThisTask * rat;
        *last  = (ThisTask+1) * rat - 1;
    } else {
        int newnumfiles  = numfiles - (NTask-rem)*rat;
        int newnumtask   = rem;
        int newfirsttask = NTask - rem;
        int newfirstfile = newfirsttask * rat;
        int newtask      = ThisTask - newfirsttask;
        rat    = newnumfiles / newnumtask;
        *first = newfirstfile + newtask * rat;
        *last  = newfirstfile + (newtask+1) * rat - 1;
    }
}

/*#############################################################*/

static void read_header(int snapnum)
{
    char fname[500];
    if (ThisTask == 0) {
        sprintf(fname, "%s/All/All_%03d.hdf5", inputfilename, snapnum);
        hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) { printf("FATAL: cannot open %s\n", fname);
                        MPI_Abort(MPI_COMM_WORLD, 1); }
        hid_t grp = H5Gopen(file, "/Header");
        if (grp  < 0) { printf("FATAL: /Header not found in %s\n", fname);
                        MPI_Abort(MPI_COMM_WORLD, 2); }
        H5Aread(H5Aopen(grp,"NumPixels",               H5P_DEFAULT),H5T_NATIVE_INT,    &N);
        H5Aread(H5Aopen(grp,"BoxSize",                 H5P_DEFAULT),H5T_NATIVE_DOUBLE, &BOXSIZE);
        H5Aread(H5Aopen(grp,"Time",                    H5P_DEFAULT),H5T_NATIVE_DOUBLE, &afac);
        H5Aread(H5Aopen(grp,"HubbleParam",             H5P_DEFAULT),H5T_NATIVE_DOUBLE, &hubble);
        H5Aread(H5Aopen(grp,"UnitMass_in_g",           H5P_DEFAULT),H5T_NATIVE_DOUBLE, &MASS);
        H5Aread(H5Aopen(grp,"UnitLength_in_cm",        H5P_DEFAULT),H5T_NATIVE_DOUBLE, &LENGTH);
        H5Aread(H5Aopen(grp,"UnitVelocity_in_cm_per_s",H5P_DEFAULT),H5T_NATIVE_DOUBLE, &VEL);
        H5Gclose(grp); H5Fclose(file);
        NumSubhalos = 0; Numfiles = 1;
    }
    MPI_Bcast(&N,           1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&BOXSIZE,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&afac,        1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&hubble,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MASS,        1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&LENGTH,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&VEL,         1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NumSubhalos, 1, MPI_INT,    0, MPI_COMM_WORLD);
}

/*#############################################################*/

static void read_single_field(const char *field, double *dest, int snapnum,
                               ptrdiff_t local_n0, ptrdiff_t local_0_start,
                               float *buffer, myint length_vector)
{
    char fname[500];
    sprintf(fname, "%s/%s/%s_%03d.hdf5", inputfilename, field, field, snapnum);

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);

    char dsetname[256];
    sprintf(dsetname, "/%s", field);
    hid_t dataset   = H5Dopen(hdf5_file, dsetname);
    hid_t dataspace = H5Dget_space(dataset);

    hsize_t start[3] = {local_0_start, 0, 0};
    hsize_t count[3] = {local_n0, N, N};
    hid_t   memspace = H5Screate_simple(3, count, NULL);
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, xfer, buffer);
    H5Pclose(xfer);

    for (myint j = 0; j < length_vector; j++)
        dest[j] = buffer[j];

    H5Sclose(memspace); H5Sclose(dataspace);
    H5Dclose(dataset);  H5Fclose(hdf5_file);
}

static void read_data(int snapnum, int first, int last,
                      ptrdiff_t local_n0, ptrdiff_t local_0_start)
{
    myint length_vector  = local_n0 * N * N;
    dset_length_task     = length_vector;

    dset  = malloc(length_vector*sizeof(double)); /* SpectralDensityHI         */
    dset2 = malloc(length_vector*sizeof(double)); /* Density                   */
    dset3 = malloc(length_vector*sizeof(double)); /* HII_VolumeFraction        */
    dsetT = malloc(length_vector*sizeof(double)); /* TemperatureVolumeWeighted */
    dset4 = malloc(length_vector*sizeof(double)); /* StarFormationRate         */
    dset5 = malloc(length_vector*sizeof(double)); /* SpectralDensity           */
    dset6 = malloc(length_vector*sizeof(double)); /* HeII_VolumeFraction       */
    dset7 = malloc(length_vector*sizeof(double)); /* HeIII_VolumeFraction      */

    float *buffer = malloc(length_vector*sizeof(float));

    read_single_field("SpectralDensityHI",         dset,  snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field("Density",                   dset2, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field("HII_VolumeFraction",        dset3, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field("TemperatureVolumeWeighted", dsetT, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field("StarFormationRate",         dset4, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field("SpectralDensity",           dset5, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field("HeII_VolumeFraction",       dset6, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field("HeIII_VolumeFraction",      dset7, snapnum, local_n0, local_0_start, buffer, length_vector);

    free(buffer);
}

/*#############################################################*/

static void allocate_shared(void)
{
    if (ThisTask == 0) {
        MPI_Win_allocate_shared(N*N*N*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&rho_HI_spec,&win_rho_HI_spec);
        MPI_Win_allocate_shared(N*N*N*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&rho_spec,   &win_rho_spec);
        MPI_Win_allocate_shared(N*N*N*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&rho_gas,    &win_rho_gas);
        MPI_Win_allocate_shared(N*N*N*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&f_HII,      &win_f_HII);
        MPI_Win_allocate_shared(N*N*N*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&f_HeII,     &win_f_HeII);
        MPI_Win_allocate_shared(N*N*N*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&f_HeIII,    &win_f_HeIII);
        MPI_Win_allocate_shared(N*N*N*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&SFRD,       &win_SFRD);
        MPI_Win_allocate_shared(N*N*N*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&T_gas,      &win_T_gas);
        MPI_Win_allocate_shared(N*N*N*sizeof(double),sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&diffuse,    &win_diffuse);
    } else {
        int disp_unit; MPI_Aint size;
        MPI_Win_allocate_shared(0,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&rho_HI_spec,&win_rho_HI_spec);
        MPI_Win_shared_query(win_rho_HI_spec,0,&size,&disp_unit,&rho_HI_spec);
        MPI_Win_allocate_shared(0,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&rho_gas,  &win_rho_gas);
        MPI_Win_shared_query(win_rho_gas,    0,&size,&disp_unit,&rho_gas);
        MPI_Win_allocate_shared(0,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&rho_spec, &win_rho_spec);
        MPI_Win_shared_query(win_rho_spec,   0,&size,&disp_unit,&rho_spec);
        MPI_Win_allocate_shared(0,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&f_HII,    &win_f_HII);
        MPI_Win_shared_query(win_f_HII,      0,&size,&disp_unit,&f_HII);
        MPI_Win_allocate_shared(0,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&f_HeII,   &win_f_HeII);
        MPI_Win_shared_query(win_f_HeII,     0,&size,&disp_unit,&f_HeII);
        MPI_Win_allocate_shared(0,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&f_HeIII,  &win_f_HeIII);
        MPI_Win_shared_query(win_f_HeIII,    0,&size,&disp_unit,&f_HeIII);
        MPI_Win_allocate_shared(0,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&SFRD,     &win_SFRD);
        MPI_Win_shared_query(win_SFRD,       0,&size,&disp_unit,&SFRD);
        MPI_Win_allocate_shared(0,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&T_gas,    &win_T_gas);
        MPI_Win_shared_query(win_T_gas,      0,&size,&disp_unit,&T_gas);
        MPI_Win_allocate_shared(0,sizeof(double),MPI_INFO_NULL,MPI_COMM_WORLD,&diffuse,  &win_diffuse);
        MPI_Win_shared_query(win_diffuse,    0,&size,&disp_unit,&diffuse);
    }
}

static void free_shared(void)
{
    MPI_Win_free(&win_rho_HI_spec); MPI_Win_free(&win_rho_spec);
    MPI_Win_free(&win_rho_gas);     MPI_Win_free(&win_f_HII);
    MPI_Win_free(&win_f_HeII);      MPI_Win_free(&win_f_HeIII);
    MPI_Win_free(&win_T_gas);       MPI_Win_free(&win_SFRD);
    MPI_Win_free(&win_diffuse);
}

static double input_data(int i, int j, int k, int line)
{
    if (line == 3)
        return intensity[i*N*N+j*N+k]*(1.0 - sum_diffuse*constK/sum_gal)
             + diffuse[i*N*N+j*N+k]*constK;
    else
        return intensity[i*N*N+j*N+k];
}

static double input_data1(int i,int j,int k){return rho_HI_spec[i*N*N+j*N+k];}
static double input_data0(int i,int j,int k){return rho_gas[i*N*N+j*N+k];}
static double input_data4(int i,int j,int k){return f_HII[i*N*N+j*N+k];}
static double input_data2(int i,int j,int k){return T_gas[i*N*N+j*N+k];}
static double input_data3(int i,int j,int k){return SFRD[i*N*N+j*N+k];}
static double input_data5(int i,int j,int k){return rho_spec[i*N*N+j*N+k];}
static double input_data6(int i,int j,int k){return f_HeII[i*N*N+j*N+k];}
static double input_data7(int i,int j,int k){return f_HeIII[i*N*N+j*N+k];}

/*#############################################################*/

/* Reads a Jalpha_He slab from a precomputed HDF5 file. */
static void read_Jalpha_He_slab(const char *filename, const char *dsetname,
                                ptrdiff_t local_n0, ptrdiff_t local_0_start,
                                double *Jalpha)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);

    if (file < 0) {
        fprintf(stderr, "FATAL: cannot open Jalpha_He file %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    hid_t dset   = H5Dopen(file, dsetname);
    hid_t dspace = H5Dget_space(dset);

    hsize_t start[3] = {local_0_start, 0, 0};
    hsize_t count[3] = {local_n0,      N, N};
    H5Sselect_hyperslab(dspace, H5S_SELECT_SET, start, NULL, count, NULL);
    hid_t memspace = H5Screate_simple(3, count, NULL);

    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, dspace, xfer, Jalpha);
    H5Pclose(xfer);

    H5Sclose(memspace); H5Sclose(dspace);
    H5Dclose(dset);     H5Fclose(file);
}

/*#############################################################*/

void output_PS(int local_n0, int local_0_start,
               fftw_complex *cout, fftw_complex *cout1,
               char *outputfilename)
{
    FILE *fd = fopen(outputfilename, "w");
    L = BOXSIZE;

    double normalization = 1.0 / sqrt(totmass * tot1mass);
    double kfx = 2.0*M_PI/L, kfy = 2.0*M_PI/L, kfz = 2.0*M_PI/L;
    double Kmin    = 2.0*M_PI/L;
    double Kmax    = Kmin * N / 2;
    double binfac  = BINS_PS / (log(Kmax) - log(Kmin));
    double log_Kmin= log(Kmin);

    double *SumPower   = (double *)calloc(BINS_PS, sizeof(double));
    double *CountModes = (double *)calloc(BINS_PS, sizeof(double));

    double *k1arr = malloc(local_n0 * sizeof(double));
    double *k2arr = malloc(N        * sizeof(double));
    for (int ii = 0; ii < local_n0; ii++) {
        int II = (2*(ii+local_0_start) < N) ? (ii+local_0_start) : N-(ii+local_0_start);
        k1arr[ii] = (double)II * kfx;
    }
    for (int jj = 0; jj < N; jj++) {
        int JJ = (2*jj < N) ? jj : N-jj;
        k2arr[jj] = (double)JJ * kfy;
    }

    for (int i = 0; i < local_n0; i++) {
        double k1 = k1arr[i];
        for (int j = 0; j < N; j++) {
            double k2 = k2arr[j];
            for (int k = 0; k < N/2+1; k++) {
                double k3  = (double)k * kfz;
                double sqk = sqrt(k1*k1 + k2*k2 + k3*k3);
                if (sqk >= Kmin && sqk < Kmax) {
                    int    idx = (i*N+j)*(N/2+1)+k;
                    double mag = (cout[idx][0]*cout1[idx][0]
                                + cout[idx][1]*cout1[idx][1])
                               * normalization * normalization;
                    int bin = (int)(log(sqk/Kmin) * binfac);
                    SumPower[bin]   += mag;
                    CountModes[bin] += 1.0;
                }
            }
        }
    }
    free(k1arr); free(k2arr);

    MPI_Allreduce(MPI_IN_PLACE, SumPower,   BINS_PS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, CountModes, BINS_PS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (ThisTask == 0) {
        fprintf(fd, "%g\n",    1.0/afac - 1.0);
        fprintf(fd, "%d\n",    BINS_PS);
        fprintf(fd, "%g\n",    sqrt(totmass*tot1mass));
        fprintf(fd, "%g\n",    totvolume);
        fprintf(fd, "%15lld\n",(myint)N*N*N);
        for (int i = 0; i < BINS_PS; i++) {
            double Kval = exp((i + 0.5) / binfac + log_Kmin);
            if (CountModes[i] > 0) SumPower[i] /= CountModes[i];
            double delta     = 4*M_PI*pow(Kval,3)/pow(2*M_PI/L,3) * SumPower[i];
            double shotnoise = 4*M_PI*pow(Kval,3)/pow(2*M_PI/L,3)
                             * (sqrt(totmass2*tot1mass2)/totmass/tot1mass);
            fprintf(fd, "%g %g %g %g %g\n",
                    Kval, delta, SumPower[i], (double)CountModes[i], shotnoise);
        }
        fclose(fd);
    }
    free(SumPower); free(CountModes);
}
