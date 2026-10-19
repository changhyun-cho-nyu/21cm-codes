/*A MPI paralellised program to read
  render snapshots, store them in
  shared memory and then perform
  a MPI 3D FFTW. Currenlty works within
  a single node.
  Need to couple FFTW MPI and HDF5 libraries
  Author: Rahul Kannan
  Date:   12/12/2020

  Improved MPI support
  Edited by Changhyun

  Performance optimisations:
  - Collective MPI-IO for field reads (#5)
  - SFRD slab read before loop (#2)
  - All loop-invariant constants precomputed (#3)
  - Duplicate second loop replaced with memcpy (#1)
  - Collision log-tables precomputed in collisions.c (#4)
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
#include <unistd.h>
#include "functions.h"

/* ------------------------------------------------------------------
 * Field choices (compile-time):
 *   TS_VOLUME_WEIGHTED = 0 (default): the spin temperature uses the
 *     mass-weighted render fields HII_Fraction, Temperature,
 *     HeII_Fraction, HeIII_Fraction (consistent with the signal, which
 *     is weighted by H I / He II mass); = 1: the volume-weighted fields
 *     HII_VolumeFraction, TemperatureVolumeWeighted, He*_VolumeFraction.
 * A field is read from <dir>/<field>/<field>_NNN.hdf5 and, if that file
 * does not exist, from <dir>/All/All_NNN.hdf5.
 * The He fractions are per H nucleus in the renders (max = y = 0.079);
 * the code aborts if a He fraction exceeds 1.05 y.
 * ------------------------------------------------------------------ */
#ifndef TS_VOLUME_WEIGHTED
#define TS_VOLUME_WEIGHTED 0
#endif
#if TS_VOLUME_WEIGHTED
#define F_HII_FIELD   "HII_VolumeFraction"
#define T_FIELD       "TemperatureVolumeWeighted"
#define F_HEII_FIELD  "HeII_VolumeFraction"
#define F_HEIII_FIELD "HeIII_VolumeFraction"
#else
#define F_HII_FIELD   "HII_Fraction"
#define T_FIELD       "Temperature"
#define F_HEII_FIELD  "HeII_Fraction"
#define F_HEIII_FIELD "HeIII_Fraction"
#endif

/* H_REDSHIFT_SPACE = 1 (default): the signal is the render's SpectralDensityHI, the H I density in
 *   redshift space along the 3rd axis (each gas cell shifted by its own velocity in the render: exact).
 * = 0: real space, 0.76 Density (1 - HII_Fraction), with the mass-weighted HII_Fraction whatever
 *   TS_VOLUME_WEIGHTED is (it is the H I mass). The spin factor is real space in both cases. */
#ifndef H_REDSHIFT_SPACE
#define H_REDSHIFT_SPACE 1
#endif

/* TK_REFERENCE (off by default): the no-heating reference. In cells with x_HII < TK_REF_XTHRESH
 * (default 1e-3, the neutral-gas criterion of Zier+26 App. B) the gas temperature used for the
 * spin temperature (x_c, S_alpha, T_S) is replaced by the adiabatic temperature of unheated gas.
 * Other cells keep the simulated T. x_HII is the field the spin temperature uses (F_HII_FIELD).
 * TK_REF_ANCHOR = 1 (default; decided 2026-10-19): anchored to Lumina's own gas, pixel by pixel, at
 *   render snapshot TK_REF_SNAP0 (default 0, z0 = 29.43):
 *     T_ref(p, z) = T(p, z0) [(1+z)/(1+z0)]^2 [rho(p, z)/rho(p, z0)]^(2/3)   (comoving densities),
 *   adiabatic expansion and the pixel's density change; T_ref = T at z0. The z0 render is read from
 *   the same render directory, or from $TK_REF_DIR0 if set.
 * TK_REF_ANCHOR = 0: the analytic law T_ref = T_CMB,0 (1+z)^2/(1+z_dec) (1+delta)^(2/3), z_dec = 124.8
 *   (Zier+26 Eq. 27). Its temperature lies below Lumina's unheated gas at z ~ 25-30 (2026-10-19). */
#ifdef TK_REFERENCE
#ifndef TK_REF_XTHRESH
#define TK_REF_XTHRESH 1e-3
#endif
#ifndef TK_REF_ANCHOR
#define TK_REF_ANCHOR 1
#endif
#ifndef TK_REF_SNAP0
#define TK_REF_SNAP0 0
#endif
#define TK_REF_ZDEC  124.8
#define TK_REF_TCMB0 2.7255
#endif

typedef long long myint ;

#define M_PI 3.14159265358979323846
#define T_STAR   0.0681687        // K, h nu21 / k_B (nu21 = 1420.405751768 MHz)
#define F_ALPHA  0.4162           // oscillator strength
#define A10      2.86888e-15      // s^-1, 21 cm Einstein coefficient
#define NU21     1.420405751768e9 // Hz
#define H_PLANCK 6.62607015e-27   // erg s
#define K_BOLTZ  1.380649e-16     // erg/K
#define G_GRAV   6.6743e-8        // cm^3 g^-1 s^-2
#define MPC_CM   3.0856776e24     // cm
#define M_H      1.6735575e-24    // g, hydrogen atom

#define E_CHARGE 4.80320425e-10   // statC (cgs)
#define M_ELEC      9.10938356e-28   // g
#define C_LIGHT  2.99792458e10    // cm/s

#define BINS_PS 2000

static void divide_galaxies(int numfiles, int *first, int *last) ;
static void read_header(int snapnum) ;
static void read_data(int snapnum, int first, int last, ptrdiff_t local_n0, ptrdiff_t local_0_start);
static void calculate_cell_intensity(int first, int last, int line) ;
static double get_intensity(double logsfr, int line) ;
static void grid_data(void) ;
static void get_offsets_and_data(void) ;
static double input_data(int i, int j, int k, int line) ;
static double input_data0(int i, int j, int k) ;
static double input_data1(int i, int j, int k) ;
static double input_data2(int i, int j, int k) ;
static double input_data3(int i, int j, int k) ;
static double input_data4(int i, int j, int k) ;
static double input_data5(int i, int j, int k) ;
static double input_data6(int i, int j, int k) ;
static double input_data7(int i, int j, int k) ;
static double get_Hz(double afac) ;


static double sum_gal = 0.0 ;
static double sum_diffuse = 0.0 ;

static void allocate_shared(void) ;
static void free_shared(void) ;

static void output_PS(int local_n0, int local_0_start, fftw_complex *cout, fftw_complex *cout1, char *outputfilename) ;
static void read_Jalpha_slab(const char *filename, const char *dsetname,
                              ptrdiff_t local_n0, ptrdiff_t local_0_start, double *Jalpha) ;


static double hubble, MASS, LENGTH, VEL ;
static int N, Numfiles ;

static int ThisTask, NTask ;
static double totmass, totmass2, totvolume ;
static double tot1mass, tot1mass2, tot1volume ;
static char inputfilename[500], inputfilename1[500], outputfilename[500], debugfilename[500];
static double L ;
static double intensity_constant ;
static double volume_cell ;

static double BOXSIZE, afac ;
static int NumSubhalos ;
static myint *offsets ;

double *sfr, *SFRD, *T_gas, *xpos, *ypos, *zpos, *intensity, *Lum,
       *f_HII, *f_HeII, *f_HeIII,
       *rho_HI_spec, *rho_spec, *rho_gas,
       *mass, *VZ, *diffuse;
static MPI_Win win_sfr, win_SFRD, win_T_gas, win_xpos, win_ypos, win_zpos, win_intensity, win_Lum, win_rho_HI_spec, win_rho_spec, win_f_HII, win_f_HeII, win_f_HeIII, win_rho_gas, win_mass, win_VZ, win_diffuse ;

static double *dset, *dset2, *shared_dset,
              *dset3, *dsetT, *dset4,
              *dset5, *dset6, *dset7;
static myint dset_length_task, total_length ;
static double global_gas ;
#if defined(TK_REFERENCE) && TK_REF_ANCHOR == 1
static double *ref_T0 = NULL, *ref_rho0 = NULL, ref_z0 = 0.0;   /* this task's slab of the z0 render */
#endif

static double constK  = 47229.765 ;
/* Lumina (Planck 2018). Omega_m, Omega_b are replaced by the render header's Omega0 /
 * OmegaBaryon when present; h is always the header's HubbleParam (read_header). */
static double omega_m = 0.30964144 ;
static double omega_b = 0.048975 ;
static double omega_l = 0.6902672 ;   /* header OmegaLambda when present; as the J_alpha codes */
static double h       = 0.6766 ;

static inline double T_CMB(double z)
{
    return 2.7255 * (1.0 + z);
}


int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  clock_t start, end;
  double cpu_time_used ;

  start = clock() ;

  fftw_plan plan, plan1;
  double *rin, *rin1;
  fftw_complex *cout, *cout1;
  ptrdiff_t alloc_local, local_n0, local_0_start ;

  if (argc < 8) {
      if (ThisTask == 0)
          fprintf(stderr, "Usage: %s <snap> <render_dir> <PS.out> <Tb_map.hdf5> <line> <Jalpha_H.hdf5> <Jalpha_AGN_H.hdf5>\n", argv[0]);
      MPI_Finalize(); return 1;
  }
  int snapnum = atoi(argv[1]);
  strcpy(inputfilename,  argv[2]);
  strcpy(outputfilename, argv[3]);
  strcpy(debugfilename,  argv[4]);
  /* argv[5] = line (unused in current code)  */
  char jalpha_filename[500];
  strcpy(jalpha_filename, argv[6]);     /* precomputed Jalpha_H HDF5 file (stars) */
  /* [CHANGE 1] AGN Jalpha file */
  char jalpha_agn_filename[500];
  strcpy(jalpha_agn_filename, argv[7]); /* precomputed Jalpha_AGN_H HDF5 file */

  int line = atoi(argv[5]);

  read_header(snapnum) ;
  if(ThisTask==0)
    {
      printf("Doing LIM PS\n") ;
      printf("Read header succesfully \n") ;
    }

  MPI_Barrier(MPI_COMM_WORLD) ;
  allocate_shared() ;
  MPI_Barrier(MPI_COMM_WORLD) ;


  volume_cell = pow((BOXSIZE)/((double) (N)), 3) ; // (sim length units)^3
  L = BOXSIZE ;

  int firstfile = 0;
  int lastfile  = 0;

  fftw_mpi_init();

  alloc_local = fftw_mpi_local_size_3d(N, N, N/2+1, MPI_COMM_WORLD, &local_n0, &local_0_start);

  read_data(snapnum, firstfile, lastfile, local_n0, local_0_start);

  MPI_Barrier(MPI_COMM_WORLD) ;
  get_offsets_and_data() ;

  if(ThisTask==0)
    {
      printf("Read data\n") ;
      printf("BOXSIZE = %g \t volume_cell = %g \t N = %d \n", BOXSIZE, volume_cell, N) ;
    }

  MPI_Barrier(MPI_COMM_WORLD) ;
  MPI_Barrier(MPI_COMM_WORLD) ;

  rin  = fftw_alloc_real(2 * alloc_local);
  cout = fftw_alloc_complex(alloc_local);

  rin1  = fftw_alloc_real(2 * alloc_local);
  cout1 = fftw_alloc_complex(alloc_local);

  plan  = fftw_mpi_plan_dft_r2c_3d(N, N, N, rin,  cout,  MPI_COMM_WORLD, FFTW_MEASURE);
  plan1 = fftw_mpi_plan_dft_r2c_3d(N, N, N, rin1, cout1, MPI_COMM_WORLD, FFTW_MEASURE);

  /* -------------------------------------------------------
   * Opt #3: precompute all loop-invariant scalars once
   * ------------------------------------------------------- */
  double z = 1.0/afac - 1.0;
  double tcmb = T_CMB(z);
  double oneplusz3 = pow(1.0 + z, 3);

  double length_to_cgs   = LENGTH / h;
  double volume_to_cgs   = pow(length_to_cgs, 3);
  double mass_to_cgs     = MASS / h;
  double density_to_cgs  = mass_to_cgs / volume_to_cgs;
  double velocity_to_cgs = VEL;
  double volume_cell_cgs = pow(length_to_cgs * BOXSIZE / (double)N, 3);

  double HYDROGEN_MASSFRAC = 0.76;
  double y_frac            = (1.0 - HYDROGEN_MASSFRAC) / (4.0 * HYDROGEN_MASSFRAC);
  double m_p               = M_H;
  double msol_cgs          = 1.98847e33;
  double yr_sec            = 3.15576e7;


  double xa_const          = (16.0 * M_PI * M_PI * T_STAR
                              * E_CHARGE * E_CHARGE * F_ALPHA)
                              / (27.0 * A10 * tcmb * M_ELEC * C_LIGHT);

  /* Combined factor: value1 (gas mass per cell, sim units)
     → proper hydrogen number density (cm^-3)               */
  double mass_to_nH = HYDROGEN_MASSFRAC * density_to_cgs / volume_cell * oneplusz3 / m_p;

  /* 21cm prefactor [mK], constant for this snapshot: delta T_b = prefactor x_HI (1+delta) (1 - T_gamma/T_S),
     from the optically thin tau T_S = 3 c^3 h_P A10 n_HI / (32 pi k_B nu21^2 H(z)) at the mean H density,
     with H(z) including Lambda (the textbook 27 mK form assumes matter domination). */
  double H0_cgs   = 100.0 * h * 1.0e5 / MPC_CM;
  double Hz_cgs   = H0_cgs * sqrt(omega_m * oneplusz3 + omega_l);
  double nH_mean  = HYDROGEN_MASSFRAC * omega_b * 3.0 * H0_cgs * H0_cgs / (8.0 * M_PI * G_GRAV) * oneplusz3 / M_H;
  double prefactor_21cm = 1.0e3 * 3.0 * pow(C_LIGHT, 3) * H_PLANCK * A10 * nH_mean
                        / (32.0 * M_PI * K_BOLTZ * NU21 * NU21 * Hz_cgs * (1.0 + z));

  /* -------------------------------------------------------
   * Load precomputed Jalpha_H field from HDF5.
   * Stored value = J_alpha  (xa_const, S_alpha not included).
   * No SFRD.hdf5 reads, no dot product — just one field read.
   * ------------------------------------------------------- */
  size_t ncells_local = (size_t)local_n0 * N * N;
  double *Jalpha_field = malloc(ncells_local * sizeof(double));
  if (!Jalpha_field) {
      fprintf(stderr, "Task %d: failed to alloc Jalpha_field\n", ThisTask);
      MPI_Abort(MPI_COMM_WORLD, 1);
  }
  read_Jalpha_slab(jalpha_filename, "/J_alpha", local_n0, local_0_start, Jalpha_field);

  /* [CHANGE 2] Load AGN Jalpha_H field */
  double *Jalpha_AGN_field = malloc(ncells_local * sizeof(double));
  if (!Jalpha_AGN_field) {
      fprintf(stderr, "Task %d: failed to alloc Jalpha_AGN_field\n", ThisTask);
      MPI_Abort(MPI_COMM_WORLD, 1);
  }
  read_Jalpha_slab(jalpha_agn_filename, "/Jalpha_AGN_H", local_n0, local_0_start, Jalpha_AGN_field);

  /* Warm up collision tables before OpenMP region */
  (void)x_c(z, 1000.0, 1.0, 0.0, 0.0);

  if(ThisTask==0) {
      printf("Jalpha_H (stars) loaded from %s\n", jalpha_filename);
      printf("Jalpha_AGN_H     loaded from %s\n", jalpha_agn_filename);
  }

  /* -------------------------------------------------------
   * Main loop — single pass (opt #1: second loop removed,
   * rin1 is filled by memcpy after this loop)
   * ------------------------------------------------------- */
  double local_totmass  = 0.0 ;
  double local_totmass2 = 0.0 ;
  double local_volume   = 0.0 ;
  int    nzerocells     = 0 ;

  /* diagnostic accumulators */
  double d_nH        = 0.0, d_nH_min    =  1e300, d_nH_max    = -1e300;
  double d_Tgas      = 0.0, d_Tgas_min  =  1e300, d_Tgas_max  = -1e300;
  double d_fHI       = 0.0;
  double d_Jalpha    = 0.0, d_Jalpha_min=  1e300, d_Jalpha_max= -1e300;
  double d_xa        = 0.0, d_xa_min    =  1e300, d_xa_max    = -1e300;
  double d_xc        = 0.0, d_xc_min    =  1e300, d_xc_max    = -1e300;
  double d_Tspin     = 0.0, d_Tspin_min =  1e300, d_Tspin_max = -1e300;
  double d_sf        = 0.0, d_sf_min    =  1e300, d_sf_max    = -1e300;
  double d_rin       = 0.0, d_rin_min   =  1e300, d_rin_max   = -1e300;
  double d_overdense = 0.0;
  long   d_ncells    = 0, d_nHI_zero = 0, d_Ts_neg = 0;
  long   d_ref_cells = 0;                       /* TK_REFERENCE diagnostics (zero otherwise) */
  double d_ref_HI    = 0.0, d_all_HI = 0.0;
#ifdef TK_REFERENCE
#if TK_REF_ANCHOR == 1
  const double tk_ref_scale = ((1.0 + z) / (1.0 + ref_z0)) * ((1.0 + z) / (1.0 + ref_z0));   /* (1+z)^2 expansion */
  if (ThisTask == 0)
    printf("TK_REFERENCE: T = T(p, z0 = %.4f) x %.6f x [rho(p)/rho(p, z0)]^(2/3) [per pixel, anchor snapshot %d] where %s < %g\n",
           ref_z0, tk_ref_scale, TK_REF_SNAP0, F_HII_FIELD, (double)TK_REF_XTHRESH);
#else
  const double tk_ref_mean    = TK_REF_TCMB0 * (1.0 + z) * (1.0 + z) / (1.0 + TK_REF_ZDEC); /* K, at mean density */
  const double mean_mass_cell = global_gas * volume_cell;                                   /* sim units */
  if (ThisTask == 0)
    printf("TK_REFERENCE: T = %.4f K (1+delta)^(2/3) [Zier+26 Eq. 27, z_dec = %.1f] where %s < %g\n",
           tk_ref_mean, TK_REF_ZDEC, F_HII_FIELD, (double)TK_REF_XTHRESH);
#endif
#endif

  for (int i = 0; i < local_n0; ++i)
    for (int j = 0; j < N; ++j)
      {
        /* Opt #5: OpenMP over k — all cells are independent. */
        #pragma omp parallel for schedule(static) \
            reduction(+:local_totmass,local_totmass2,local_volume,nzerocells, \
                        d_nH,d_Tgas,d_fHI,d_Jalpha,d_xa,d_xc,d_Tspin, \
                        d_sf,d_rin,d_overdense,d_ncells,d_nHI_zero,d_Ts_neg, \
                        d_ref_cells,d_ref_HI,d_all_HI) \
            reduction(min:d_nH_min,d_Tgas_min,d_Jalpha_min,d_xa_min, \
                          d_xc_min,d_Tspin_min,d_sf_min,d_rin_min) \
            reduction(max:d_nH_max,d_Tgas_max,d_Jalpha_max,d_xa_max, \
                          d_xc_max,d_Tspin_max,d_sf_max,d_rin_max)
        for (int k = 0; k < N; ++k)
          {
            double value  = input_data1(local_0_start+i, j, k); // SpectralDensityHI
            double value5 = input_data5(local_0_start+i, j, k); // SpectralDensity
            double value1 = input_data0(local_0_start+i, j, k); // gas mass per cell

            double num_H_cgs = value1 * mass_to_nH;

            double value2 = input_data2(local_0_start+i, j, k); // Gas temperature
            double value4 = input_data4(local_0_start+i, j, k); // f_HII
#ifdef TK_REFERENCE
            if (value4 < TK_REF_XTHRESH) {             /* adiabatic reference temperature (see TK_REFERENCE) */
#if TK_REF_ANCHOR == 1
                myint li = ((myint)i * N + j) * N + k;     /* this task's slab (global plane local_0_start + i) */
                value2 = ref_T0[li] * tk_ref_scale * cbrt(pow(value1 / (ref_rho0[li] * volume_cell), 2));
#else
                value2 = tk_ref_mean * cbrt(pow(value1 / mean_mass_cell, 2));
#endif
                d_ref_cells++; d_ref_HI += value1 * (1.0 - value4);
            }
            d_all_HI += value1 * (1.0 - value4);
#endif

            double f_HI        = 1.0 - value4;
            double num_HII_cgs = value4 * num_H_cgs;
            double num_He_cgs  = y_frac * num_H_cgs;

            double value6 = input_data6(local_0_start+i, j, k); // f_HeII
            double value7 = input_data7(local_0_start+i, j, k); // f_HeIII

            double num_HeII_cgs  = value6 * num_H_cgs;
            double num_HeIII_cgs = value7 * num_H_cgs;

            double num_p_cgs = num_HII_cgs;
            double num_e_cgs = num_HII_cgs + num_HeII_cgs + 2.0 * num_HeIII_cgs;

            /* [CHANGE 3] sum stellar + AGN J_alpha before computing xa */
            double Jalpha = Jalpha_field[(size_t)(i*N + j)*N + k]
                          + Jalpha_AGN_field[(size_t)(i*N + j)*N + k];
            double xa     = xa_const * compute_S_alpha(z, value2, f_HI) * Jalpha;
            double xc     = x_c(z, value2, f_HI * num_H_cgs, num_e_cgs, num_p_cgs);
            double Tspin  = T_spin(z, value2, xc, xa);

            double sf      = 1.0 - tcmb / Tspin;
            double rin_val = prefactor_21cm * value / global_gas * sf; //f_HI * value5

            rin[(i*N + j) * (2*(N/2+1)) + k] = rin_val;

            local_totmass  += value1 * volume_cell; // shouldn't be changed?
            local_totmass2 += value1 * value1 * volume_cell * volume_cell;
            local_volume   += volume_cell;
            if (value <= 0.0) nzerocells++;

            /* diagnostic accumulation */
            d_nH        += num_H_cgs;
            d_Tgas      += value2;
            d_fHI       += f_HI;
            d_Jalpha    += Jalpha;
            d_xa        += xa;
            d_xc        += xc;
            d_Tspin     += Tspin;
            d_sf        += sf;
            d_rin       += rin_val/volume_cell;
            d_overdense += value / global_gas;
            d_ncells++;

            if (num_H_cgs  < d_nH_min)     d_nH_min     = num_H_cgs;
            if (num_H_cgs  > d_nH_max)     d_nH_max     = num_H_cgs;
            if (value2     < d_Tgas_min)   d_Tgas_min   = value2;
            if (value2     > d_Tgas_max)   d_Tgas_max   = value2;
            if (Jalpha     < d_Jalpha_min) d_Jalpha_min = Jalpha;
            if (Jalpha     > d_Jalpha_max) d_Jalpha_max = Jalpha;
            if (xa         < d_xa_min)     d_xa_min     = xa;
            if (xa         > d_xa_max)     d_xa_max     = xa;
            if (xc         < d_xc_min)     d_xc_min     = xc;
            if (xc         > d_xc_max)     d_xc_max     = xc;
            if (Tspin      < d_Tspin_min)  d_Tspin_min  = Tspin;
            if (Tspin      > d_Tspin_max)  d_Tspin_max  = Tspin;
            if (sf         < d_sf_min)     d_sf_min     = sf;
            if (sf         > d_sf_max)     d_sf_max     = sf;
            if (rin_val    < d_rin_min)    d_rin_min    = rin_val;
            if (rin_val    > d_rin_max)    d_rin_max    = rin_val;

            if (f_HI  <= 0.0) d_nHI_zero++;
            if (Tspin <= 0.0) d_Ts_neg++;
          }
      }

#ifdef TK_REFERENCE
  {
    long gc; double gh[2], lh[2] = {d_ref_HI, d_all_HI};
    MPI_Allreduce(&d_ref_cells, &gc, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(lh, gh, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (ThisTask == 0)
      printf("TK_REFERENCE: replaced in %.4f of the cells, holding %.4f of the H I mass\n",
             (double)gc / ((double)N * N * N), gh[1] > 0 ? gh[0] / gh[1] : 0.0);
  }
#endif
  /* --- MPI reductions for diagnostics --- */
  {
    double ls[10] = { d_nH, d_Tgas, d_fHI, d_Jalpha, d_xa,
                      d_xc, d_Tspin, d_sf, d_rin, d_overdense };
    double gs[10];
    MPI_Allreduce(ls, gs, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double g_nH_min,  g_nH_max,  g_Tgas_min, g_Tgas_max;
    double g_Ja_min,  g_Ja_max,  g_xa_min,   g_xa_max;
    double g_xc_min,  g_xc_max,  g_Ts_min,   g_Ts_max;
    double g_sf_min,  g_sf_max,  g_rin_min,  g_rin_max;
    long   g_ncells,  g_nHI_zero, g_Ts_neg;

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
    MPI_Allreduce(&d_nHI_zero,  &g_nHI_zero, 1, MPI_LONG,   MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&d_Ts_neg,    &g_Ts_neg,   1, MPI_LONG,   MPI_SUM, MPI_COMM_WORLD);

    if (ThisTask == 0) {
      double nc = (double)g_ncells;
      printf("\n");
      printf("========================================================\n");
      printf("  DIAGNOSTIC: rin variable audit   z = %.4f\n", z);
      printf("  Total cells = %ld  (expected %d^3 = %ld)\n",
             g_ncells, N, (long)N*N*N);
      printf("  T_CMB            = %.6f K\n", tcmb);
      printf("  prefactor_21cm   = %.6e mK\n", prefactor_21cm);
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
             "f_HI",              gs[2]/nc);
      printf("  %-22s  %12.4e  %12.4e  %12.4e   [ph/s/cm^2/Hz/sr]\n",
             "J_alpha (star+AGN)",gs[3]/nc, g_Ja_min,  g_Ja_max);
      printf("  %-22s  %12.4e  %12.4e  %12.4e\n",
             "x_alpha (WF)",      gs[4]/nc, g_xa_min,  g_xa_max);
      printf("  %-22s  %12.4e  %12.4e  %12.4e\n",
             "x_c (coll.)",       gs[5]/nc, g_xc_min,  g_xc_max);
      printf("  %-22s  %12.4e  %12.4e  %12.4e   [K]\n",
             "T_spin",            gs[6]/nc, g_Ts_min,  g_Ts_max);
      printf("  %-22s  %12.6f  %12.6f  %12.6f\n",
             "sf = 1-Tcmb/Ts",   gs[7]/nc, g_sf_min,  g_sf_max);
      printf("  %-22s  %12.6f\n",
             "rho_HI/global_gas", gs[9]/nc);
      printf("  %-22s  %12.4e  %12.4e  %12.4e   [mK]\n",
             "rin (T_b)",         gs[8]/nc, g_rin_min, g_rin_max);
      printf("--------------------------------------------------------\n");
      printf("  Cells with f_HI  <= 0 : %ld  (%.2f%%)\n",
             g_nHI_zero, 100.0*g_nHI_zero/nc);
      printf("  Cells with T_spin <= 0 : %ld  (%.2f%%)\n",
             g_Ts_neg,   100.0*g_Ts_neg/nc);
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

  /* Opt #1: rin and rin1 are identical — copy instead of recomputing */
  memcpy(rin1, rin, 2 * alloc_local * sizeof(double));

  /* tot1* mirrors tot* since both fields are the same */
  tot1mass   = totmass;
  tot1mass2  = totmass2;
  tot1volume = totvolume;

  if(ThisTask==0)
    printf("Volume weighted mean = %g | Old way = %g | Zero cells = %d | non-zero vol = %g | vol = %g\n",
           totmass/totvolume, totmass/pow(BOXSIZE,3),
           Nzerocells, totvolume, pow(BOXSIZE,3));

  /* Debug HDF5 output */
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

  hid_t file = H5Fcreate(debugfilename, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);
  if (file < 0) {
    if (ThisTask == 0) fprintf(stderr, "FATAL: cannot create %s (does the directory exist?)\n", debugfilename);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  hsize_t dims[3] = {N, N, N};
  hid_t dspace = H5Screate_simple(3, dims, NULL);

  hid_t dset = H5Dcreate(file, "/rin", H5T_NATIVE_FLOAT, dspace, H5P_DEFAULT);

  hsize_t hstart[3] = {local_0_start, 0, 0};
  hsize_t count[3]  = {local_n0, N, N};

  hid_t filespace = H5Dget_space(dset);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, hstart, NULL, count, NULL);

  hid_t memspace = H5Screate_simple(3, count, NULL);

  size_t total_cells = (size_t)local_n0 * N * N;
  float *buffer = malloc(total_cells * sizeof(float));

  for (int i = 0; i < local_n0; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < N; k++)
        buffer[(i*N + j)*N + k] =
            (float)(rin[(i*N + j)*(2*(N/2+1)) + k] / volume_cell);

  hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);

  if (H5Dwrite(dset, H5T_NATIVE_FLOAT, memspace, filespace, xfer, buffer) < 0) {
    if (ThisTask == 0) fprintf(stderr, "FATAL: cannot write the T_b map to %s\n", debugfilename);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  {   /* record what the map is */
    double zz = 1.0/afac - 1.0;
    hid_t sc = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(dset, "Redshift", H5T_NATIVE_DOUBLE, sc, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &zz); H5Aclose(a);
    hid_t st = H5Tcopy(H5T_C_S1); H5Tset_size(st, 128);
    char u[128];
    snprintf(u, sizeof u, "%s%s", H_REDSHIFT_SPACE ? "delta T_b [mK], real-space spin factor x redshift-space H I density"
                                                    : "delta T_b [mK], real space",
#ifdef TK_REFERENCE
#if TK_REF_ANCHOR == 1
             "; no-heating reference T_K (each pixel's T at the anchor snapshot, adiabatic) where x_HII < TK_REF_XTHRESH"
#else
             "; adiabatic reference T_K (Eq. 27) where x_HII < TK_REF_XTHRESH"
#endif
#else
             ""
#endif
             );
    a = H5Acreate2(dset, "content", st, sc, H5P_DEFAULT, H5P_DEFAULT); H5Awrite(a, st, u); H5Aclose(a);
    H5Tclose(st); H5Sclose(sc);
  }

  free(buffer);
  H5Pclose(xfer);
  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Dclose(dset);
  H5Sclose(dspace);
  if (H5Fclose(file) < 0) {
    if (ThisTask == 0) fprintf(stderr, "FATAL: closing %s failed (disk full?)\n", debugfilename);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  fftw_execute(plan);
  fftw_execute(plan1);

  if(ThisTask==0) printf("Finished fftw\n");

  output_PS(local_n0, local_0_start, cout, cout1, outputfilename);

  if(ThisTask==0) printf("Finished power spectrum\n");

  fftw_free(rin1);
  fftw_free(cout1);
  fftw_free(rin);
  fftw_free(cout);

  fftw_destroy_plan(plan);
  fftw_destroy_plan(plan1);

  MPI_Barrier(MPI_COMM_WORLD);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

  if(ThisTask == 0)
    printf("Time = %g seconds\n", cpu_time_used);

  MPI_Finalize();

  return 0;
}


/*#############################################################*/

static void get_offsets_and_data(void)
{
  double density_factor = (MASS / hubble) / pow(afac * LENGTH / hubble, 3);

  offsets = (myint *)malloc(NTask * sizeof(myint)) ;
  MPI_Allgather(&dset_length_task, sizeof(myint), MPI_BYTE, offsets, sizeof(myint), MPI_BYTE, MPI_COMM_WORLD) ;
  myint temp = 0 ;
  for(int i=0; i<NTask; i++)
    {
      myint temp1 = offsets[i] ;
      offsets[i] += temp ;
      temp += temp1 ;
    }

  MPI_Barrier(MPI_COMM_WORLD) ;

  /* Self-checks on the raw fields:
   *  - SpectralDensityHI is the H I mass density remapped along +z, with X = 0.76 and
   *    star-forming (equation-of-state) gas counted as fully neutral. The render
   *    (AREPO RT_movie_3D_rendering.cc, set_partout) sets MassHI = Mass for that gas and builds
   *    HII_Fraction from Mass - MassHI, so the EoS gas is neutral in HII_Fraction too. The
   *    remapping conserves mass, so sum(SpectralDensityHI)/0.76 / sum(Density (1 - x_HII)) = 1
   *    for the mass-weighted HII_Fraction (measured to 1e-5 on Lumina); with TS_VOLUME_WEIGHTED
   *    it differs from 1 only by the mass vs volume weighting. Reported only. A missing X would
   *    give 1/0.76.
   *  - the He fractions must be per H nucleus (<= y). */
  {
    double ls[2] = {0.0, 0.0}, gs[2], lmax = 0.0, gmax;
    for (myint i = 0; i < dset_length_task; i++) {
      ls[0] += dset[i];
      ls[1] += dset2[i] * (1.0 - dset3[i]);
      if (dset6[i] > lmax) lmax = dset6[i];
      if (dset7[i] > lmax) lmax = dset7[i];
    }
    MPI_Allreduce(ls, gs, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double ratio = (gs[1] > 0.0) ? gs[0] / 0.76 / gs[1] : 1.0;
    if (ThisTask == 0)
      printf("Check: sum(SpectralDensityHI)/0.76 / sum(Density x_HI%s) = %.5f (1 if mass-weighted; "
             "1/0.76 = 1.316 would mean no X); max He fraction = %.4f (<= %.4f)\n",
             TS_VOLUME_WEIGHTED ? ", volume-weighted x" : "", ratio, gmax, 0.24 / (4.0 * 0.76));
    if (gmax > 1.05 * 0.24 / (4.0 * 0.76)) {
      if (ThisTask == 0)
        fprintf(stderr, "FATAL: He fraction up to %.3f > y: the fields look per He nucleus, the code assumes per H\n", gmax);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  double local_gas=0.0 ;
  global_gas=0.0 ;
  for(myint i=0; i<dset_length_task; i++)
    {
      dset2[i] *= volume_cell ; // density → mass (sim units)
      local_gas += dset2[i] ;
      dset[i] /= 0.76 ; // to get back to the density
    }
  MPI_Allreduce(&local_gas, &global_gas, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  global_gas /= (L*L*L) ; // mean gas density in sim units

  if(ThisTask==0)
    printf("global_gas = %g \n", global_gas) ;

  for(myint j=0; j<dset_length_task; j++)
    {
      myint index ;
      if(ThisTask == 0)
        index = j ;
      else
        index = offsets[ThisTask-1] + j ;

      rho_HI_spec[index] = dset[j] * volume_cell;
      rho_gas[index]     = dset2[j] ;
      f_HII[index]       = dset3[j] ;
      T_gas[index]       = dsetT[j] ;
      SFRD[index]        = dset4[j] ;
      rho_spec[index]    = dset5[j] * volume_cell;
      f_HeII[index]      = dset6[j] ;
      f_HeIII[index]     = dset7[j] ;
    }

  free(dset) ;
  free(dset2) ;
  free(dset3) ;
  free(dset4);
  free(dsetT);
  free(dset5);
  free(dset6);
  free(dset7);
}

/*#############################################################*/

static void grid_data(void)
{
  double dx = BOXSIZE / ((double) (N)) ;
  if(ThisTask==0)
    {
      memset(intensity, 0,  N*N*N*sizeof(double)) ;
      memset(SFRD, 0,  N*N*N*sizeof(double)) ;
      for(int index=0; index<NumSubhalos; index++)
        {
          zpos[index] += VZ[index] / (get_Hz(afac) * afac) ;
          if(zpos[index]>=BOXSIZE) zpos[index] -= BOXSIZE ;
          if(zpos[index]<0.0)     zpos[index] += BOXSIZE ;

          int i = (int) (xpos[index]/dx) ;
          int j = (int) (ypos[index]/dx) ;
          int k = (int) (zpos[index]/dx) ;

          intensity[i*N*N + j*N + k] += Lum[index] ;
          sum_gal += Lum[index] ;
        }
    }
  MPI_Bcast(&sum_gal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD) ;
}

static double get_Hz(double afac)
{
  double redz = 1.0/afac - 1.0 ;
  return 67.74 * sqrt(0.3089*pow((1.0+redz),3) + (1.0-0.3089)) ;
}


/*#############################################################*/

static void divide_galaxies(int numfiles, int *first, int *last)
{
  int rat = numfiles / NTask ;
  int rem = numfiles % NTask ;

  int newfirsttask, newnumfiles, newtask, newnumtask, newfirstfile ;

  if(ThisTask < NTask -rem)
    {
      *first = ThisTask*rat ;
      *last = (ThisTask+1) * rat - 1 ;
    }
  else
    {
      newnumfiles = numfiles - (NTask-rem)*rat ;
      newnumtask = rem ;
      newfirsttask = (NTask-rem) ;
      newfirstfile = newfirsttask * rat ;
      newtask = ThisTask - newfirsttask ;
      rat = newnumfiles / newnumtask ;
      *first = newfirstfile + newtask*rat ;
      *last = newfirstfile + (newtask+1)*rat -1 ;
    }

  return ;
}

/*#############################################################*/

static void read_header(int snapnum)
{
    char fname[500];

    if (ThisTask == 0)
    {
        sprintf(fname, "%s/All/All_%03d.hdf5", inputfilename, snapnum);

        hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
        {
            printf("FATAL: cannot open header file %s\n", fname);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        hid_t grp = H5Gopen(file, "/Header");
        if (grp < 0)
        {
            printf("FATAL: /Header not found in %s\n", fname);
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        H5Aread(H5Aopen(grp, "NumPixels",              H5P_DEFAULT), H5T_NATIVE_INT,    &N);
        H5Aread(H5Aopen(grp, "BoxSize",                H5P_DEFAULT), H5T_NATIVE_DOUBLE, &BOXSIZE);
        H5Aread(H5Aopen(grp, "Time",                   H5P_DEFAULT), H5T_NATIVE_DOUBLE, &afac);
        H5Aread(H5Aopen(grp, "HubbleParam",            H5P_DEFAULT), H5T_NATIVE_DOUBLE, &hubble);
        H5Aread(H5Aopen(grp, "UnitMass_in_g",          H5P_DEFAULT), H5T_NATIVE_DOUBLE, &MASS);
        H5Aread(H5Aopen(grp, "UnitLength_in_cm",       H5P_DEFAULT), H5T_NATIVE_DOUBLE, &LENGTH);
        H5Aread(H5Aopen(grp, "UnitVelocity_in_cm_per_s", H5P_DEFAULT), H5T_NATIVE_DOUBLE, &VEL);
        if (H5Aexists(grp, "Omega0") > 0)      H5Aread(H5Aopen(grp, "Omega0",      H5P_DEFAULT), H5T_NATIVE_DOUBLE, &omega_m);
        if (H5Aexists(grp, "OmegaBaryon") > 0) H5Aread(H5Aopen(grp, "OmegaBaryon", H5P_DEFAULT), H5T_NATIVE_DOUBLE, &omega_b);
        if (H5Aexists(grp, "OmegaLambda") > 0) H5Aread(H5Aopen(grp, "OmegaLambda", H5P_DEFAULT), H5T_NATIVE_DOUBLE, &omega_l);

        H5Gclose(grp);
        H5Fclose(file);

        NumSubhalos = 0;
        Numfiles    = 1;
    }

    MPI_Bcast(&N,           1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&BOXSIZE,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&afac,        1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&hubble,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MASS,        1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&LENGTH,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&VEL,         1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NumSubhalos, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&omega_m,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&omega_b,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&omega_l,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    h = hubble;    /* unit conversions and the T_b prefactor use the render's own h */
    if (ThisTask == 0)
        printf("Cosmology: h = %.5f (header), Omega_m = %.6f, Omega_b = %.6f\n", h, omega_m, omega_b);
}

/*#############################################################*/

/* Opt #5: collective MPI-IO for all field reads (checked) */
static void read_single_field(const char *field, double *dest, int snapnum,
                               ptrdiff_t local_n0, ptrdiff_t local_0_start,
                               float *buffer, myint length_vector)
{
    /* <dir>/<field>/<field>_NNN.hdf5, else the combined <dir>/All/All_NNN.hdf5 */
    char fname[1024], dsetname[256];
    sprintf(dsetname, "/%s", field);
    sprintf(fname, "%s/%s/%s_%03d.hdf5", inputfilename, field, field, snapnum);
    if (access(fname, R_OK) != 0)
        sprintf(fname, "%s/All/All_%03d.hdf5", inputfilename, snapnum);

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t hdf5_file = H5Fopen(fname, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (hdf5_file < 0) {
        if (ThisTask == 0) fprintf(stderr, "FATAL: cannot open %s for field %s\n", fname, field);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    hid_t dataset = H5Dopen(hdf5_file, dsetname);
    if (dataset < 0) {
        if (ThisTask == 0) fprintf(stderr, "FATAL: no dataset %s in %s\n", dsetname, fname);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    hid_t dataspace = H5Dget_space(dataset);
    hsize_t dm[3] = {0, 0, 0};
    if (H5Sget_simple_extent_ndims(dataspace) != 3 || H5Sget_simple_extent_dims(dataspace, dm, NULL) < 0
        || dm[0] != (hsize_t)N || dm[1] != (hsize_t)N || dm[2] != (hsize_t)N) {
        if (ThisTask == 0) fprintf(stderr, "FATAL: %s in %s is not %d^3\n", dsetname, fname, N);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    hsize_t start[3] = {local_0_start, 0, 0};
    hsize_t count[3] = {local_n0, N, N};
    hid_t   memspace = H5Screate_simple(3, count, NULL);
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    if (H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, xfer, buffer) < 0) {
        if (ThisTask == 0)
            fprintf(stderr, "FATAL: cannot read %s from %s. If it is compressed with a non-standard filter\n"
                    "       (Blosc/Zstd/Bitshuffle), set HDF5_PLUGIN_PATH (python -c 'import hdf5plugin; "
                    "print(hdf5plugin.PLUGIN_PATH)')\n", dsetname, fname);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    H5Pclose(xfer);

    for (myint j = 0; j < length_vector; j++)
        dest[j] = buffer[j];

    H5Sclose(memspace); H5Sclose(dataspace);
    H5Dclose(dataset);  H5Fclose(hdf5_file);
}

static void read_data(int snapnum,
                      int first,
                      int last,
                      ptrdiff_t local_n0,
                      ptrdiff_t local_0_start)
{
    myint length_vector = local_n0 * N * N;
    dset_length_task    = length_vector;

    dset  = malloc(length_vector*sizeof(double)); // SpectralDensityHI (or its real-space equivalent)
    dset2 = malloc(length_vector*sizeof(double)); // Density
    dset3 = malloc(length_vector*sizeof(double)); // F_HII_FIELD
    dsetT = malloc(length_vector*sizeof(double)); // T_FIELD
    dset4 = malloc(length_vector*sizeof(double)); // StarFormationRate
    dset5 = malloc(length_vector*sizeof(double)); // SpectralDensity
    dset6 = malloc(length_vector*sizeof(double)); // F_HEII_FIELD
    dset7 = malloc(length_vector*sizeof(double)); // F_HEIII_FIELD

    float *buffer = malloc(length_vector*sizeof(float));

#if H_REDSHIFT_SPACE
    read_single_field("SpectralDensityHI",       dset,  snapnum, local_n0, local_0_start, buffer, length_vector);
#else
    read_single_field("HII_Fraction",            dset,  snapnum, local_n0, local_0_start, buffer, length_vector);
#endif
    read_single_field("Density",                 dset2, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field(F_HII_FIELD,               dset3, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field(T_FIELD,                   dsetT, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field("StarFormationRate",       dset4, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field("SpectralDensity",         dset5, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field(F_HEII_FIELD,              dset6, snapnum, local_n0, local_0_start, buffer, length_vector);
    read_single_field(F_HEIII_FIELD,             dset7, snapnum, local_n0, local_0_start, buffer, length_vector);
#if !H_REDSHIFT_SPACE
    for (myint i = 0; i < length_vector; i++)
        dset[i] = 0.76 * dset2[i] * (1.0 - dset[i]);   /* real-space H I density, in SpectralDensityHI units */
#endif
#if defined(TK_REFERENCE) && TK_REF_ANCHOR == 1
    {   /* the anchor: T and Density of render snapshot TK_REF_SNAP0, same slab */
        char saved[500];
        const char *d0 = getenv("TK_REF_DIR0");
        strcpy(saved, inputfilename);
        if (d0 && *d0) snprintf(inputfilename, sizeof inputfilename, "%s", d0);
        ref_T0   = malloc(length_vector * sizeof(double));
        ref_rho0 = malloc(length_vector * sizeof(double));
        read_single_field(T_FIELD,   ref_T0,   TK_REF_SNAP0, local_n0, local_0_start, buffer, length_vector);
        read_single_field("Density", ref_rho0, TK_REF_SNAP0, local_n0, local_0_start, buffer, length_vector);
        if (ThisTask == 0) {
            char fname[1024];
            snprintf(fname, sizeof fname, "%s/All/All_%03d.hdf5", inputfilename, TK_REF_SNAP0);
            hid_t file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
            hid_t grp  = (file >= 0) ? H5Gopen(file, "/Header") : -1;
            if (grp < 0 || H5Aexists(grp, "Redshift") <= 0) {
                fprintf(stderr, "FATAL: no /Header Redshift in %s (TK_REFERENCE anchor)\n", fname);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            H5Aread(H5Aopen(grp, "Redshift", H5P_DEFAULT), H5T_NATIVE_DOUBLE, &ref_z0);
            H5Gclose(grp); H5Fclose(file);
            printf("TK_REFERENCE anchor: render snapshot %d (z0 = %.4f) from %s\n", TK_REF_SNAP0, ref_z0, inputfilename);
        }
        MPI_Bcast(&ref_z0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        strcpy(inputfilename, saved);
    }
#endif
    if (ThisTask == 0)
        printf("Spin-temperature fields: %s, %s, %s, %s; signal: %s\n", F_HII_FIELD, T_FIELD, F_HEII_FIELD, F_HEIII_FIELD,
               H_REDSHIFT_SPACE ? "SpectralDensityHI (redshift space, 3rd axis)" : "0.76 Density (1 - HII_Fraction) (real space)");

    free(buffer);
}


/*#############################################################*/

static void allocate_shared(void)
{
  if(ThisTask == 0)
    {
      MPI_Win_allocate_shared(N * N * N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &rho_HI_spec, &win_rho_HI_spec) ;
      MPI_Win_allocate_shared(N * N * N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &rho_spec,    &win_rho_spec) ;
      MPI_Win_allocate_shared(N * N * N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &rho_gas,     &win_rho_gas) ;
      MPI_Win_allocate_shared(N * N * N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &f_HII,       &win_f_HII) ;
      MPI_Win_allocate_shared(N * N * N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &f_HeII,      &win_f_HeII) ;
      MPI_Win_allocate_shared(N * N * N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &f_HeIII,     &win_f_HeIII) ;
      MPI_Win_allocate_shared(N * N * N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &SFRD,        &win_SFRD) ;
      MPI_Win_allocate_shared(N * N * N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &T_gas,       &win_T_gas) ;
      MPI_Win_allocate_shared(N * N * N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &diffuse,     &win_diffuse) ;
    }
  else
    {
      int disp_unit ;
      MPI_Aint size ;

      MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &rho_HI_spec, &win_rho_HI_spec) ;
      MPI_Win_shared_query(win_rho_HI_spec, 0, &size, &disp_unit, &rho_HI_spec);

      MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &rho_gas, &win_rho_gas) ;
      MPI_Win_shared_query(win_rho_gas, 0, &size, &disp_unit, &rho_gas);

      MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &rho_spec, &win_rho_spec) ;
      MPI_Win_shared_query(win_rho_spec, 0, &size, &disp_unit, &rho_spec);

      MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &f_HII, &win_f_HII) ;
      MPI_Win_shared_query(win_f_HII, 0, &size, &disp_unit, &f_HII);

      MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &f_HeII, &win_f_HeII) ;
      MPI_Win_shared_query(win_f_HeII, 0, &size, &disp_unit, &f_HeII);

      MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &f_HeIII, &win_f_HeIII) ;
      MPI_Win_shared_query(win_f_HeIII, 0, &size, &disp_unit, &f_HeIII);

      MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &SFRD, &win_SFRD) ;
      MPI_Win_shared_query(win_SFRD, 0, &size, &disp_unit, &SFRD);

      MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &T_gas, &win_T_gas) ;
      MPI_Win_shared_query(win_T_gas, 0, &size, &disp_unit, &T_gas);

      MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &diffuse, &win_diffuse) ;
      MPI_Win_shared_query(win_diffuse, 0, &size, &disp_unit, &diffuse);
    }
  return ;
}

static void free_shared(void)
{
  MPI_Win_free(&win_rho_HI_spec) ;
  MPI_Win_free(&win_rho_spec) ;
  MPI_Win_free(&win_rho_gas) ;
  MPI_Win_free(&win_f_HII) ;
  MPI_Win_free(&win_f_HeII) ;
  MPI_Win_free(&win_f_HeIII) ;
  MPI_Win_free(&win_T_gas) ;
  MPI_Win_free(&win_SFRD);
  MPI_Win_free(&win_diffuse) ;
}


static double input_data(int i, int j, int k, int line)
{
  if(line==3)
    return  intensity[i*N*N + j*N + k]*(1.0 - sum_diffuse*constK/sum_gal) + diffuse[i*N*N + j*N + k]*constK ;
  else
    return intensity[i*N*N + j*N + k] ;
}

static double input_data1(int i, int j, int k) { return rho_HI_spec[i*N*N + j*N + k]; }
static double input_data0(int i, int j, int k) { return rho_gas[i*N*N + j*N + k]; }
static double input_data4(int i, int j, int k) { return f_HII[i*N*N + j*N + k]; }
static double input_data2(int i, int j, int k) { return T_gas[i*N*N + j*N + k]; }
static double input_data3(int i, int j, int k) { return SFRD[i*N*N + j*N + k]; }
static double input_data5(int i, int j, int k) { return rho_spec[i*N*N + j*N + k]; }
static double input_data6(int i, int j, int k) { return f_HeII[i*N*N + j*N + k]; }
static double input_data7(int i, int j, int k) { return f_HeIII[i*N*N + j*N + k]; }

/*#############################################################*/

/* ------------------------------------------------------------------
 * read_Jalpha_slab
 * Reads the local MPI slab from a precomputed Jalpha HDF5 file.
 * Stored value = J_alpha  (xa_const, S_alpha not included).
 * ------------------------------------------------------------------ */
static void read_Jalpha_slab(const char *filename, const char *dsetname,
                              ptrdiff_t local_n0, ptrdiff_t local_0_start,
                              double *Jalpha)
{
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);

    if (file < 0) {
        if (ThisTask == 0) fprintf(stderr, "FATAL: cannot open J_alpha file %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    hid_t dset = H5Dopen(file, dsetname);
    if (dset < 0) {
        if (ThisTask == 0) fprintf(stderr, "FATAL: no dataset %s in %s\n", dsetname, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* the J map must be for this snapshot and this grid */
    double z_render = 1.0 / afac - 1.0, zJ = -1.0;
    if (H5Aexists(dset, "Redshift") > 0) {
        hid_t a = H5Aopen(dset, "Redshift", H5P_DEFAULT);
        H5Aread(a, H5T_NATIVE_DOUBLE, &zJ); H5Aclose(a);
        if (fabs(zJ - z_render) > 1e-3) {
            if (ThisTask == 0) fprintf(stderr, "FATAL: %s is for z = %.4f, the render is at z = %.4f\n",
                                       filename, zJ, z_render);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else if (ThisTask == 0)
        printf("WARNING: %s:%s has no Redshift attribute: snapshot match not checked\n", filename, dsetname);
    hid_t dspace = H5Dget_space(dset);
    hsize_t dm[3] = {0, 0, 0};
    if (H5Sget_simple_extent_ndims(dspace) != 3 || H5Sget_simple_extent_dims(dspace, dm, NULL) < 0
        || dm[0] != (hsize_t)N || dm[1] != (hsize_t)N || dm[2] != (hsize_t)N) {
        if (ThisTask == 0) fprintf(stderr, "FATAL: %s in %s is not %d^3\n", dsetname, filename, N);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    hsize_t start[3] = {local_0_start, 0, 0};
    hsize_t count[3] = {local_n0,      N, N};
    H5Sselect_hyperslab(dspace, H5S_SELECT_SET, start, NULL, count, NULL);
    hid_t memspace = H5Screate_simple(3, count, NULL);

    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    if (H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, dspace, xfer, Jalpha) < 0) {
        if (ThisTask == 0) fprintf(stderr, "FATAL: cannot read %s from %s\n", dsetname, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    H5Pclose(xfer);
    if (ThisTask == 0)
        printf("J_alpha %s:%s  (z = %.4f, render z = %.4f)\n", filename, dsetname, zJ, z_render);

    H5Sclose(memspace); H5Sclose(dspace);
    H5Dclose(dset);     H5Fclose(file);
}

/*#############################################################*/

void output_PS(int local_n0, int local_0_start, fftw_complex *cout, fftw_complex *cout1, char *outputfilename)
{
  FILE *fd = NULL ;

  if (ThisTask == 0) {
    fd = fopen(outputfilename, "w") ;
    if (!fd) { fprintf(stderr, "FATAL: cannot write %s\n", outputfilename); MPI_Abort(MPI_COMM_WORLD, 1); }
  }

  L = BOXSIZE ;

  double normalization = 1.0/sqrt(totmass*tot1mass) ;
  double kfx, kfy, kfz, k1, k2, k3 ;

  double d = L / N ;
  double dhalf = 0.5*d ;

  double Kmin = 2.0 * M_PI / L ;
  double Kmax = Kmin * N /2 ;

  double binfac = BINS_PS / (log(Kmax) - log(Kmin)) ;

  double *SumPower   = (double *)malloc(BINS_PS * sizeof(double)) ;
  double *CountModes = (double *)malloc(BINS_PS * sizeof(double)) ;

  memset(SumPower,   0, BINS_PS*sizeof(double)) ;
  memset(CountModes, 0, BINS_PS*sizeof(double)) ;

  kfx = 2.0*M_PI/L ;
  kfy = 2.0*M_PI/L ;
  kfz = 2.0*M_PI/L ;

  /* Precompute k1 and k2 arrays to avoid repeated conditionals */
  double *k1arr = malloc(N * sizeof(double));
  double *k2arr = malloc(N * sizeof(double));
  for (int ii = 0; ii < local_n0; ii++) {
      int II = (2*(ii+local_0_start) < N) ? (ii+local_0_start) : N-(ii+local_0_start);
      k1arr[ii] = (double)II * kfx;
  }
  for (int jj = 0; jj < N; jj++) {
      int JJ = (2*jj < N) ? jj : N-jj;
      k2arr[jj] = (double)JJ * kfy;
  }

  double log_Kmin = log(Kmin);

  for (int i = 0; i < local_n0; i++)
    {
      k1 = k1arr[i];
      for (int j = 0; j < N; j++)
        {
          k2 = k2arr[j];
          for (int k = 0; k < N/2+1; k++)
            {
              k3 = ((double)k) * kfz;

              double ksq = k1*k1 + k2*k2 + k3*k3;
              double sqk = sqrt(ksq);

              if (sqk >= Kmin && sqk < Kmax)
                {
                  int index = (i*N + j) * (N/2+1) + k;
                  double mag = (cout[index][0]*cout1[index][0]
                              + cout[index][1]*cout1[index][1])
                              * normalization * normalization;

                  int bin = (int)(log(sqk/Kmin) * binfac);
                  SumPower[bin]   += mag;
                  CountModes[bin] += 1.0;
                }
            }
        }
    }

  free(k1arr);
  free(k2arr);

  MPI_Allreduce(MPI_IN_PLACE, SumPower,   BINS_PS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, CountModes, BINS_PS, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if(ThisTask==0)
    {
      fprintf(fd, "%g\n", 1.0/afac - 1.0) ;
      fprintf(fd, "%d\n", (int)BINS_PS) ;
      fprintf(fd, "%g\n", sqrt(totmass*tot1mass)) ;
      fprintf(fd, "%g\n", totvolume) ;
      fprintf(fd, "%15lld\n", (myint) N*N*N) ;
      for(int i=0; i<BINS_PS; i++)
        {
          double Kval = exp((i + 0.5) / binfac + log_Kmin);
          if(CountModes[i] > 0)
            SumPower[i] /= CountModes[i];

          double delta     = 4*M_PI * pow(Kval,3) / pow(2*M_PI/L,3) * SumPower[i];
          double shotnoise = 4*M_PI * pow(Kval,3) / pow(2*M_PI/L,3)
                           * (sqrt(totmass2*tot1mass2)/totmass/tot1mass);

          fprintf(fd, "%g %g %g %g %g \n", Kval, delta, SumPower[i], (double)CountModes[i], shotnoise);
        }
    }
  if (ThisTask == 0) fclose(fd) ;
  free(CountModes) ;
  free(SumPower) ;
}


/*#############################################################*/

static void calculate_cell_intensity(int first, int last, int line)
{
  double z = 1.0/afac - 1.0 ;
  double val1 = sqrt( omega_m*(1.0+z)*(1.0+z)*(1.0+z) + (1.0-omega_m) ) ;
  double val2 = volume_cell ;

  constK /= val1 ;

  if(line==0)      constK *= 0.37275 ;
  else if(line==1) constK *= 0.4861 ;
  else if(line==2) constK *= 0.4983 ;
  else if(line==3) constK *= 0.6563 ;
  else if(line==4) constK *= 0.6583 ;
  else if(line==5) constK *= 0.67235 ;
  else if(line==6) constK *= 88.0 ;
  else if(line==7) constK *= 158.0 ;
}

static double get_intensity(double x, int line)
{
  double a, ma, mb, mc, xb, xc ;
  xb = 0.0 ;
  if(line==0)
    { a=7.08; ma=1.11; mb=1.31; mc=0.64; xc=0.54; }
  else if(line==1)
    { a=7.62; ma=0.96; mb=0.86; mc=0.41; xc=0.96; }
  else if(line==2)
    { a=7.84; ma=1.24; mb=1.19; mc=0.53; xc=0.66; }
  else if(line==3)
    { a=8.08; ma=0.96; mb=0.88; mc=0.45; xc=0.96; }
  else if(line==4)
    { a=5.82; ma=1.35; mb=1.72; mc=0.99; xc=0.59; }
  else if(line==5)
    { a=6.25; ma=1.25; mb=1.43; mc=0.78; xc=0.64; }
  else if(line==6)
    { a=6.98; ma=1.37; mb=1.37; mc=1.0;  xc=10.0; }
  else /* line==7 */
    {
      double z = 1.0/afac - 1.0 ;
      a=7.1-0.07*z; ma=1.4-0.07*z; mb=1.4-0.07*z; mc=1.0; xc=10.0;
    }

  double val ;
  if(x<xb)       val = a + ma*x ;
  else if(x<xc)  val = a + (ma-mb)*xb + mb*x ;
  else            val = a + (ma-mb)*xb + (mc-mb)*xc + mc*x ;

  return pow(10.0, val) ;
}
