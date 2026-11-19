/*!
 * \file    lumina-mfp-hist-grid.cc
 *
 * Lumina version (Changhyun Cho, 2026) of thesan-mfp-hist-grid.cc by Meredith Neyer
 * (https://github.com/meredithneyer/mfp-bubbles-thesan, MIT License, see LICENSE; Neyer et al. 2024,
 * arXiv:2310.03783). Changes from the original:
 *  - reads the Lumina 640^3 renders: <ren_dir>/All/All_NNN.hdf5 (Header) and the per-field files
 *    <ren_dir>/<Field>/<Field>_NNN.hdf5 (else the field in All_NNN.hdf5);
 *  - the ionized fraction is the VOLUME-weighted render field (user decision 2026-11-19: bubble sizes describe the
 *    ionized volume; the mass-weighted fields stop rays at dense He II / H I clumps inside bubbles, which made the He
 *    bubbles 1.3-2.1x smaller at z = 4.5-3.4):
 *      default      : x_HII   = HII_VolumeFraction
 *      -DLINE_HE    : x_HeIII = HeIII_VolumeFraction / y  (the field is per H nucleus; y = 0.24 / (4 x 0.76))
 *    the ray stops where the interpolated fraction crosses 0.5, as in the original;
 *  - 64-bit histogram counters (the 32-bit mfp_hist overflowed in the capped last bin at He snapshot 695: 2.19e9 rays);
 *  - read errors stop the program (they were silent; a missing file gave x = 0 everywhere);
 *  - no NumFiles header attribute (the renders are single files).
 * Build (ORCD): mpicxx -fopenmp -std=c++14 -O3 [-DLINE_HE] lumina-mfp-hist-grid.cc -o <exe> -lhdf5 -lz
 *
 * \brief   Master file containing all program functions.
 *
 * \details This file contains the main function with high level parsing of
 *          command line arguments, parallelism, and program functionality.
 */

// Standard headers
#include <iomanip>    // int -> string
#include <iostream>   // input / output
#include <string>     // Standard strings
#include <vector>     // Standard vectors
#include <random>     // Random numbers
#include <algorithm>  // Max function
#include <sys/stat.h> // Make directories
#include <time.h>     // Timing utilities
#include <omp.h>      // OpenMP parallelism
#include <hdf5.h>     // HDF5 read / write

// Standard library usage
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;
typedef long long myint;

//! \brief Convenience for handling 3D vectors in (x,y,z) order.
struct Vec3 {
  float x = 0.;
  float y = 0.;
  float z = 0.;
  Vec3() = default;
  Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// Configuration constants
static const bool VERBOSE = true;            // Print extra information
static const bool USE_RENDERS = true;       // Use renders instead of smooth_renders
// static const int n_exp = 0;                  // Healpix exponent (12 directions)
// static const int n_exp = 1;                  // Healpix exponent (48 directions)
static const int n_exp = 2;                  // Healpix exponent (192 directions)
// static const int n_exp = 3;                  // Healpix exponent (768 directions)
// static const int n_exp = 4;                  // Healpix exponent (3072 directions)
static const int n_LOS = 12 * pow(4, n_exp); // Number of directions = 12 * 4^n_exp
static const int n_bins_per_cell = 8;        // Number of histogram bins per cell
static const int n_loops_per_box = 2;        // Maximum distance in units of box size
static const double _2pi = 2. * M_PI;

// Distinguish between directions
enum DirectionTypes { NotSet = 0, PosX, NegX, PosY, NegY, PosZ, NegZ };

// Main control variables
static string executable;                    // Executable name (command line)
static string ren_dir;                       // Initial conditions directory (command line)
static string snap_str;                      // Snapshot number (option command line)
static int snap = -1;                        // Snapshot number (converted from string)
static string out_dir;                       // Output directory name
vector<vector<double>> directions;           // Healpix direction vectors

// OpenMP variables
static int thread;                           // OpenMP local thread
static int n_threads;                        // Number of OpenMP threads
static unsigned long long seed;              // Seed for random number generator
#pragma omp threadprivate(seed, thread)      // Unique to each thread

// Header variables
static double a, BoxSize, BoxSize_cMpc, PixSize, PixSize_cMpc, InvPix, h, Omega0, OmegaBaryon, OmegaLambda;
static double UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
static double length_to_cgs, volume_to_cgs, mass_to_cgs, velocity_to_cgs;

// Grid variables
static int n_files = 1;                      // Number of render files
static myint Ngrid;                          // Number of pixels on a side
static myint Ngrid2;                         // Save hyperslab size
static myint Ngrid3;                         // Total number of cells
static double Ngrid_double;                  // Number of pixels on a side (double)
static double Ngrid3_double;                 // Total number of cells (double)
static double l_min, l_max;                  // Max ray-tracing distance (cell width units)
static int n_bins;                           // Number of histogram bins
static const int n_logR_bins = 360;          // Number of log(R_eff) histogram bins
static const int n_xHII_bins = 360;          // Number of x_HII histogram bins
static const int n_logT_bins = 360;          // Number of log(T) histogram bins
static const int n_logD_bins = 360;          // Number of log(delta+1) histogram bins
static double logR_min, logR_max;            // Min/Max log(R_eff)
static double xHII_min, xHII_max;            // Min/Max log(R_eff)
static double logT_min, logT_max;            // Min/Max log(T)
static double logD_min, logD_max;            // Min/Max log(delta+1)
static double iw_logR, iw_xHII, iw_logT, iw_logD; // Inverse bin widths
static vector<double> edges;                 // Cell edge positions (cell width units)
static vector<double> centers;               // Cell center positions (cell width units)
static vector<float> HII_Fraction;           // HII fraction of each grid cell
static vector<Vec3> HII_Gradient;            // HII fraction gradient of each grid cell
static vector<float> logT;                   // log(T) of each grid cell
static vector<float> logD;                   // log(delta+1) of each grid cell
static vector<long long> mfp_hist;                 // Histogram of bubble sizes
static vector<long long> mfp_hist_xHII;            // Histogram of bubble sizes also with xHII
static vector<long long> mfp_hist_logT;            // Histogram of bubble sizes also with log(T)
static vector<long long> mfp_hist_logD;            // Histogram of bubble sizes also with log(delta+1)
static vector<float> mfp_avgs;               // Grid of averages bubble sizes

static void read_header();                   // Read header information
static void read_data();                     // Read grid data
static void read_render_data();              // Read grid data
// static void apply_threshold();               // Apply HII fraction threshold
static void calculate_gradients();           // Calculate HII fraction gradients
static void initialize_healpix_directions(const int order); // Calculate healpix directions
static void calculate_mfp_avgs();            // Calculate all mean-free-paths
static double calculate_mfp_avg(const myint start_cell); // Calculate mean-free-path of one cell (cell width units)
static double calculate_mfp_LOS(myint start_cell, const int i_LOS); // Calculate mean-free-path of one direction in one cell (cell width units)
static void write_data();                    // Write data to an output file

// Simple function to print an error message and exit
#define error(message) { error_info(message, __FILE__, __LINE__, __func__); }
template <typename T>
void error_info(const T message, const string file, const int line, const string func) {
  std::cerr << "\nError: " << message
            << "\nCheck: " << file << ":" << line << " in " << func << "()" << endl;
  exit(EXIT_FAILURE);
}


/*! \brief Uniform random number generator for everyday use.
 *
 *  This is the Ranq1 random number generator suggested by Numerical Recipes.
 *  The period is \f$\sim 1.8 \times 10^{19}\f$ sufficient for our applications.
 *
 *  \return Random number drawn uniformly in the interval \f$[0,1)\f$.
 */
double ran() {
  seed ^= seed >> 21;
  seed ^= seed << 35;
  seed ^= seed >> 4;
  return 5.42101086242752217E-20 * (seed * 2685821657736338717LL);
}


/*! \brief Main function of the program.
 *
 *  \param[in] argc Number of arguments passed to the program.
 *  \param[in] argv Full list of arguments passed to the program.
 *
 *  \return Exit code (0 = success, 1 = failure).
 */
int main(int argc, char** argv) {
  // Start the clock
  const clock_t start = clock();

  // Get the number of OpenMP threads
  #pragma omp parallel
  #pragma omp single
  n_threads = omp_get_num_threads();

  // Set unique thread and random seed
  #pragma omp parallel
  {
    thread = omp_get_thread_num();           // Get the thread number once
    std::random_device rd;                   // Obtains seeds for the random number engine
    seed = rd();                             // Set random seeds for parallel processes
  }

  // Configuration and general error checking
  executable = argv[0];                      // Executable name (command line)
  if (argc != 4)
    error("Please specify all arguments: " + executable + " [ren_dir] [out_dir] [snapshot]");
  ren_dir = argv[1];                         // Initial conditions directory
  out_dir = argv[2];                         // Output directory name
  snap_str = argv[3];                        // Snapshot number
  snap = std::stoi(snap_str);                // Convert to integer
  if (snap < 0)
    error("Expected snapshot number to be >= 0, but recieved " + snap_str);
  snap_str = "_" + string(3 - snap_str.length(), '0') + snap_str; // %03d format
  int status = mkdir(out_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
  if (status != 0 && errno != EEXIST)
    error("Could not create the output directory: " + out_dir);

  // Read header information
  read_header();
  initialize_healpix_directions(n_exp);

  cout << " ___       ___  __             \n"
       << "  |  |__| |__  /__`  /\\  |\\ |\n"
       << "  |  |  | |___ .__/ /--\\ | \\|\n"
       << "\nInput Directory: " << ren_dir
       << "\nOutput Directory: " << out_dir
       << "\nRunning with " << n_threads << " threads"
       << "\n\nSnap " << snap << ": Ngrid = " << Ngrid << "  (n_exp = " << n_exp << ", n_LOS = " << n_LOS
       << ")\n\nz = " << 1./a - 1. << ", a = " << a << ", h = " << h
       << "\nBoxSize = " << 1e-3*BoxSize << " cMpc/h = " << 1e-3*BoxSize/h
       << " cMpc\nPixSize = " << 1e-3*PixSize << " cMpc/h = " << 1e-3*PixSize/h << " cMpc" << endl;

  // Read grid data
  if (USE_RENDERS)
    read_render_data();
  else
    read_data();

  // Apply HII fraction threshold
  // apply_threshold();

  // Calculate HII gradients
  calculate_gradients();

  // Calculate all mean-free-paths
  calculate_mfp_avgs();

  // Write mfp data to a file
  write_data();

  // Free memory (overwrite with empty vectors)
  HII_Gradient = vector<Vec3>();
  mfp_avgs = vector<float>();
  mfp_hist_logD = vector<long long>();
  mfp_hist_logT = vector<long long>();
  mfp_hist_xHII = vector<long long>();
  mfp_hist = vector<long long>();
  logD = vector<float>();
  logT = vector<float>();
  HII_Fraction = vector<float>();
  centers = vector<double>();
  edges = vector<double>();

  // End the clock and report timings
  const clock_t end = clock();
  const double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  cout << "\nElapsed time = " << cpu_time_used << " seconds" << endl;

  return 0;
}


//! \brief Read header information.
static void read_header() {
  string filename;
  if (USE_RENDERS)
    filename = ren_dir + "/All/All" + snap_str + ".hdf5";
  else
    filename = ren_dir + "/smooth" + snap_str + ".hdf5";

  hid_t file_id, group_id, attribute_id;
  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  group_id = H5Gopen(file_id, "Header", H5P_DEFAULT);

  if (file_id < 0 || group_id < 0)
    error("Could not open the Header of " + filename);

  int NumPixels; // Saved as int type
  attribute_id = H5Aopen_name(group_id, "NumPixels");
  H5Aread(attribute_id, H5T_NATIVE_INT, &NumPixels);
  H5Aclose(attribute_id);
  Ngrid = NumPixels;                         // Convert to myint type
  Ngrid2 = Ngrid * Ngrid;                    // Number of grid cells in slab
  Ngrid3 = Ngrid * Ngrid2;                   // Total number of grid cells
  Ngrid_double = double(Ngrid);              // As double type
  Ngrid3_double = double(Ngrid3);            // As double type
  l_min = 1.000000001 / double(n_bins_per_cell); // Min distance (cell width units)
  l_max = 0.999999999 * double(Ngrid) * double(n_loops_per_box); // Max distance (cell width units)
  edges.resize(Ngrid+1);                     // Cell edge positions (cell width units)
  centers.resize(Ngrid);                     // Cell center positions (cell width units)
  #pragma omp parallel for
  for (int i = 0; i <= Ngrid; ++i)
    edges[i] = double(i);                    // Grid is in cell width units
  #pragma omp parallel for
  for (int i = 0; i < Ngrid; ++i)
    centers[i] = 0.5 * (edges[i] + edges[i+1]); // Midpoint of edges

  attribute_id = H5Aopen_name(group_id, "Time");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &a);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "BoxSize");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &BoxSize);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "HubbleParam");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &h);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "Omega0");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &Omega0);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "OmegaBaryon");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &OmegaBaryon);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "OmegaLambda");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &OmegaLambda);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "UnitLength_in_cm");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &UnitLength_in_cm);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "UnitMass_in_g");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &UnitMass_in_g);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "UnitVelocity_in_cm_per_s");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &UnitVelocity_in_cm_per_s);
  H5Aclose(attribute_id);

  H5Gclose(group_id);
  H5Fclose(file_id);

  PixSize = BoxSize / double(Ngrid);         // Pixel width
  BoxSize_cMpc = 1e-3 * BoxSize / h;         // Box size [cMpc]
  PixSize_cMpc = 1e-3 * PixSize / h;         // Pixel width [cMpc]
  InvPix = double(Ngrid) / BoxSize;          // Inverse pixel width
  length_to_cgs = a * UnitLength_in_cm / h;
  volume_to_cgs = length_to_cgs * length_to_cgs * length_to_cgs;
  mass_to_cgs = UnitMass_in_g / h;
  velocity_to_cgs = sqrt(a) * UnitVelocity_in_cm_per_s;
}


//! \brief Read grid data.
static void read_data() {
  string filename = ren_dir + "/smooth" + snap_str + ".hdf5";

  hid_t file_id, dataset;
  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  HII_Fraction.resize(Ngrid3);               // Allocate space
  dataset = H5Dopen(file_id, "HII_Fraction", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, HII_Fraction.data());
  H5Dclose(dataset);

  logT.resize(Ngrid3);                       // Allocate space
  dataset = H5Dopen(file_id, "Temperature", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, logT.data());
  H5Dclose(dataset);

  logD.resize(Ngrid3);                       // Allocate space
  dataset = H5Dopen(file_id, "Overdensity", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, logD.data());
  H5Dclose(dataset);

  H5Fclose(file_id);

  // Convert T,D as logT,logD values
  #pragma omp parallel for
  for (myint i = 0; i < Ngrid3; ++i)
    logT[i] = log10(logT[i]);

  #pragma omp parallel for
  for (myint i = 0; i < Ngrid3; ++i)
    logD[i] = log10(logD[i] + 1.);

  // Print data
  if (VERBOSE) {
    cout << "\nHII_Fraction = ["
         << HII_Fraction[0] << " "
         << HII_Fraction[1] << " "
         << HII_Fraction[2] << " ... "
         << HII_Fraction[Ngrid3-3] << " "
         << HII_Fraction[Ngrid3-2] << " "
         << HII_Fraction[Ngrid3-1] << "]" << endl;

    cout << "\nlog(T) = ["
         << logT[0] << " "
         << logT[1] << " "
         << logT[2] << " ... "
         << logT[Ngrid3-3] << " "
         << logT[Ngrid3-2] << " "
         << logT[Ngrid3-1] << "]" << endl;

    cout << "\nlog(delta+1) = ["
         << logD[0] << " "
         << logD[1] << " "
         << logD[2] << " ... "
         << logD[Ngrid3-3] << " "
         << logD[Ngrid3-2] << " "
         << logD[Ngrid3-1] << "]" << endl;
  }
}

/*
//! \brief Read render grid data.
static void read_render_data() {
  hid_t file_id, dataspace, dataset;

  HII_Fraction.resize(Ngrid3);               // Allocate space
  logT.resize(Ngrid3);
  logD.resize(Ngrid3);

  myint offset = 0;
  for (int i = 0; i < n_files; ++i) {
    string i_str = to_string(i);
    i_str = "." + string(3 - i_str.length(), '0') + i_str; // %03d format
    string filename = ren_dir + "/render" + snap_str + "/render" + snap_str + i_str+ "." + "hdf5";

    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    dataset = H5Dopen(file_id, "HII_Fraction", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);
    const int n_dims = H5Sget_simple_extent_ndims(dataspace);
    hsize_t dims[n_dims];
    H5Sget_simple_extent_dims(dataspace, dims, NULL);
    H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &HII_Fraction[offset]);
    H5Dclose(dataset);

    dataset = H5Dopen(file_id, "Temperature", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);
    // const int n_dims = H5Sget_simple_extent_ndims(dataspace);
    // hsize_t dims[n_dims];
    H5Sget_simple_extent_dims(dataspace, dims, NULL);
    H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &logT[offset]);
    H5Dclose(dataset);

    dataset = H5Dopen(file_id, "Density", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);
    // const int n_dims = H5Sget_simple_extent_ndims(dataspace);
    // hsize_t dims[n_dims];
    H5Sget_simple_extent_dims(dataspace, dims, NULL);
    H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &logD[offset]);
    H5Dclose(dataset);

    H5Fclose(file_id);
    myint n_buf = dims[0];
    offset += n_buf;
  }

  // Convert T,D as logT,logD values
  #pragma omp parallel for
  for (myint i = 0; i < Ngrid3; ++i)
    logT[i] = log10(logT[i]);

  double DensitySum = 0.;
  #pragma omp parallel for reduction(+:DensitySum)
  for (myint i = 0; i < Ngrid3; ++i)
    DensitySum += logD[i];
  double InvDensityAvg = double(Ngrid3) / DensitySum;

  #pragma omp parallel for
  for (myint i = 0; i < Ngrid3; ++i)
    logD[i] = log10(InvDensityAvg*logD[i]);

  // Print data
  if (VERBOSE) {
    cout << "\nHII_Fraction = ["
         << HII_Fraction[0] << " "
         << HII_Fraction[1] << " "
         << HII_Fraction[2] << " ... "
         << HII_Fraction[Ngrid3-3] << " "
         << HII_Fraction[Ngrid3-2] << " "
         << HII_Fraction[Ngrid3-1] << "]" << endl;

    cout << "\nlog(T) = ["
         << logT[0] << " "
         << logT[1] << " "
         << logT[2] << " ... "
         << logT[Ngrid3-3] << " "
         << logT[Ngrid3-2] << " "
         << logT[Ngrid3-1] << "]" << endl;

    cout << "\nlog(delta+1) = ["
         << logD[0] << " "
         << logD[1] << " "
         << logD[2] << " ... "
         << logD[Ngrid3-3] << " "
         << logD[Ngrid3-2] << " "
         << logD[Ngrid3-1] << "]" << endl;
  }
}
*/

//! \brief Read one 640^3 render field: <ren_dir>/<name>/<name>_NNN.hdf5, else dataset <name> of All_NNN.hdf5.
static void read_render_field(const string name, vector<float>& buf) {
  string path = ren_dir + "/" + name + "/" + name + snap_str + ".hdf5";
  struct stat st;
  if (stat(path.c_str(), &st) != 0)
    path = ren_dir + "/All/All" + snap_str + ".hdf5";
  hid_t file_id = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) error("Could not open " + path);
  hid_t dataset = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) error("No dataset " + name + " in " + path);
  hid_t dataspace = H5Dget_space(dataset);
  hsize_t dims[3] = {0, 0, 0};
  if (H5Sget_simple_extent_ndims(dataspace) != 3) error(name + " in " + path + " is not 3D");
  H5Sget_simple_extent_dims(dataspace, dims, NULL);
  if (myint(dims[0]) != Ngrid || myint(dims[1]) != Ngrid || myint(dims[2]) != Ngrid)
    error(name + " in " + path + " is not Ngrid^3");
  if (H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data()) < 0)
    error("Could not read " + name + " from " + path + " (compressed renders need HDF5_PLUGIN_PATH)");
  H5Sclose(dataspace);
  H5Dclose(dataset);
  H5Fclose(file_id);
  if (VERBOSE) cout << "\nRead " << name << " from " << path << endl;
}

static void read_render_data() {
  HII_Fraction.resize(Ngrid3);
  logT.resize(Ngrid3);
  logD.resize(Ngrid3);

#ifdef LINE_HE
  // x_HeIII = HeIII_VolumeFraction / y (per H nucleus in the renders; y = 0.24 / (4 x 0.76) = 0.0789474)
  read_render_field("HeIII_VolumeFraction", HII_Fraction);
  const float x_scale = float(4. * 0.76 / 0.24);
#else
  // x_HII = HII_VolumeFraction
  read_render_field("HII_VolumeFraction", HII_Fraction);
  const float x_scale = 1.f;
#endif
  read_render_field("Temperature", logT);
  read_render_field("Density", logD);

  float x_max = 0.f;
  #pragma omp parallel for reduction(max:x_max)
  for (myint i = 0; i < Ngrid3; ++i) {
    HII_Fraction[i] *= x_scale;
    if (HII_Fraction[i] > x_max) x_max = HII_Fraction[i];
  }
  if (x_max > 1.001f) error("ionized fraction > 1 after scaling: " + to_string(x_max));
  if (VERBOSE) cout << "\nmax ionized fraction = " << x_max << endl;

  // =========================
  // Convert Temperature
  // =========================
  #pragma omp parallel for
  for (myint i = 0; i < Ngrid3; ++i)
    logT[i] = log10(logT[i]);

  // =========================
  // Normalize Density
  // =========================
  double DensitySum = 0.;
  #pragma omp parallel for reduction(+:DensitySum)
  for (myint i = 0; i < Ngrid3; ++i)
    DensitySum += logD[i];

  double InvDensityAvg = double(Ngrid3) / DensitySum;

  #pragma omp parallel for
  for (myint i = 0; i < Ngrid3; ++i)
    logD[i] = log10(InvDensityAvg * logD[i]);

  // =========================
  // Debug print
  // =========================
  if (VERBOSE) {
    cout << "\nHII_Fraction = ["
         << HII_Fraction[0] << " "
         << HII_Fraction[1] << " "
         << HII_Fraction[2] << " ... "
         << HII_Fraction[Ngrid3-3] << " "
         << HII_Fraction[Ngrid3-2] << " "
         << HII_Fraction[Ngrid3-1] << "]" << endl;

    cout << "\nlog(T) = ["
         << logT[0] << " "
         << logT[1] << " "
         << logT[2] << " ... "
         << logT[Ngrid3-3] << " "
         << logT[Ngrid3-2] << " "
         << logT[Ngrid3-1] << "]" << endl;

    cout << "\nlog(delta+1) = ["
         << logD[0] << " "
         << logD[1] << " "
         << logD[2] << " ... "
         << logD[Ngrid3-3] << " "
         << logD[Ngrid3-2] << " "
         << logD[Ngrid3-1] << "]" << endl;
  }
}

//! \brief Apply HII fraction threshold.
static void apply_threshold() {
  // Set HII fraction to either 0 or 1
  myint n_neutral = 0;
  double x_ionized = 0.;
  #pragma omp parallel for reduction(+:n_neutral) reduction(+:x_ionized)
  for (myint i = 0; i < Ngrid3; ++i) {
    x_ionized += HII_Fraction[i];
    if (HII_Fraction[i] > 0.5) {
      HII_Fraction[i] = 1.;
    } else {
      HII_Fraction[i] = 0.;
      n_neutral++;
    }
  }

  // Print data
  if (VERBOSE) {
    cout << "\nHII_Fraction = ["
         << HII_Fraction[0] << " "
         << HII_Fraction[1] << " "
         << HII_Fraction[2] << " ... "
         << HII_Fraction[Ngrid3-3] << " "
         << HII_Fraction[Ngrid3-2] << " "
         << HII_Fraction[Ngrid3-1] << "]" << endl;
    cout << "\nGlobal x_HI = " << 1. - x_ionized / double(Ngrid3)
         << "  (x_HI_thresholds = " << double(n_neutral) / double(Ngrid3) << ")" << endl;
  }
}


//! \breif Returns the integer n, which fulfills n^2 <= arg < (n+1)^2.
static inline int isqrt(const int arg) {
  return sqrt(arg + 0.5);
}

//! \brief Calculate HII fraction gradients in units of 1/(cell widths).
static void calculate_gradients() {
  // Initial gradient calculation
  HII_Gradient.resize(Ngrid3);               // Allocate space
  #pragma omp parallel for
  for (myint i = 0; i < Ngrid3; ++i) {
    // Initialize (x,y,z) indices
    myint ip = i;                            // Temporary cell index
    const myint ix = ip / Ngrid2;            // x index
    ip -= ix * Ngrid2;                       // Hyperslab remainder
    const myint iy = ip / Ngrid;             // y index
    const myint iz = ip - iy * Ngrid;        // Column remainder
    myint ixp = ix + 1, ixm = ix - 1;        // Neighbor indices
    myint iyp = iy + 1, iym = iy - 1;
    myint izp = iz + 1, izm = iz - 1;
    if (ixp == Ngrid) ixp = 0;               // Periodic x boundaries
    if (ixm == -1) ixm = Ngrid - 1;
    if (iyp == Ngrid) iyp = 0;               // Periodic y boundaries
    if (iym == -1) iym = Ngrid - 1;
    if (izp == Ngrid) izp = 0;               // Periodic z boundaries
    if (izm == -1) izm = Ngrid - 1;
    // Note: cell = (ix * Ngrid + iy) * Ngrid + iz; // Cell index
    const myint i_xp = (ixp * Ngrid + iy) * Ngrid + iz;
    const myint i_xm = (ixm * Ngrid + iy) * Ngrid + iz;
    const myint i_yp = (ix * Ngrid + iyp) * Ngrid + iz;
    const myint i_ym = (ix * Ngrid + iym) * Ngrid + iz;
    const myint i_zp = (ix * Ngrid + iy) * Ngrid + izp;
    const myint i_zm = (ix * Ngrid + iy) * Ngrid + izm;
    const float HII_i = HII_Fraction[i];
    const float HII_xp = HII_Fraction[i_xp], HII_xm = HII_Fraction[i_xm];
    const float HII_yp = HII_Fraction[i_yp], HII_ym = HII_Fraction[i_ym];
    const float HII_zp = HII_Fraction[i_zp], HII_zm = HII_Fraction[i_zm];

    // Calculate gradients
    float grad_x = 0.5 * (HII_xp - HII_xm); // Note: dx = 1 in cell width units
    float grad_y = 0.5 * (HII_yp - HII_ym); // Usually: dHII / (2 * dx)
    float grad_z = 0.5 * (HII_zp - HII_zm);

    // Limit gradients based on neighbors
    float dphi, psi, alpha = 1.;
    float dphi_max = std::max({HII_i, HII_xp, HII_xm, HII_yp, HII_ym, HII_zp, HII_zm}) - HII_i;
    float dphi_min = HII_i - std::min({HII_i, HII_xp, HII_xm, HII_yp, HII_ym, HII_zp, HII_zm});
    dphi = 0.5 * fabs(grad_x);           // Check x (Usually: grad_x * dx / 2)
    if (dphi > 0.) {
      psi = dphi_max / dphi; if (psi < alpha) alpha = psi;
      psi = dphi_min / dphi; if (psi < alpha) alpha = psi;
    }
    dphi = 0.5 * fabs(grad_y);           // Check y
    if (dphi > 0.) {
      psi = dphi_max / dphi; if (psi < alpha) alpha = psi;
      psi = dphi_min / dphi; if (psi < alpha) alpha = psi;
    }
    dphi = 0.5 * fabs(grad_z);           // Check z
    if (dphi > 0.) {
      psi = dphi_max / dphi; if (psi < alpha) alpha = psi;
      psi = dphi_min / dphi; if (psi < alpha) alpha = psi;
    }

    // Limit gradients based on corners, i.e. restrict to [0,1]
    dphi = 0.5 * (fabs(grad_x) + fabs(grad_y) + fabs(grad_z));
    if (HII_i + dphi > 1.) {             // Highest corner
      psi = (1. - HII_i) / dphi; if (psi < alpha) alpha = psi;
    }
    if (HII_i - dphi < 0.) {             // Lowest corner
      psi = HII_i / dphi; if (psi < alpha) alpha = psi;
    }

    // Save the final gradients
    HII_Gradient[i] = Vec3(alpha*grad_x, alpha*grad_y, alpha*grad_z);
  }

  // Print data
  if (VERBOSE) {
    cout << "\nGrad(HII) = [\n ["
         << HII_Gradient[0].x << " " << HII_Gradient[0].y << " " << HII_Gradient[0].z << "]\n ["
         << HII_Gradient[1].x << " " << HII_Gradient[1].y << " " << HII_Gradient[1].z << "]\n ["
         << HII_Gradient[2].x << " " << HII_Gradient[2].y << " " << HII_Gradient[2].z << "]\n ...\n ["
         << HII_Gradient[Ngrid3-3].x << " " << HII_Gradient[Ngrid3-3].y << " " << HII_Gradient[Ngrid3-3].z << "]\n ["
         << HII_Gradient[Ngrid3-2].x << " " << HII_Gradient[Ngrid3-2].y << " " << HII_Gradient[Ngrid3-2].z << "]\n ["
         << HII_Gradient[Ngrid3-1].x << " " << HII_Gradient[Ngrid3-1].y << " " << HII_Gradient[Ngrid3-1].z << "]\n] per cell width" << endl;
  }
}

//! \brief Calculate healpix directions in RING ordering.
void initialize_healpix_directions(const int order) {
  const int nside = 1 << order;              // nside = 2^order
  const int npface = nside << order;         // npface = nside^2
  const int ncap = (npface - nside) << 1;    // ncap = 2 nside (nside - 1)
  const int npix = 12 * npface;              // npix = 12 nside^2 = 12*4^order
  const double fact2 = 4. / npix;            // Useful constants
  const double fact1 = (nside << 1) * fact2; // 8 nside / npix

  directions.resize(npix);                   // Allocate space

  #pragma omp parallel for
  for (int pix = 0; pix < npix; ++pix) {
    double z, phi, sth;                      // pix -> (z, phi, sqrt(theta))
    if (pix < ncap) {                        // North polar cap
      int iring = (1 + isqrt(1 + 2*pix)) >> 1; // Counted from north pole
      int iphi = (pix+1) - 2*iring*(iring-1);

      double tmp = (iring * iring) * fact2;
      z = 1. - tmp;
      phi = (iphi - 0.5) * M_PI_2 / iring;
      sth = (z > 0.99) ? sqrt(tmp * (2. - tmp)) : sqrt(1. - z * z);
    } else if (pix < npix - ncap) {          // Equatorial region
      int nl4 = 4 * nside;
      int ip  = pix - ncap;
      int tmp = (order >= 0) ? ip >> (order+2) : ip / nl4;
      int iring = tmp + nside;
      int iphi = ip - nl4*tmp + 1;
      // 1 if iring + nside is odd, 1/2 otherwise
      double fodd = ((iring + nside) & 1) ? 1. : 0.5;

      z = (2*nside - iring) * fact1;
      phi = (iphi - fodd) * M_PI * 0.75 * fact1;
      sth = sqrt(1. - z * z);
    } else {                                 // South polar cap
      int ip = npix - pix;
      int iring = (1 + isqrt(2*ip - 1)) >> 1; // Counted from south pole
      int iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1));

      double tmp = (iring * iring) * fact2;
      z = tmp - 1.;
      phi = (iphi - 0.5) * M_PI_2 / iring;
      sth = (z < -0.99) ? sqrt(tmp * (2. - tmp)) : sqrt(1. - z * z);
    }
    directions[pix] = {sth * cos(phi), sth * sin(phi), z};
    // cout << "\nDirection " << pix << ": (" << directions[pix][0] << " " << directions[pix][1] << " " << directions[pix][2] << ")" << endl;
  }
}


//! \brief Calculate all cell mean-free-paths and store them in a grid.
static void calculate_mfp_avgs() {
  n_bins = int(Ngrid) * n_bins_per_cell * n_loops_per_box; // Number of bins
  mfp_hist = vector<long long>(n_bins);            // Allocate space and initialize to zero
  mfp_hist_xHII = vector<long long>(n_xHII_bins * n_logR_bins);
  mfp_hist_logT = vector<long long>(n_logT_bins * n_logR_bins);
  mfp_hist_logD = vector<long long>(n_logD_bins * n_logR_bins);
  mfp_avgs = vector<float>(Ngrid3);

  // Find the min/max of logT and logD
  logR_min = log10(PixSize_cMpc / double(n_bins_per_cell));
  logR_max = log10(BoxSize_cMpc * double(n_loops_per_box));
  xHII_min = 0.5, xHII_max = 1.;
  logT_min = 10., logT_max = -10.;
  #pragma omp parallel for reduction(min:logT_min) reduction(max:logT_max)
  for (myint i = 0; i < Ngrid3; ++i) {
    if (logT[i] < logT_min)
      logT_min = logT[i]; // Update minimum
    if (logT[i] > logT_max)
      logT_max = logT[i]; // Update maximum
  }

  logD_min = 10., logD_max = -10.;
  #pragma omp parallel for reduction(min:logD_min) reduction(max:logD_max)
  for (myint i = 0; i < Ngrid3; ++i) {
    if (logD[i] < logD_min)
      logD_min = logD[i]; // Update minimum
    if (logD[i] > logD_max)
      logD_max = logD[i]; // Update maximum
  }

  // Define width constants
  iw_logR = double(n_logR_bins) / (logR_max - logR_min); // Inverse bin width
  iw_xHII = double(n_xHII_bins) / (xHII_max - xHII_min);
  iw_logT = double(n_logT_bins) / (logT_max - logT_min);
  iw_logD = double(n_logD_bins) / (logD_max - logD_min);

  #pragma omp parallel for
  for (myint i_cell = 0; i_cell < Ngrid3; ++i_cell)
    mfp_avgs[i_cell] = calculate_mfp_avg(i_cell); // Mean-free-path in units of cell widths

  // Print histogram
  if (VERBOSE) {
    cout << "\nmfp_hist = ["
         << mfp_hist[0] << " "
         << mfp_hist[1] << " "
         << mfp_hist[2] << " ... "
         << mfp_hist[n_bins-3] << " "
         << mfp_hist[n_bins-2] << " "
         << mfp_hist[n_bins-1] << "]" << endl;
    cout << "\nmfp_avgs = ["
         << mfp_avgs[0] << " "
         << mfp_avgs[1] << " "
         << mfp_avgs[2] << " ... "
         << mfp_avgs[Ngrid3-3] << " "
         << mfp_avgs[Ngrid3-2] << " "
         << mfp_avgs[Ngrid3-1] << "]" << endl;
  }
}


//! \brief Calculate average mean-free-path for a cell, returning in units of cell widths.
static double calculate_mfp_avg(const myint start_cell) {
  // Loop through all directions
  double l_tot_sum = 0.;                     // Cumulative distance over all LOS (cell width units)
  if (HII_Fraction[start_cell] <= 0.5)
    return 0.;

  // Indices in 2D histograms
  int i_xHII = floor(iw_xHII * (HII_Fraction[start_cell] - xHII_min));
  int i_logT = floor(iw_logT * (logT[start_cell] - logT_min));
  int i_logD = floor(iw_logD * (logD[start_cell] - logD_min));
  if (i_xHII < 0)
    i_xHII = 0;
  if (i_xHII >= n_xHII_bins)
    i_xHII = n_xHII_bins - 1;
  if (i_logT < 0)
    i_logT = 0;
  if (i_logT >= n_logT_bins)
    i_logT = n_logT_bins - 1;
  if (i_logD < 0)
    i_logD = 0;
  if (i_logD >= n_logD_bins)
    i_logD = n_logD_bins - 1;

  for (int i_LOS = 0; i_LOS < n_LOS; ++i_LOS) {
    const double mfp = calculate_mfp_LOS(start_cell, i_LOS); // Mean-free-path (cell width units)
    l_tot_sum += mfp; // Add LOS to the sum (cell width units)

    // 1D Histogram
    const int i_bin = floor(mfp * double(n_bins_per_cell)); // Mean-free-path bin (oversampled units)
    // if (i_bin < 0) error("i_bin < 0");
    // if (i_bin >= n_bins) error("i_bin >= n_bins");
    #pragma omp atomic                       // Atomic is for thread safety
    mfp_hist[i_bin]++;                       // Increment histogram counter

    // 2D Histograms
    const double log_mfp = log10(mfp*PixSize_cMpc); // Log mean-free-path (cMpc)
    const int i_logR = floor(iw_logR * (log_mfp - logR_min)); // Index in 2D histogram
    // if (i_logR < 0) error("i_logR < 0");
    // if (i_logR >= n_logR_bins) error("i_logR >= n_logR_bins");
    const int i_logR_xHII = i_logR * n_xHII_bins + i_xHII; // 1D index
    const int i_logR_logT = i_logR * n_logT_bins + i_logT;
    const int i_logR_logD = i_logR * n_logD_bins + i_logD;
    #pragma omp atomic
    mfp_hist_xHII[i_logR_xHII]++;            // Increment xHII histogram counter
    #pragma omp atomic
    mfp_hist_logT[i_logR_logT]++;            // Increment logT histogram counter
    #pragma omp atomic
    mfp_hist_logD[i_logR_logD]++;            // Increment logD histogram counter
  }

  return l_tot_sum / double(n_LOS);          // Average mean-free-path (cell width units)
}


//! \brief Calculate mean-free-path for one direction for a cell, returning in units of cell widths.
static double calculate_mfp_LOS(myint cell, const int i_LOS) {
  // Initialize (x,y,z) position and indices
  myint ix, iy, iz;                          // Cell indices
  myint prev_cell = cell;                    // Temporary cell indices
  ix = prev_cell / Ngrid2;                   // x index
  prev_cell -= ix * Ngrid2;                  // Hyperslab remainder
  iy = prev_cell / Ngrid;                    // y index
  iz = prev_cell - iy * Ngrid;               // Column remainder
  double x = centers[ix];                    // Cell center position
  double y = centers[iy];
  double z = centers[iz];
  double l_tot = 0.;                         // Cumulative distance (cell width units)
  double l, l_comp;                          // Comparison distances
  int direction_type = NotSet;               // Direction enumerated type
  double HII = HII_Fraction[cell], prev_HII; // Current/previous HII fractions
  double kx = directions[i_LOS][0];
  double ky = directions[i_LOS][1];
  double kz = directions[i_LOS][2];

  // Ray-tracing
  do {
    l = l_max;                               // Initialize to large value
    // x direction
    if (kx != 0.) {
      if (kx > 0.) {
        l_comp = (edges[ix+1] - x) / kx;     // Right x face
        if (l_comp < l) {
          l = l_comp;
          direction_type = PosX;
        }
      } else {
        l_comp = (edges[ix] - x) / kx;       // Left x face
        if (l_comp < l) {
          l = l_comp;
          direction_type = NegX;
        }
      }
    }
    // y direction
    if (ky != 0.) {
      if (ky > 0.) {
        l_comp = (edges[iy+1] - y) / ky;     // Right y face
        if (l_comp < l) {
          l = l_comp;
          direction_type = PosY;
        }
      } else {
        l_comp = (edges[iy] - y) / ky;       // Left y face
        if (l_comp < l) {
          l = l_comp;
          direction_type = NegY;
        }
      }
    }
    // z direction
    if (kz != 0.) {
      if (kz > 0.) {
        l_comp = (edges[iz+1] - z) / kz;     // Right z face
        if (l_comp < l) {
          l = l_comp;
          direction_type = PosZ;
        }
      } else {
        l_comp = (edges[iz] - z) / kz;       // Left z face
        if (l_comp < l) {
          l = l_comp;
          direction_type = NegZ;
        }
      }
    }
    // Avoid negative distances
    if (l < 0.)
      l = 0.;

    // Calculate the next coordinates and new HII fraction
    x += l * kx; y += l * ky; z += l * kz;
    prev_HII = HII;                          // Save for interpolation
    HII = HII_Fraction[cell]
        + HII_Gradient[cell].x * (x - centers[ix])
        + HII_Gradient[cell].y * (y - centers[iy])
        + HII_Gradient[cell].z * (z - centers[iz]);
    if (HII <= 0.5) {                        // Stop ray at x_HII = 0.5
      l *= (0.5 - prev_HII) / (HII - prev_HII); // Adjust distance
      l_tot += l;                            // Add current path length
      break;
    }

    // Update position and indices based on the direction
    switch (direction_type) {                // Actions based on direction
      case PosX:                             // Positive X
        ++ix;
        if (ix == Ngrid) {
          ix = 0;                            // Periodic boundaries
          x -= Ngrid_double;
        }
        break;
      case NegX:                             // Negative X
        --ix;
        if (ix == -1) {
          ix = Ngrid - 1;                    // Periodic boundaries
          x += Ngrid_double;
        }
        break;
      case PosY:                             // Positive Y
        ++iy;
        if (iy == Ngrid) {
          iy = 0;                            // Periodic boundaries
          y -= Ngrid_double;
        }
        break;
      case NegY:                             // Negative Y
        --iy;
        if (iy == -1) {
          iy = Ngrid - 1;                    // Periodic boundaries
          y += Ngrid_double;
        }
        break;
      case PosZ:                             // Positive Z
        ++iz;
        if (iz == Ngrid) {
          iz = 0;                            // Periodic boundaries
          z -= Ngrid_double;
        }
        break;
      case NegZ:                             // Negative Z
        --iz;
        if (iz == -1) {
          iz = Ngrid - 1;                    // Periodic boundaries
          z += Ngrid_double;
        }
        break;
      default:
        error("Unrecognized direction type in ray-tracing calculation: " + to_string(direction_type));
    }

    // Update cell index, coordinates, and HII fraction
    prev_cell = cell;                        // Reset previous cell index
    cell = (ix * Ngrid + iy) * Ngrid + iz;   // Update cell index
    HII = HII_Fraction[cell]
        + HII_Gradient[cell].x * (x - centers[ix])
        + HII_Gradient[cell].y * (y - centers[iy])
        + HII_Gradient[cell].z * (z - centers[iz]);
    l_tot += l;                              // Add current path length
  } while (HII > 0.5 && l_tot < l_max); // Limit total distance by l_max

  if (l_tot > l_max)
    l_tot = l_max;                           // Adjust overshooting distance
  if (l_tot < l_min)
    l_tot = l_min;                           // Adjust undershooting distance

  return l_tot;                              // Final mean-free-path (cell width units)
}


#define WRITE_ATTRIBUTE(attr_name, attr_value, attr_type)                                           \
  dataspace_id = H5Screate(H5S_SCALAR);                                                             \
  attribute_id = H5Acreate(group_id, attr_name, attr_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); \
  status       = H5Awrite(attribute_id, attr_type, &(attr_value));                                  \
  status       = H5Aclose(attribute_id);                                                            \
  status       = H5Sclose(dataspace_id);

struct UnitAttrs
{
  double a;  // exponent of the cosmological a factor
  double h;  // exponent of the hubble parameter
  double L;  // length unit scaling
  double M;  // mass unit scaling
  double V;  // velocity unit scaling
  double c;  // conversion factor to cgs units (zero indicates dimensionless quantity, integer count, etc)
};

/*! \brief Function for setting units of an output field.
 *
 *  \param[in/out] *ua UnitAttrs pointer to be set.
 *  \param[in] a the exponent of the cosmological a factor.
 *  \param[in] h the exponent of the hubble parameter.
 *  \param[in] L the length unit scaling.
 *  \param[in] M the mass unit scaling.
 *  \param[in] V the velocity unit scaling.
 *  \param[in] c conversion factor to cgs units (zero indicates dimensionless
 *             quantity, integer count, etc).
 *
 *  \return void
 */
static inline void set_unit_attrs(struct UnitAttrs *ua, double a, double h, double L, double M, double V, double c)
{
  ua->a = a;
  ua->h = h;
  ua->L = L;
  ua->M = M;
  ua->V = V;
  ua->c = c;
}

/*! \brief Function for adding units to an output field.
 *
 *  \param[in] file_id specifies the file location.
 *  \param[in] name specifies the dataset location relative to file_id.
 *  \param[in] ua the UnitAttrs struct holding (a,h,L,M,V,c) attributes.
 *
 *  \return void
 */
static void write_units(hid_t file_id, const char *name, struct UnitAttrs *ua)
{
  herr_t status;
  hid_t dataspace_id, attribute_id;

  hid_t group_id = H5Dopen(file_id, name, H5P_DEFAULT); // group_id is for convenience (macro)

  WRITE_ATTRIBUTE("a_scaling", ua->a, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("h_scaling", ua->h, H5T_NATIVE_DOUBLE)

  WRITE_ATTRIBUTE("length_scaling", ua->L, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("mass_scaling", ua->M, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("velocity_scaling", ua->V, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("to_cgs", ua->c, H5T_NATIVE_DOUBLE)

  H5Dclose(group_id); // Close the dataset
}

//! \brief Writes out a vector quantity, e.g. (a1,a2,a3,...)
static void write_1d(hid_t file_id, vector<long long>& vec, const char *name) {
  // Identifier
  hsize_t vec_size = vec.size();
  hid_t dataspace_id, dataset_id;
  hsize_t dims1d[1] = {vec_size};

  dataspace_id = H5Screate_simple(1, dims1d, NULL);
  dataset_id = H5Dcreate(file_id, name, H5T_NATIVE_LLONG, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);

  // Open dataset and get dataspace
  hid_t dataset   = H5Dopen(file_id, name, H5P_DEFAULT);
  hid_t filespace = H5Dget_space(dataset);

  // File hyperslab
  hsize_t file_offset[1] = {0};
  hsize_t file_count[1] = {vec_size};

  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, file_offset, NULL, file_count, NULL);

  // Memory hyperslab
  hsize_t mem_offset[1] = {0};
  hsize_t mem_count[1] = {vec_size};

  hid_t memspace = H5Screate_simple(1, mem_count, NULL);
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, NULL, mem_count, NULL);

  // Write
  H5Dwrite(dataset, H5T_NATIVE_LLONG, memspace, filespace, H5P_DEFAULT, vec.data());

  // Close handles
  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Dclose(dataset);
}

//! \brief Writes out a 2d vector quantity, e.g. ((a1,a2,a3,...), (b1,b2,b3,...), ...)
static void write_2d(hid_t file_id, vector<long long>& vec, const char *name, const hsize_t nx, const hsize_t ny) {
  // Check dimensions
  if (vec.size() != nx * ny)
    error("bad # dims [" + to_string(nx) + ", " + to_string(ny) + "], " + to_string(vec.size()));

  // Identifier
  hid_t dataspace_id, dataset_id;
  hsize_t dims2d[2] = {nx, ny};

  dataspace_id = H5Screate_simple(2, dims2d, NULL);
  dataset_id = H5Dcreate(file_id, name, H5T_NATIVE_LLONG, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);

  // Open dataset and get dataspace
  hid_t dataset   = H5Dopen(file_id, name, H5P_DEFAULT);
  hid_t filespace = H5Dget_space(dataset);

  // File hyperslab
  hsize_t file_offset[2] = {0, 0};
  hsize_t file_count[2] = {nx, ny};

  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, file_offset, NULL, file_count, NULL);

  // Memory hyperslab
  hsize_t mem_offset[2] = {0, 0};
  hsize_t mem_count[2] = {nx, ny};

  hid_t memspace = H5Screate_simple(2, mem_count, NULL);
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, NULL, mem_count, NULL);

  // Write
  H5Dwrite(dataset, H5T_NATIVE_LLONG, memspace, filespace, H5P_DEFAULT, vec.data());

  // Close handles
  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Dclose(dataset);
}

//! \brief Writes out a 3d vector quantity
static void write_3d(hid_t file_id, vector<float>& vec, const char *name, const hsize_t nx, const hsize_t ny, const hsize_t nz) {
  // Check dimensions
  if (vec.size() != nx * ny * nz)
    error("bad # dims");

  // Identifier
  hid_t dataspace_id, dataset_id;
  hsize_t dims3d[3] = {nx, ny, nz};

  dataspace_id = H5Screate_simple(3, dims3d, NULL);
  dataset_id = H5Dcreate(file_id, name, H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);

  // Open dataset and get dataspace
  hid_t dataset   = H5Dopen(file_id, name, H5P_DEFAULT);
  hid_t filespace = H5Dget_space(dataset);

  // File hyperslab
  hsize_t file_offset[3] = {0, 0, 0};
  hsize_t file_count[3] = {nx, ny, nz};

  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, file_offset, NULL, file_count, NULL);

  // Memory hyperslab
  hsize_t mem_offset[3] = {0, 0, 0};
  hsize_t mem_count[3] = {nx, ny, nz};

  hid_t memspace = H5Screate_simple(3, mem_count, NULL);
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, NULL, mem_count, NULL);

  // Write
  H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, vec.data());

  // Close handles
  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Dclose(dataset);
}

//! \brief Write data to an output file.
static void write_data() {
  string filename = out_dir + "/mfp" + snap_str + ".hdf5";

  // Identifiers
  herr_t status;
  hid_t file_id, group_id, dataspace_id, dataset_id, attribute_id;

  // Open file and write header
  file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  group_id = H5Gcreate(file_id, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // BoxSize, Time, Units, HubbleParam, Omega0, Redshift
  const double Redshift = 1.0 / a - 1.0;
  const int NumPixels = Ngrid;
  WRITE_ATTRIBUTE("BoxSize", BoxSize, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("NumPixels", NumPixels, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumBins", n_bins, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumBinsLogR", n_logR_bins, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumBinsHII", n_xHII_bins, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumBinsLogT", n_logT_bins, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumBinsLogD", n_logD_bins, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("MinLogR", logR_min, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("MaxLogR", logR_max, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("MinHII", xHII_min, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("MaxHII", xHII_max, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("MinLogT", logT_min, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("MaxLogT", logT_max, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("MinLogD", logD_min, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("MaxLogD", logD_max, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("NumExp", n_exp, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumDirections", n_LOS, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumBinsPerCell", n_bins_per_cell, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumLoopsPerBox", n_loops_per_box, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("Time", a, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("UnitLength_in_cm", UnitLength_in_cm, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("UnitMass_in_g", UnitMass_in_g, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("UnitVelocity_in_cm_per_s", UnitVelocity_in_cm_per_s, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("HubbleParam", h, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("Omega0", Omega0, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("OmegaBaryon", OmegaBaryon, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("OmegaLambda", OmegaLambda, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("Redshift", Redshift, H5T_NATIVE_DOUBLE)

  status = H5Gclose(group_id);

  // Main data
  write_1d(file_id, mfp_hist, "mfp_hist");
  write_2d(file_id, mfp_hist_xHII, "mfp_hist_xHII", n_logR_bins, n_xHII_bins);
  write_2d(file_id, mfp_hist_logT, "mfp_hist_logT", n_logR_bins, n_logT_bins);
  write_2d(file_id, mfp_hist_logD, "mfp_hist_logD", n_logR_bins, n_logD_bins);
  write_3d(file_id, mfp_avgs, "mfp_avgs", NumPixels, NumPixels, NumPixels);

  // Close file
  H5Fclose(file_id);
}
