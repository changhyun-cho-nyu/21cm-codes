/*
 * jalpha_transmission.h
 * =====================
 * Optional IGM transmission of the He II Lyman band for compute_Jalpha_He.c
 * and compute_Jalpha_AGN.c (species He). Compile with
 *     -DTRANSMISSION_DIR='"/path/to/transmission"'
 * to read <dir>/transmission_He_NNN.hdf5 (compute_transmission_He.c) for the
 * target snapshot. Each shell average of the emissivity at radius R is then
 * multiplied, voxel by voxel, by T(x, R): the direction-averaged transmission
 * from distance R to x, interpolated linearly in ln R between the stored
 * radii (T = 1 at R = 0) and trilinearly (periodic) from the origin grid to
 * the voxels. Line n = 2 uses T_n2, lines n >= 4 use T_n4.
 * Without TRANSMISSION_DIR nothing changes.
 */
#ifndef JALPHA_TRANSMISSION_H
#define JALPHA_TRANSMISSION_H

#ifdef TRANSMISSION_DIR
typedef struct {
    int    loaded_k;      /* number of radii held in T (0 = none) */
    float *T;             /* [loaded_k][No^3]                     */
} TransGroup;

static int        TR_NR = 0, TR_No = 0;
static double    *TR_R = NULL, *TR_mean2 = NULL, *TR_mean4 = NULL;
static TransGroup TR_G[2];
static char       TR_FILE[2048];
static int       *TR_i0 = NULL, *TR_i1 = NULL;   /* fine voxel -> origin grid, per axis */
static float     *TR_w1 = NULL;
static float     *TR_slice = NULL;               /* T at one radius on the origin grid  */

static void trans_init(int snap, double z_obs, double box_mpc, int N, int rank) {
    snprintf(TR_FILE, sizeof TR_FILE, "%s/transmission_He_%03d.hdf5", TRANSMISSION_DIR, snap);
    hid_t f = H5Fopen(TR_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) { fprintf(stderr, "ERROR: cannot open %s\n", TR_FILE); MPI_Abort(MPI_COMM_WORLD, 1); }
    double z, box;
    hid_t a = H5Aopen(f, "Redshift", H5P_DEFAULT);     H5Aread(a, H5T_NATIVE_DOUBLE, &z);   H5Aclose(a);
    a = H5Aopen(f, "BoxSize_cMpc", H5P_DEFAULT);        H5Aread(a, H5T_NATIVE_DOUBLE, &box); H5Aclose(a);
    if (fabs(z - z_obs) > 1e-4 || fabs(box - box_mpc) > 1e-3 * box_mpc) {
        fprintf(stderr, "ERROR: %s is for z = %.4f, box %.2f; this run has z = %.4f, box %.2f\n",
                TR_FILE, z, box, z_obs, box_mpc);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double sfm = 0.0, llsA = 0.0;                     /* recorded so the log says which variant is used */
    if (H5Aexists(f, "sf_mask") > 0) { a = H5Aopen(f, "sf_mask", H5P_DEFAULT); H5Aread(a, H5T_NATIVE_DOUBLE, &sfm); H5Aclose(a); }
    if (H5Aexists(f, "LLS_A") > 0)   { a = H5Aopen(f, "LLS_A", H5P_DEFAULT);   H5Aread(a, H5T_NATIVE_DOUBLE, &llsA); H5Aclose(a); }
    hid_t d = H5Dopen(f, "T_n2", H5P_DEFAULT), s = H5Dget_space(d);
    hsize_t dims[4]; H5Sget_simple_extent_dims(s, dims, NULL); H5Sclose(s); H5Dclose(d);
    TR_NR = (int)dims[0]; TR_No = (int)dims[1];
    TR_R = malloc(TR_NR * sizeof(double)); TR_mean2 = malloc(TR_NR * sizeof(double)); TR_mean4 = malloc(TR_NR * sizeof(double));
    d = H5Dopen(f, "R_cMpc", H5P_DEFAULT);    H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, TR_R);     H5Dclose(d);
    d = H5Dopen(f, "mean_T_n2", H5P_DEFAULT); H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, TR_mean2); H5Dclose(d);
    d = H5Dopen(f, "mean_T_n4", H5P_DEFAULT); H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, TR_mean4); H5Dclose(d);
    H5Fclose(f);
    TR_G[0].loaded_k = TR_G[1].loaded_k = 0; TR_G[0].T = TR_G[1].T = NULL;
    /* per-axis trilinear maps from the N fine cells to the No origin cells (periodic) */
    TR_i0 = malloc(N * sizeof(int)); TR_i1 = malloc(N * sizeof(int)); TR_w1 = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        double u = (i + 0.5) * TR_No / (double)N - 0.5;
        int i0 = (int)floor(u);
        TR_w1[i] = (float)(u - i0);
        TR_i0[i] = ((i0 % TR_No) + TR_No) % TR_No;
        TR_i1[i] = (TR_i0[i] + 1) % TR_No;
    }
    TR_slice = malloc((size_t)TR_No * TR_No * TR_No * sizeof(float));
    if (rank == 0)
        printf("  IGM transmission: %s (%d radii %.3f .. %.1f cMpc, origin grid %d^3; star-forming cells %s, "
               "LLS term %s)\n", TR_FILE, TR_NR, TR_R[0], TR_R[TR_NR - 1], TR_No, sfm > 0 ? "masked" : "kept",
               llsA > 0 ? "on" : "off");
}

/* make sure group g holds all radii up to R_need (reads a hyperslab of radii) */
static void trans_need(int g, double R_need) {
    int k = 0;
    while (k < TR_NR - 1 && TR_R[k] < R_need) k++;
    int nk = k + 1;
    if (TR_G[g].loaded_k >= nk) return;
    size_t No3 = (size_t)TR_No * TR_No * TR_No;
    free(TR_G[g].T);
    TR_G[g].T = malloc((size_t)nk * No3 * sizeof(float));
    hid_t f = H5Fopen(TR_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t d = H5Dopen(f, g == 0 ? "T_n2" : "T_n4", H5P_DEFAULT), s = H5Dget_space(d);
    hsize_t off[4] = {0, 0, 0, 0}, cnt[4] = {(hsize_t)nk, (hsize_t)TR_No, (hsize_t)TR_No, (hsize_t)TR_No};
    H5Sselect_hyperslab(s, H5S_SELECT_SET, off, NULL, cnt, NULL);
    hid_t m = H5Screate_simple(4, cnt, NULL);
    if (!TR_G[g].T || H5Dread(d, H5T_NATIVE_FLOAT, m, s, H5P_DEFAULT, TR_G[g].T) < 0) {
        fprintf(stderr, "ERROR: cannot read %d radii of %s from %s\n", nk, g == 0 ? "T_n2" : "T_n4", TR_FILE);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    H5Sclose(m); H5Sclose(s); H5Dclose(d); H5Fclose(f);
    TR_G[g].loaded_k = nk;
}

/* box-mean transmission at radius R (for the scalar check) */
static double trans_mean(int g, double R) {
    const double *m = g == 0 ? TR_mean2 : TR_mean4;
    if (R <= TR_R[0]) return 1.0 + (m[0] - 1.0) * R / TR_R[0];
    if (R >= TR_R[TR_NR - 1]) return m[TR_NR - 1];
    int k = 0; while (TR_R[k + 1] < R) k++;
    double q = log(R / TR_R[k]) / log(TR_R[k + 1] / TR_R[k]);
    return m[k] + q * (m[k + 1] - m[k]);
}

/* field[i] *= T(x_i, R) for line group g on the N^3 grid */
static void trans_apply(int g, double R, float *field, int N) {
    size_t No3 = (size_t)TR_No * TR_No * TR_No;
    const float *T = TR_G[g].T;
    if (R <= TR_R[0]) {
        double q = R / TR_R[0];
        #pragma omp parallel for num_threads(N_OMP)
        for (size_t o = 0; o < No3; o++) TR_slice[o] = (float)(1.0 + q * (T[o] - 1.0));
    } else {
        int k = 0; while (k < TR_G[g].loaded_k - 2 && TR_R[k + 1] < R) k++;
        double q = log(R / TR_R[k]) / log(TR_R[k + 1] / TR_R[k]);
        if (q > 1.0) q = 1.0;
        const float *A = T + (size_t)k * No3, *B = T + (size_t)(k + 1) * No3;
        #pragma omp parallel for num_threads(N_OMP)
        for (size_t o = 0; o < No3; o++) TR_slice[o] = (float)(A[o] + q * (B[o] - A[o]));
    }
    const int No = TR_No;
    #pragma omp parallel for collapse(2) num_threads(N_OMP)
    for (int ix = 0; ix < N; ix++)
    for (int iy = 0; iy < N; iy++) {
        int a0 = TR_i0[ix], a1 = TR_i1[ix], b0 = TR_i0[iy], b1 = TR_i1[iy];
        float wa = TR_w1[ix], wb = TR_w1[iy];
        const float *p00 = TR_slice + ((size_t)a0 * No + b0) * No, *p01 = TR_slice + ((size_t)a0 * No + b1) * No;
        const float *p10 = TR_slice + ((size_t)a1 * No + b0) * No, *p11 = TR_slice + ((size_t)a1 * No + b1) * No;
        float *row = field + ((size_t)ix * N + iy) * N;
        for (int iz = 0; iz < N; iz++) {
            int c0 = TR_i0[iz], c1 = TR_i1[iz]; float wc = TR_w1[iz];
            float v0 = (1 - wb) * ((1 - wc) * p00[c0] + wc * p00[c1]) + wb * ((1 - wc) * p01[c0] + wc * p01[c1]);
            float v1 = (1 - wb) * ((1 - wc) * p10[c0] + wc * p10[c1]) + wb * ((1 - wc) * p11[c0] + wc * p11[c1]);
            row[iz] *= (1 - wa) * v0 + wa * v1;
        }
    }
}
#endif /* TRANSMISSION_DIR */
#endif
