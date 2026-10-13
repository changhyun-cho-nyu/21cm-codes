#define NSNAPS 709

void load_redshift(void);
void open_sfrd_file(void);
void close_sfrd_file(void);

void read_cell_sfh(int i, int j, int k, double sfrd[NSNAPS]);
void read_sfrd_jrow(int i_global, int j, int N_grid, double *jrow);
void read_sfrd_islab(int i_global, int N_grid, double *slab, double *buf);
void open_sfrd_file_parallel(MPI_Comm comm);

/* Precompute the J_alpha coupling weights (once per snapshot).
   weights[s] such that:
     x_alpha = compute_S_alpha(z,T_K,HI_frac) * dot(weights, sfrd_cell)
   Folds in: Lyman integration, f_rec, epsilon_b, H(z), (1+z)^2/(4pi),
             and the x_alpha prefactor. Only S_alpha is left per-cell. */
double epsilon_b(double nu);
extern double redshift[NSNAPS];

void precompute_Jalpha_weights(double z, double *weights);

/* S_alpha suppression factor — the only per-cell part of x_alpha */
double compute_S_alpha(double z, double T_K, double HI_frac);

double x_c(double z,
           double T_K,
           double n_H,
           double n_e,
           double n_p);

double x_alpha(double z,
               double T_K,
               double HI_frac,
               double sfrd_cell[NSNAPS]);

double T_spin(double z,
              double T_K,
              double x_c,
              double x_alpha);
