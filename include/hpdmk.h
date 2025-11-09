#ifndef HPDMK_H
#define HPDMK_H

#include <mpi.h>

typedef enum : int {
    DIRECT = 1,
    PROXY = 2,
} hpdmk_init;

typedef struct HPDMKParams {
    int n_per_leaf = 200; // maximum number of particles per leaf
    int digits = 3; // number of digits of accuracy
    double L; // length of the box
    double prolate_order = 16; // order of the prolate polynomial
    hpdmk_init init = PROXY; // method to initialize the outgoing planewave, DIRECT means direct calculation on all nodes, PROXY for proxy charge
} HPDMKParams;

typedef void *hpdmk_tree;

#ifdef __cplusplus
extern "C" {
#endif


int hpdmk_mpi_init(void);
int hpdmk_mpi_initialized(void);
MPI_Comm hpdmk_comm_world(void);

hpdmk_tree hpdmk_tree_create(MPI_Comm comm, HPDMKParams params, int n_src, const double *r_src, const double *charge);
hpdmk_tree hpdmk_tree_create_f(MPI_Comm comm, HPDMKParams params, int n_src, const float *r_src, const float *charge);

void hpdmk_tree_destroy(hpdmk_tree tree);
void hpdmk_tree_destroy_f(hpdmk_tree tree);

void hpdmk_tree_form_outgoing_pw(hpdmk_tree tree);
void hpdmk_tree_form_outgoing_pw_f(hpdmk_tree tree);
void hpdmk_tree_form_incoming_pw(hpdmk_tree tree);
void hpdmk_tree_form_incoming_pw_f(hpdmk_tree tree);

double hpdmk_eval_energy(hpdmk_tree tree);
float hpdmk_eval_energy_f(hpdmk_tree tree);
double hpdmk_eval_energy_window(hpdmk_tree tree);
float hpdmk_eval_energy_window_f(hpdmk_tree tree);
double hpdmk_eval_energy_diff(hpdmk_tree tree);
float hpdmk_eval_energy_diff_f(hpdmk_tree tree);
double hpdmk_eval_energy_res(hpdmk_tree tree);
float hpdmk_eval_energy_res_f(hpdmk_tree tree);

double hpdmk_eval_shift_energy(hpdmk_tree tree, long long i_particle, double dx, double dy, double dz);
float hpdmk_eval_shift_energy_f(hpdmk_tree tree, long long i_particle, float dx, float dy, float dz);
void hpdmk_update_shift(hpdmk_tree tree, long long i_particle, double dx, double dy, double dz);
void hpdmk_update_shift_f(hpdmk_tree tree, long long i_particle, float dx, float dy, float dz);

#ifdef __cplusplus
}
#endif

#endif
