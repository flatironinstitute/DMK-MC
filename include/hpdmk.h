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


hpdmk_tree hpdmk_tree_create(MPI_Comm comm, HPDMKParams params, int n_src, const double *r_src, const double *charge);
void hpdmk_tree_destroy(hpdmk_tree tree);

void hpdmk_tree_form_outgoing_pw(hpdmk_tree tree);
void hpdmk_tree_form_incoming_pw(hpdmk_tree tree);

double hpdmk_eval_energy(hpdmk_tree tree);
double hpdmk_eval_energy_window(hpdmk_tree tree);
double hpdmk_eval_energy_diff(hpdmk_tree tree);
double hpdmk_eval_energy_res(hpdmk_tree tree);

double hpdmk_eval_shift_energy(hpdmk_tree tree, long long i_particle, double dx, double dy, double dz);
void hpdmk_update_shift(hpdmk_tree tree, long long i_particle, double dx, double dy, double dz);

#ifdef __cplusplus
}
#endif

#endif
