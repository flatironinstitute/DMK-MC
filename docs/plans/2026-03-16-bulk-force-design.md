# Bulk Force Evaluation Design

## Goal
Add a C++-only bulk force evaluation path to `HPDMKPtTree` that computes the force on every particle after the tree, outgoing plane waves, and incoming plane waves have already been initialized.

## Scope
- Add a public `eval_force()` API on the core tree type only.
- Compute reciprocal-space force contributions for the `window` and `difference` terms using analytic `ik` differentiation of the existing plane-wave representation.
- Compute the `residual` contribution from the analytic derivative of the existing real-space polynomial approximation.
- Add an Ewald bulk-force reference for correctness tests.

## Non-Goals
- No C API changes in `include/hpdmk.h`.
- No Julia wrapper changes.
- No tree update or accepted-move logic changes.

## Public API
Expose `sctl::Vector<Real> eval_force();` on `HPDMKPtTree<Real>`. The returned vector has length `3 * n_particles` and is ordered in the original unsorted particle order, matching the input coordinates used to construct the tree.

The method assumes `form_outgoing_pw()` and `form_incoming_pw()` have already been called. It is a read-only evaluation step and does not rebuild or mutate tree state.

## Reciprocal-Space Formulation
The implementation is particle-centric. For each particle in sorted order:

1. Build that particle’s single-particle plane-wave coefficients with `form_outgoing_pw_single(...)`.
2. For the root (`window`) contribution, contract those coefficients against the already-built root field using the existing interaction weights and `k_x`, `k_y`, `k_z` factors from differentiating `exp(-i k \cdot x)`.
3. For each internal level along the particle path, accumulate the `difference` contribution by contracting the particle’s local coefficients against `incoming_pw[node]` with the corresponding interaction weights and `ik` factors.

This reuses the same wavenumber grids, half-plane storage, and field state that the existing energy path uses. The force path does not route through `eval_shift_energy(...)`.

## Residual Formulation
The residual term stays local and pairwise. Reuse the same leaf/self and leaf-neighbor traversal structure as the current residual energy evaluation. For each accepted pair interaction, compute the scalar derivative of

`P(r / cutoff) / r`

with respect to `r`, then project along the particle separation vector to accumulate equal-and-opposite force contributions. This preserves symmetry and avoids double counting.

## Reference Validation
Extend `Ewald` with a bulk `compute_force()` reference in core C++ only. Its short-range and reciprocal-space pieces should both be analytic, not finite-difference approximations. Then add HPDMK tests that:

- compare the full HPDMK force vector against Ewald on random neutral systems;
- check that the total force sums to approximately zero;
- spot-check a few components against centered finite differences of the total HPDMK energy rebuilt from scratch.

## Expected File Touches
- `include/tree.hpp`
- `src/force.cpp`
- `CMakeLists.txt`
- `include/kernels.hpp`
- `include/pswf.hpp`
- `include/ewald.hpp`
- `src/ewald.cpp`
- `test/hpdmk_test.cpp`
