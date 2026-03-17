# Bulk Force Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a bulk `eval_force()` API to the core C++ tree that computes all particle forces after plane-wave initialization and verifies the result against an analytic Ewald reference.

**Architecture:** Implement force evaluation as a read-only post-processing step on `HPDMKPtTree`. Reuse `form_outgoing_pw_single(...)`, `outgoing_pw`, `incoming_pw`, and the existing residual traversal structure so the force decomposition matches the current energy split: reciprocal-space `window + diff` via `ik`, residual via the derivative of the PSWF polynomial approximation.

**Tech Stack:** C++20, CMake, GoogleTest, MPI, OpenMP, complex Fourier sums

---

Follow `@superpowers/test-driven-development` for each code task. Before claiming success on any task, follow `@superpowers/verification-before-completion`.

### Task 1: Add a reference Ewald bulk-force API

**Files:**
- Modify: `include/ewald.hpp`
- Modify: `src/ewald.cpp`
- Test: `test/ewald_test.cpp`

**Step 1: Write the failing test**

Add a new test in `test/ewald_test.cpp` that constructs a small neutral system, calls `Ewald::compute_force()`, and checks two invariants:
- the returned vector length is `3 * n_particles`
- the total force sums to approximately zero in each Cartesian component

**Step 2: Run the test to verify it fails**

Run: `cmake -S . -B build && cmake --build build -j 8 --target hpdmk_test && ctest --test-dir build --output-on-failure -R Ewald`
Expected: build or link failure because `compute_force()` does not exist yet.

**Step 3: Write the minimal implementation**

Add `std::vector<double> compute_force();` to `include/ewald.hpp`. In `src/ewald.cpp`, implement:
- short-range pair forces from the derivative of `erfc(alpha r) / r`
- reciprocal-space forces from the derivative of the existing plane-wave sum using `ik_x`, `ik_y`, `ik_z`
- a flat `3 * n_particles` return vector

Match the current Ewald conventions: periodic images, stored `interaction_matrix`, and the same charge and normalization factors used by `compute_energy()`.

**Step 4: Run the test to verify it passes**

Run: `cmake --build build -j 8 --target hpdmk_test && ctest --test-dir build --output-on-failure -R Ewald`
Expected: the new Ewald force test passes.

**Step 5: Commit**

```bash
git add include/ewald.hpp src/ewald.cpp test/ewald_test.cpp
git commit -m "add ewald bulk force reference"
```

### Task 2: Add analytic kernel derivative helpers and tree API declarations

**Files:**
- Modify: `include/tree.hpp`
- Modify: `include/kernels.hpp`
- Modify: `include/pswf.hpp`

**Step 1: Write the failing test**

In `test/hpdmk_test.cpp`, add a new `compare_force(int digits)` test shell that:
- builds a tree
- calls `form_outgoing_pw()` and `form_incoming_pw()`
- calls `tree.eval_force()`
- compares its size against the Ewald force vector size

Keep the assertions minimal for now so the test fails on the missing API, not on unfinished math.

**Step 2: Run the test to verify it fails**

Run: `cmake --build build -j 8 --target hpdmk_test && ctest --test-dir build --output-on-failure -R HPDMKTest.Force`
Expected: compile failure because `eval_force()` is undeclared.

**Step 3: Write the minimal implementation**

Add to `include/tree.hpp`:
- `sctl::Vector<Real> eval_force();`
- helper declarations for `eval_force_window(...)`, `eval_force_diff(...)`, and residual accumulation helpers if needed

Add to `include/kernels.hpp`:
- helpers that expose the per-mode `k_x`, `k_y`, `k_z` factors for the half-plane storage loops, or a reusable helper that converts an index triplet to Cartesian wavenumbers

Add to `include/pswf.hpp`:
- `PolyFun<Real>::eval_derivative(Real x) const`
- a helper for the radial derivative of `residual_kernel`

Keep the interface narrow and avoid C API changes.

**Step 4: Run the test to verify it passes this stage**

Run: `cmake --build build -j 8 --target hpdmk_test`
Expected: declarations compile, but the link still fails until the implementation file is added.

**Step 5: Commit**

```bash
git add include/tree.hpp include/kernels.hpp include/pswf.hpp test/hpdmk_test.cpp
git commit -m "add bulk force api declarations"
```

### Task 3: Implement reciprocal-space bulk forces

**Files:**
- Create: `src/force.cpp`
- Modify: `CMakeLists.txt`
- Modify: `include/tree.hpp`
- Test: `test/hpdmk_test.cpp`

**Step 1: Write the failing test**

Extend the new HPDMK force test to compare the `window + diff` force output against Ewald on a moderate random neutral system after:
- `tree.form_outgoing_pw();`
- `tree.form_incoming_pw();`

Use a loose tolerance first, scaled from `digits`, and keep the residual check disabled until Task 4.

**Step 2: Run the test to verify it fails**

Run: `cmake --build build -j 8 --target hpdmk_test && ctest --test-dir build --output-on-failure -R HPDMKTest.Force`
Expected: link failure or numerical failure because `src/force.cpp` is missing or incomplete.

**Step 3: Write the minimal implementation**

Create `src/force.cpp` and add it to `CMakeLists.txt`. Implement:
- `HPDMKPtTree<Real>::eval_force()`
- a sorted-order force buffer plus a final scatter back to unsorted input order using `indices_map_sorted`
- root `window` force accumulation by differentiating the root reciprocal energy with `ik`
- level-by-level `diff` force accumulation using each particle’s local path, `form_outgoing_pw_single(...)`, `incoming_pw[node]`, and the same `delta_k[l]` / `interaction_mat[l]` conventions as the energy path

Do not mutate any stored plane-wave fields. `eval_force()` must work as a pure read-only evaluation.

**Step 4: Run the test to verify the reciprocal path passes**

Run: `cmake --build build -j 8 --target hpdmk_test && ctest --test-dir build --output-on-failure -R HPDMKTest.Force`
Expected: the reciprocal-space-only assertions pass; any residual-force assertions remain disabled or marked TODO.

**Step 5: Commit**

```bash
git add CMakeLists.txt include/tree.hpp src/force.cpp test/hpdmk_test.cpp
git commit -m "add reciprocal bulk force evaluation"
```

### Task 4: Implement residual bulk forces

**Files:**
- Modify: `src/force.cpp`
- Modify: `include/kernels.hpp`
- Modify: `include/pswf.hpp`
- Test: `test/hpdmk_test.cpp`

**Step 1: Write the failing test**

Enable the full HPDMK-vs-Ewald force comparison in `test/hpdmk_test.cpp`, including the residual contribution. Add a momentum-balance check:
- `sum(F_x)`, `sum(F_y)`, `sum(F_z)` each near zero

**Step 2: Run the test to verify it fails**

Run: `cmake --build build -j 8 --target hpdmk_test && ctest --test-dir build --output-on-failure -R HPDMKTest.Force`
Expected: numerical mismatch dominated by the missing residual force term.

**Step 3: Write the minimal implementation**

In `src/force.cpp`, add residual force accumulation that reuses the same local pair traversal as `eval_energy_res_i(...)` and `eval_energy_res_ij(...)`. For each interacting pair:
- compute the same periodic-image-adjusted separation used by the residual energy path
- evaluate the radial derivative of `P(r / cutoff) / r`
- accumulate equal-and-opposite Cartesian forces to both particles

Keep pair counting consistent with the current energy implementation so no interaction is double-counted.

**Step 4: Run the test to verify it passes**

Run: `cmake --build build -j 8 --target hpdmk_test && ctest --test-dir build --output-on-failure -R HPDMKTest.Force`
Expected: the full force comparison passes for the selected test cases.

**Step 5: Commit**

```bash
git add src/force.cpp include/kernels.hpp include/pswf.hpp test/hpdmk_test.cpp
git commit -m "add residual bulk force evaluation"
```

### Task 5: Add an independent finite-difference cross-check

**Files:**
- Modify: `test/hpdmk_test.cpp`

**Step 1: Write the failing test**

Add a tiny-system test that:
- computes `tree.eval_force()`
- perturbs one or two particle coordinates by `+h` and `-h`
- rebuilds the tree from scratch for each perturbation
- compares the analytic force component against the centered finite difference of `eval_energy()`

Use a very small particle count to keep runtime acceptable.

**Step 2: Run the test to verify it fails**

Run: `cmake --build build -j 8 --target hpdmk_test && ctest --test-dir build --output-on-failure -R HPDMKTest.Force`
Expected: mismatch if sign, normalization, or indexing is still wrong.

**Step 3: Write the minimal implementation**

Tune the force sign, normalization factors, and unsorted-order scatter in `src/force.cpp` until both:
- HPDMK vs Ewald
- HPDMK vs centered finite differences

agree within the precision-scaled tolerances.

**Step 4: Run the test to verify it passes**

Run: `cmake --build build -j 8 --target hpdmk_test && ctest --test-dir build --output-on-failure -R HPDMKTest.Force`
Expected: all force-focused tests pass.

**Step 5: Commit**

```bash
git add src/force.cpp test/hpdmk_test.cpp
git commit -m "validate bulk force against finite differences"
```

### Task 6: Run full verification

**Files:**
- Test: `test/ewald_test.cpp`
- Test: `test/hpdmk_test.cpp`

**Step 1: Run the focused force suite**

Run: `ctest --test-dir build --output-on-failure -R 'Ewald|HPDMKTest.Force'`
Expected: all focused force tests pass.

**Step 2: Run the full C++ suite**

Run: `ctest --test-dir build --output-on-failure`
Expected: all existing and new tests pass.

**Step 3: Inspect the final diff**

Run: `git diff --stat`
Expected: changes are limited to the planned C++ source, headers, CMake, and tests.

**Step 4: Commit**

```bash
git add CMakeLists.txt include/tree.hpp include/kernels.hpp include/pswf.hpp include/ewald.hpp src/force.cpp src/ewald.cpp test/ewald_test.cpp test/hpdmk_test.cpp
git commit -m "add bulk force evaluation"
```
