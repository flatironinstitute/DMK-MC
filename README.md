# Periodic dual-space multilevel kernel-splitting method for Monte Carlo simulation

This is a C++ implementation of the periodic dual-space multilevel kernel-splitting method for Monte Carlo simulation long-range electrostatics systems.

## Work done

- [x] set up cpp project and ci
- [x] implement the ewald summation
- [x] implement the planewave version of hpdmk
- [x] detailed unit tests
- [x] replace the Gaussian kernel with PSWF kernel
- [x] arbitrary single particle shifting and system update
- [x] incoming planewave and energy update

## Future to do list

- [ ] user level interface (not necessary for now)

## How to use

1. Clone the repository
2. Run `git submodule update --init --recursive` to clone the submodules
3. Run `cmake . -B build` to create the build directory
4. Run `cmake --build build -j 8` to build the project with 8 threads
5. Run `ctest --test-dir build` to run the tests

Tips: for usage on `rusty`, need to run `module load intel-oneapi-compilers fftw openmpi`

## Julia interface

A lightweight Julia wrapper is provided in the [`julia/`](julia) directory. After building the
shared library (`libhpdmk`), activate the Julia project and construct trees directly from Julia:

```julia
using Pkg

Pkg.activate("julia")
using PDMK4MC

params = PDMK4MC.HPDMKParams(L = 20.0)
coords = rand(3, 100)
charges = randn(100)
tree = PDMK4MC.create_tree(coords, charges; params=params)
energy = PDMK4MC.eval_energy(tree)
```

By default the wrapper looks for `libhpdmk` using the standard library search path. Set the
environment variable `HPDMK_LIBRARY` to point to the shared library if it lives in a non-standard
location. The bindings ask `libhpdmk` itself to initialise MPI so that Julia and the native library
always share the same MPI runtime; the optional `comm` keyword can be left as `nothing` to use the
library's `MPI_COMM_WORLD`, or set to an existing communicator (for example `MPI.COMM_WORLD`). The
`precision` keyword selects either `Float64` (default) or `Float32` computations.
