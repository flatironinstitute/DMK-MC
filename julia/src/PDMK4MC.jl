module PDMK4MC

using MPI
using Base: Cdouble, Cint, Clonglong
import Base: length

export HPDMKParams, hpdmk_init, DIRECT, PROXY, Tree,
       create_tree, destroy_tree!, form_outgoing_pw!, form_incoming_pw!,
       eval_energy, eval_energy_window, eval_energy_diff, eval_energy_res,
       eval_shift_energy, update_shift!

const libhpdmk = get(ENV, "HPDMK_LIBRARY", "libhpdmk")

@enum hpdmk_init::Cint begin
    DIRECT = 1
    PROXY = 2
end

Base.@kwdef struct HPDMKParams
    n_per_leaf::Cint = Cint(200)
    digits::Cint = Cint(3)
    L::Cdouble = 1.0
    prolate_order::Cdouble = Cdouble(16.0)
    init::hpdmk_init = PROXY
end

mutable struct Tree
    handle::Ptr{Cvoid}
    coords::Vector{Cdouble}
    charges::Vector{Cdouble}
    n_particles::Int
end

length(tree::Tree) = tree.n_particles

function _coords_buffer(r_src::AbstractMatrix{<:Real})
    size(r_src, 1) == 3 || throw(ArgumentError("source coordinate matrix must have size (3, N)"))
    n = size(r_src, 2)
    buf = Vector{Cdouble}(undef, 3n)
    copyto!(buf, vec(Matrix{Float64}(r_src)))
    return buf, n
end

function _coords_buffer(r_src::AbstractVector{<:Real})
    length(r_src) % 3 == 0 || throw(ArgumentError("source coordinate vector length must be a multiple of 3"))
    buf = Vector{Cdouble}(Float64.(r_src))
    n = length(buf) ÷ 3
    return buf, n
end

function _coords_buffer(r_src)
    throw(ArgumentError("unsupported container for source coordinates"))
end

_charges_buffer(charge::AbstractVector{<:Real}, n::Integer) = begin
    length(charge) == n || throw(ArgumentError("charge vector must have the same length as the number of particles"))
    Vector{Cdouble}(Float64.(charge))
end

function _to_comm(comm::MPI.Comm)
    return comm.val
end

_to_comm(comm::MPI.MPI_Comm) = comm
_to_comm(comm::Ptr) = comm
_to_comm(comm::Integer) = comm

function _to_comm(::Nothing)
    throw(ArgumentError("MPI communicator must be provided"))
end

"""
    create_tree(r_src, charge; params=HPDMKParams(), comm=MPI.COMM_WORLD)

Create a hierarchical PDMK tree from source coordinates ``r_src`` and particle charges ``charge``.

The coordinate container can either be a ``3×N`` matrix or a length ``3N`` vector with
``x₁,y₁,z₁,\ldots,x_N,y_N,z_N`` ordering.  The communicator is forwarded to the underlying MPI
implementation; make sure to call `MPI.Init` before creating a tree.
"""
function create_tree(r_src, charge; params::HPDMKParams=HPDMKParams(), comm=MPI.COMM_WORLD)
    coords, n_src = _coords_buffer(r_src)
    charges = _charges_buffer(charge, n_src)
    handle = ccall((:hpdmk_tree_create, libhpdmk), Ptr{Cvoid},
                   (MPI.MPI_Comm, HPDMKParams, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
                   _to_comm(comm), params, Cint(n_src), coords, charges)
    handle == C_NULL && error("hpdmk_tree_create returned a null handle")
    tree = Tree(handle, coords, charges, n_src)
    finalizer(destroy_tree!, tree)
    return tree
end

function destroy_tree!(tree::Tree)
    if tree.handle != C_NULL
        ccall((:hpdmk_tree_destroy, libhpdmk), Cvoid, (Ptr{Cvoid},), tree.handle)
        tree.handle = C_NULL
    end
    return nothing
end

function form_outgoing_pw!(tree::Tree)
    ccall((:hpdmk_tree_form_outgoing_pw, libhpdmk), Cvoid, (Ptr{Cvoid},), tree.handle)
    return tree
end

function form_incoming_pw!(tree::Tree)
    ccall((:hpdmk_tree_form_incoming_pw, libhpdmk), Cvoid, (Ptr{Cvoid},), tree.handle)
    return tree
end

function eval_energy(tree::Tree)
    return ccall((:hpdmk_eval_energy, libhpdmk), Cdouble, (Ptr{Cvoid},), tree.handle)
end

function eval_energy_window(tree::Tree)
    return ccall((:hpdmk_eval_energy_window, libhpdmk), Cdouble, (Ptr{Cvoid},), tree.handle)
end

function eval_energy_diff(tree::Tree)
    return ccall((:hpdmk_eval_energy_diff, libhpdmk), Cdouble, (Ptr{Cvoid},), tree.handle)
end

function eval_energy_res(tree::Tree)
    return ccall((:hpdmk_eval_energy_res, libhpdmk), Cdouble, (Ptr{Cvoid},), tree.handle)
end

"""
    eval_shift_energy(tree, idx, dx, dy, dz)

Return the energy change associated with shifting particle ``idx`` by the displacement
``(dx, dy, dz)``.  Particle indices use Julia's 1-based convention.
"""
function eval_shift_energy(tree::Tree, idx::Integer, dx::Real, dy::Real, dz::Real)
    idx < 1 && throw(ArgumentError("particle index must be positive"))
    idx > tree.n_particles && throw(BoundsError(tree, idx))
    return ccall((:hpdmk_eval_shift_energy, libhpdmk), Cdouble,
                 (Ptr{Cvoid}, Clonglong, Cdouble, Cdouble, Cdouble),
                 tree.handle, Clonglong(idx - 1), Cdouble(dx), Cdouble(dy), Cdouble(dz))
end

"""
    update_shift!(tree, idx, dx, dy, dz)

Apply a shift of particle ``idx`` by ``(dx, dy, dz)`` and update the internal tree state in place.
Indices are 1-based.
"""
function update_shift!(tree::Tree, idx::Integer, dx::Real, dy::Real, dz::Real)
    idx < 1 && throw(ArgumentError("particle index must be positive"))
    idx > tree.n_particles && throw(BoundsError(tree, idx))
    ccall((:hpdmk_update_shift, libhpdmk), Cvoid,
          (Ptr{Cvoid}, Clonglong, Cdouble, Cdouble, Cdouble),
          tree.handle, Clonglong(idx - 1), Cdouble(dx), Cdouble(dy), Cdouble(dz))
    return tree
end

end # module
