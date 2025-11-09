module PDMK4MC

using MPI
using Base: Cdouble, Cfloat, Cint, Clonglong
import Base: length

export HPDMKParams, hpdmk_init, DIRECT, PROXY, Tree,
       create_tree, destroy_tree!, form_outgoing_pw!, form_incoming_pw!,
       eval_energy, eval_energy_window, eval_energy_diff, eval_energy_res,
       eval_shift_energy, update_shift!

const libhpdmk = get(ENV, "HPDMK_LIBRARY", "libhpdmk")

function _hpdmk_mpi_init()
    initialized = ccall((:hpdmk_mpi_initialized, libhpdmk), Cint, ())
    if initialized == 0
        status = ccall((:hpdmk_mpi_init, libhpdmk), Cint, ())
        status != 0 || error("hpdmk_mpi_init failed")
    end
    return nothing
end

function _hpdmk_comm_world()
    return ccall((:hpdmk_comm_world, libhpdmk), MPI.MPI_Comm, ())
end

function __init__()
    _hpdmk_mpi_init()
end

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

mutable struct Tree{T<:AbstractFloat}
    handle::Ptr{Cvoid}
    coords::Vector{T}
    charges::Vector{T}
    n_particles::Int
end

length(tree::Tree) = tree.n_particles

_as_precision(::Type{T}) where {T<:AbstractFloat} = T
_as_precision(precision::Nothing) = nothing
_as_precision(::Any) = throw(ArgumentError("precision must be either Float32 or Float64"))

function _resolve_precision(precision, r_src, charge)
    T = _as_precision(precision)
    if T === nothing
        coords_T = eltype(r_src)
        charge_T = eltype(charge)
        if coords_T <: AbstractFloat && charge_T <: AbstractFloat && coords_T <: Float32 && charge_T <: Float32
            return Float32
        end
        return Float64
    end
    return T
end

function _coords_buffer(::Type{T}, r_src::AbstractMatrix{<:Real}) where {T<:AbstractFloat}
    size(r_src, 1) == 3 || throw(ArgumentError("source coordinate matrix must have size (3, N)"))
    n = size(r_src, 2)
    buf = Vector{T}(undef, 3n)
    copyto!(buf, vec(Matrix{T}(r_src)))
    return buf, n
end

function _coords_buffer(::Type{T}, r_src::AbstractVector{<:Real}) where {T<:AbstractFloat}
    length(r_src) % 3 == 0 || throw(ArgumentError("source coordinate vector length must be a multiple of 3"))
    buf = Vector{T}(T.(r_src))
    n = length(buf) ÷ 3
    return buf, n
end

function _coords_buffer(::Type{T}, r_src) where {T<:AbstractFloat}
    throw(ArgumentError("unsupported container for source coordinates"))
end

_charges_buffer(::Type{T}, charge::AbstractVector{<:Real}, n::Integer) where {T<:AbstractFloat} = begin
    length(charge) == n || throw(ArgumentError("charge vector must have the same length as the number of particles"))
    Vector{T}(T.(charge))
end

_to_comm(comm::MPI.Comm) = comm.val
_to_comm(comm::MPI.MPI_Comm) = comm
_to_comm(comm::Ptr) = comm
_to_comm(comm::Integer) = comm
_to_comm(::Nothing) = _hpdmk_comm_world()

"""
    create_tree(r_src, charge; params=HPDMKParams(), comm=nothing, precision=nothing)

Create a hierarchical PDMK tree from source coordinates ``r_src`` and particle charges ``charge``.

The coordinate container can either be a ``3×N`` matrix or a length ``3N`` vector with
``x₁,y₁,z₁,\ldots,x_N,y_N,z_N`` ordering.  MPI is initialised automatically through ``libhpdmk``
itself so that the Julia bindings always talk to the same MPI implementation as the native code.
Passing ``comm=nothing`` uses the library's ``MPI_COMM_WORLD``; callers that already have a
communicator (for example from MPI.jl) can provide it explicitly.  The optional ``precision``
keyword controls the floating-point type (`Float32` or `Float64`); by default it is inferred from
the input arrays.
"""
function create_tree(r_src, charge; params::HPDMKParams=HPDMKParams(), comm=nothing, precision=nothing)
    _hpdmk_mpi_init()
    T = _resolve_precision(precision, r_src, charge)
    coords, n_src = _coords_buffer(T, r_src)
    charges = _charges_buffer(T, charge, n_src)
    handle = if T === Float32
        ccall((:hpdmk_tree_create_f, libhpdmk), Ptr{Cvoid},
              (MPI.MPI_Comm, HPDMKParams, Cint, Ptr{Cfloat}, Ptr{Cfloat}),
              _to_comm(comm), params, Cint(n_src), coords, charges)
    else
        ccall((:hpdmk_tree_create, libhpdmk), Ptr{Cvoid},
              (MPI.MPI_Comm, HPDMKParams, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
              _to_comm(comm), params, Cint(n_src), coords, charges)
    end
    handle == C_NULL && error("hpdmk_tree_create returned a null handle")
    tree = Tree{T}(handle, coords, charges, n_src)
    finalizer(destroy_tree!, tree)
    return tree
end

function destroy_tree!(tree::Tree{Float64})
    if tree.handle != C_NULL
        ccall((:hpdmk_tree_destroy, libhpdmk), Cvoid, (Ptr{Cvoid},), tree.handle)
        tree.handle = C_NULL
    end
    return nothing
end

function destroy_tree!(tree::Tree{Float32})
    if tree.handle != C_NULL
        ccall((:hpdmk_tree_destroy_f, libhpdmk), Cvoid, (Ptr{Cvoid},), tree.handle)
        tree.handle = C_NULL
    end
    return nothing
end

function form_outgoing_pw!(tree::Tree{Float64})
    ccall((:hpdmk_tree_form_outgoing_pw, libhpdmk), Cvoid, (Ptr{Cvoid},), tree.handle)
    return tree
end

function form_outgoing_pw!(tree::Tree{Float32})
    ccall((:hpdmk_tree_form_outgoing_pw_f, libhpdmk), Cvoid, (Ptr{Cvoid},), tree.handle)
    return tree
end

function form_incoming_pw!(tree::Tree{Float64})
    ccall((:hpdmk_tree_form_incoming_pw, libhpdmk), Cvoid, (Ptr{Cvoid},), tree.handle)
    return tree
end

function form_incoming_pw!(tree::Tree{Float32})
    ccall((:hpdmk_tree_form_incoming_pw_f, libhpdmk), Cvoid, (Ptr{Cvoid},), tree.handle)
    return tree
end

eval_energy(tree::Tree{Float64}) =
    ccall((:hpdmk_eval_energy, libhpdmk), Cdouble, (Ptr{Cvoid},), tree.handle)

eval_energy(tree::Tree{Float32}) =
    ccall((:hpdmk_eval_energy_f, libhpdmk), Cfloat, (Ptr{Cvoid},), tree.handle)

eval_energy_window(tree::Tree{Float64}) =
    ccall((:hpdmk_eval_energy_window, libhpdmk), Cdouble, (Ptr{Cvoid},), tree.handle)

eval_energy_window(tree::Tree{Float32}) =
    ccall((:hpdmk_eval_energy_window_f, libhpdmk), Cfloat, (Ptr{Cvoid},), tree.handle)

eval_energy_diff(tree::Tree{Float64}) =
    ccall((:hpdmk_eval_energy_diff, libhpdmk), Cdouble, (Ptr{Cvoid},), tree.handle)

eval_energy_diff(tree::Tree{Float32}) =
    ccall((:hpdmk_eval_energy_diff_f, libhpdmk), Cfloat, (Ptr{Cvoid},), tree.handle)

eval_energy_res(tree::Tree{Float64}) =
    ccall((:hpdmk_eval_energy_res, libhpdmk), Cdouble, (Ptr{Cvoid},), tree.handle)

eval_energy_res(tree::Tree{Float32}) =
    ccall((:hpdmk_eval_energy_res_f, libhpdmk), Cfloat, (Ptr{Cvoid},), tree.handle)

"""
    eval_shift_energy(tree, idx, dx, dy, dz)

Return the energy change associated with shifting particle ``idx`` by the displacement
``(dx, dy, dz)``.  Particle indices use Julia's 1-based convention.
"""
function eval_shift_energy(tree::Tree{T}, idx::Integer, dx::Real, dy::Real, dz::Real) where {T<:AbstractFloat}
    idx < 1 && throw(ArgumentError("particle index must be positive"))
    idx > tree.n_particles && throw(BoundsError(tree, idx))
    if T === Float32
        return ccall((:hpdmk_eval_shift_energy_f, libhpdmk), Cfloat,
                     (Ptr{Cvoid}, Clonglong, Cfloat, Cfloat, Cfloat),
                     tree.handle, Clonglong(idx - 1), Cfloat(dx), Cfloat(dy), Cfloat(dz))
    else
        return ccall((:hpdmk_eval_shift_energy, libhpdmk), Cdouble,
                     (Ptr{Cvoid}, Clonglong, Cdouble, Cdouble, Cdouble),
                     tree.handle, Clonglong(idx - 1), Cdouble(dx), Cdouble(dy), Cdouble(dz))
    end
end

"""
    update_shift!(tree, idx, dx, dy, dz)

Apply a shift of particle ``idx`` by ``(dx, dy, dz)`` and update the internal tree state in place.
Indices are 1-based.
"""
function update_shift!(tree::Tree{T}, idx::Integer, dx::Real, dy::Real, dz::Real) where {T<:AbstractFloat}
    idx < 1 && throw(ArgumentError("particle index must be positive"))
    idx > tree.n_particles && throw(BoundsError(tree, idx))
    if T === Float32
        ccall((:hpdmk_update_shift_f, libhpdmk), Cvoid,
              (Ptr{Cvoid}, Clonglong, Cfloat, Cfloat, Cfloat),
              tree.handle, Clonglong(idx - 1), Cfloat(dx), Cfloat(dy), Cfloat(dz))
    else
        ccall((:hpdmk_update_shift, libhpdmk), Cvoid,
              (Ptr{Cvoid}, Clonglong, Cdouble, Cdouble, Cdouble),
              tree.handle, Clonglong(idx - 1), Cdouble(dx), Cdouble(dy), Cdouble(dz))
    end
    return tree
end

end # module
