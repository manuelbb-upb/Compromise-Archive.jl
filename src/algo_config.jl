# In this file, methods for the abstract type `AbstractConfig` are defined.
# An implementation of a subtype can overwrite these. 
# Else, they work as a default. 

Broadcast.broadcastable( ac :: AbstractConfig ) = Ref( ac )

#=================================================================================
DefaultConfig
=================================================================================#
struct DefaultConfig <: AbstractConfig end
const DEFAULT_CONFIG = DefaultConfig()

#=
# Criticality Test parameters 
# # ω threshold
_eps_crit( :: AbstractConfig ) = 0.001f0
# # shrinking in critical loop
_gamma_crit( ::AbstractConfig ) = 0.51f0

# # maximum number of loops before exiting
max_critical_loops( :: AbstractConfig ) :: Int = 5

# is a database used? if yes, what is the type?
use_db( :: AbstractConfig ) :: Bool = true 

# should iterations, where the models are not fully linear, be counted for stopping?
#count_nonlinear_iterations( :: AbstractConfig ) :: Bool = true
=#

# initial box radius (for fully constrained problems this is relative to ``[0,1]^n```)
delta_0(::AbstractConfig) = 0.1f0

# radius upper bound(s)
#delta_max(::AbstractConfig) = 0.5f0

# STOPPING 
# restrict number of evaluations and iterations
max_evals( :: AbstractConfig ) :: Int = typemax(Int)
max_iter( :: AbstractConfig ) :: Int = 50

var_scaler( :: AbstractConfig ) :: Union{AbstractAffineScaler,Nothing} = nothing
#=
# relative stopping 
# stop if ||Δf|| ≤ ε ||f|| (or |Δf_ℓ| .≤ ε |f_ℓ| )
f_tol_rel( :: AbstractConfig ) = sqrt(eps(MIN_PRECISION))
# stop if ||Δx|| ≤ ε ||x||
x_tol_rel( ac :: AbstractConfig ) = f_tol_rel(ac)

# absolute stopping
f_tol_abs( :: AbstractConfig ) = -1
x_tol_abs( :: AbstractConfig ) = -1

# stop if ω ≤ omega_tol_rel && Δ .≤ Δ_tol_rel
omega_tol_rel( ac :: AbstractConfig ) = 10 * f_tol_rel( ac )[end]
delta_tol_rel( ac :: AbstractConfig ) = x_tol_rel( ac )[end]

# stop if ω <= omega_tol_abs 
omega_tol_abs(:: AbstractConfig ) = -INF 

# stop if Δ .<= Δ_tol_abs 
delta_tol_abs(ac :: AbstractConfig ) = f_tol_rel( ac )

has_stop_val( ac :: AbstractConfig ) = nothing
stop_val( ac :: AbstractConfig, ind :: FunctionIndex ) = -Inf
stop_val_sense( ac :: AbstractConfig, ind :: FunctionIndex ) = :upper_bound
stop_val_only_if_feasible( ac :: AbstractConfig ) = false

# what method to use for the subproblems
descent_method( :: AbstractConfig ) :: Union{AbstractDescentConfig, Nothing} = nothing

# acceptance test parameters
strict_acceptance_test( :: AbstractConfig ) :: Bool = true
_nu_success( :: AbstractConfig ) = 0.2f0
_nu_accept(::AbstractConfig) = 0

_mu(::AbstractConfig) = Float32(2e3)
_beta(::AbstractConfig) = Float32(1e3)

# Parameters for the radius update
radius_update_method(::AbstractConfig)::Symbol = :standard
_gamma_grow(::AbstractConfig) = 2.0f0
_gamma_shrink(::AbstractConfig) = .75f0
_gamma_shrink_much(::AbstractConfig) = .51f0

_combine_models_by_type(::AbstractConfig) :: Bool = true
=#

#=
filter_type( :: AbstractConfig ) = MaxFilter
filter_shift( :: AbstractConfig ) = Float32(1e-4)

filter_kappa_psi( :: AbstractConfig ) = Float32(1e-4)
filter_psi( :: AbstractConfig ) = 1

filter_kappa_delta(:: AbstractConfig) = 0.7f0
filter_kappa_mu( :: AbstractConfig ) = 100
filter_mu( :: AbstractConfig ) = 0.01f0
=#

#=
var_scaler( :: AbstractConfig ) :: Union{AbstractAffineScaler,Symbol} = :default # :none, :auto, :default
untransform_final_database( :: AbstractConfig ) = false
var_scaler_update( :: AbstractConfig ) = :none

save_no_model_meta_data( :: AbstractConfig ) = true
=#

#=================================================================================
AlgorithmConfig
=================================================================================#
@with_kw struct AlgorithmConfig{F <: AbstractFloat }
	T :: Type{F} = Float64

	max_iter :: Int = max_iter( DEFAULT_CONFIG )
	max_evals :: Int = max_evals( DEFAULT_CONFIG )

	delta_0 :: F = T( delta_0(DEFAULT_CONFIG) )
	delta_max :: F = T( delta_max(DEFAULT_CONFIG) )

	var_scaler :: Union{AbstractAffineScaler,Nothing} = var_scaler( DEFAULT_CONFIG )

	@assert delta_0 > 0 "Initial radius `delta_0` must be positive."
	@assert delta_max > 0 "Maximum radius `delta_max` must be positive."
end

for fn in fieldnames(AlgorithmConfig)
    @eval $fn( ac :: AlgorithmConfig ) = getfield( ac, Symbol($fn) )
end