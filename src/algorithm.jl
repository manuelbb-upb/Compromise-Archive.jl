function initialize_algo_config(
	algo_config;
	kwargs...
)
    if length(kwargs)==0
        if isnothing(algo_config)
            return DEFAULT_CONFIG
        else
            return algo_config
        end
    else
        ac_args = if isnothing(algo_config)
            kwargs
        else
            Dict(
                k => haskey(kwargs, k) ? getindex(kwargs, k) : getfield(algo_config, k)
                  for k = fieldnames(algo_config)
            )
        end
        return AlgorithmConfig(; ac_args...)
    end
end

function initialize_scaler(mop::AbstractMOP, ac::AbstractConfig)
    vars = _variables(mop)

    user_scaler = var_scaler(ac)
    if user_scaler isa VariableTransformation
        unspecified_indices = VariableIndex[]
        for vi in input_indices(user_scaler)
            if !(vi in vars)
                @warn "Index $(vi) not specified in user provided scaler."
                push!(unspecified_indices, vi)
            end
        end
        if isempty(unspecified_indices)
            return user_scaler
        else
            # TODO combined variable scaler
            # for now, returning `nothing` will error
            return nothing
        end
    end

    scaled_vars = make_scaled(vars)
    lb, ub = _bounds_vectors(mop) # variable bound vectors sorted as per `vars`
    w = ub .- lb
    w_inf = isinf.(w)
    if all(w_inf)
        # If all variables are unconstrained, return Identity:
        return AffineScaling(
            ;id = 1,
            input_output_indices=Dictionary(vars, scaled_vars))
    else
        # for constrained variables, use min-max scaling: ξ = (x-lb)/(ub-lb)
        # ξ = W⁻¹ * x - W⁻¹ * lb
        # for unconstrained variables, use mean as scaling factor
        w_finite = .!(w_inf)
        diag = 1 ./ w
        diag[w_inf] .= sum( diag[w_finite] ) / sum(w_finite)
        A = LinearAlgebra.Diagonal(diag)
        b = - A * lb
        b[w_inf] .= 0
        return AffineScaling(
            ;id=1,
            transformer=LinearVectorMapping(A,b),
            input_output_indices=Dictionary(vars, scaled_vars)
        )
    end
end
function _initialize_primal_site_fallback(mop, x0 :: Vector)
	x_vec = ensure_precision(x0)
	return x_vec
end

function _initialize_primal_site_fallback(mop, x0::AbstractDictionary)
	x_vec = ensure_precision( collect( getindices(x0, _variable_indices(mop) ) ) )
	return x_vec
end

function _initialize_primal_site_fallback(mop, :: Nothing)
    return error("`mop` has no primal values and no initial site was provided.")
end

function initialize_primal_site( mop :: AbstractMOP, x0 )
	if _has_primal_variable_values(mop)
		return ensure_precision( _primal_variable_vector(mop) )
	else
		return _initialize_primal_site_fallback(mop, x0)
	end
end

function initialize_data(
    mop::AbstractMOP,
    x0::Union{AbstractDictionary,Vec,Nothing}=nothing;
    algo_config::Union{AbstractConfig,Nothing}=nothing,
    kwargs...
)

    dim_objectives(mop) == 0 && error("`mop` has no objectives.")
    num_vars(mop) > 0 || error("There are no variables in `mop`.")
    n_vars = num_vars(mop)
    @assert length(x0) == n_vars "Number of variables $(n_vars) does not match length of `x0` ($(length(x0)))."

    @warn("TODO: Reset number of evaluations in `mop`.")

    new_algo_config = initialize_algo_config(algo_config; kwargs...)

    x_vec = initialize_primal_site(mop, x0)
    _x_dict = site_vec_to_dict(mop, x_vec)

    if !(check_variable_bounds(mop, _x_dict))
        @warn("Primal site does not conform to variable bounds. Projecting into box.")
        project_into_bounds!(_x_dict, mop)
    end

    # scale variables:
    scaler = initialize_scaler(mop, new_algo_config)
    unscaler = invert(scaler)
    x_dict = Dictionary{SCALAR_INDEX}(
        merge(_x_dict, eval_at_dict(scaler, _x_dict) )
    )

    # initialize iterate
    # `x` will be a Dictionary{SCALAR_INDEX, F} where {F<:Real}
    x = eval_at_dict!(mop, x_dict)
    F = valtype(x)
    # read initial trust region radius from config
    Δ = F(delta_0(new_algo_config))

    evaluators_tuple = outer_evaluators_needing_models(mop, Val(combine_models_by_type(new_algo_config)))
end
