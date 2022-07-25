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

function initialize_var_scaler(mop :: AbstractMOP, ac :: AbstractConfig)
    lb, ub = _bounds_vectors( mop )

	  user_scaler = var_scaler( ac )
	  vars_mop = _variable_indices(mop)

    # If the user wishes for a specific variable scaling
    # by providing `user_scaler` in the config,
    # then try to use it.
	  if user_scaler isa AbstractAffineScaler
		    vars_scaler = _variable_indices(user_scaler)
		    if isempty(setdiff(vars_mop, vars_scaler)) # check if all variables are dealt with
			      if vars_mop == vars_scaler
				        return user_scaler
			      else
                # resort variables to match mop and return new variable scaler
				        sorting_index = [ findfirst(isequal(v), var_scaler) for v = vars_mop ]
				        return _init(
                    FullScaler;
					          variable_indices = vars_mop,
					          scaling_matrix = _scaling_matrix( user_scaler )[sorting_index, sorting_index],
					          scaling_constants_vector = _scaling_constants_vector( user_scaler )[sorting_index]
				        )
			      end
		    else
			      @warn "Cannot use the provided variable scaler because the variable indices differ."
		    end
	  end

    # Determine the default scaling behavior

	  lb_inf = isinf.(lb)
	  ub_inf = isinf.(ub)
	  w_inf = .|(lb_inf, ub_inf)

	  if all(w_inf)
        # if problem is unconstrained, do not scale anything
		    return _init(
            NoVarScaling;
			      variable_indices = vars_mop,
		    )
	  else
        # scale constrained variables to [0,1]
		    w = ub .- lb
		    w[w_inf] .= 1
		    w_inv = 1 ./ w
		    scaling_constants_vector = - lb .* w_inv
		    scaling_constants_vector[w_inf] .= 0

		    scaling_matrix = LinearAlgebra.Diagonal( w_inv )
		    unscaling_matrix = LinearAlgebra.Diagonal( w )

		    return _init(
            FullScaler;
			      variable_indices = vars_mop,
			      scaling_matrix,
			      scaling_constants_vector,
			      unscaling_matrix,
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
	if _has_primal_values(mop)
		return ensure_precision( _primal_vector(mop) )
	else
		return _initialize_primal_site_fallback(mop, x0)
	end
end

function initialize_iterate(;
	unscaled_site :: AbstractVector{F},
	unscaled_dict :: AbstractDictionary{VariableIndex,T},
	variable_scaler
) where {F,T}
	scaled_site = transform( unscaled_site, variable_scaler )
	scaled_dict = transform( unscaled_dict, variable_scaler )

	X = Base.promote_type( F, T, valtype(scaled_dict), eltype(scaled_site) )
	return Iterate(;
		scaled_site = X.(scaled_site),
		unscaled_site = X.(unscaled_site),
		scaled_dict = X.(scaled_dict),
		unscaled_dict = X.(unscaled_dict),
	)
end

function initialize_data( 
	mop :: AbstractMOP, 
	x0 :: Union{AbstractDictionary,Vec,Nothing}=nothing;
	algo_config :: Union{AbstractConfig, Nothing} = nothing,
	kwargs...
)

    dim_objectives(mop) == 0 &&	error("`mop` has no objectives.")
    num_vars(mop) > 0 || error("There are no variables in `mop`.")
    n_vars = num_vars(mop)
    @assert length(x0) == n_vars "Number of variables $(n_vars) does not match length of `x0` ($(length(x0)))."

    @warn("TODO: Reset number of evaluations in `mop`.")

    new_algo_config = initialize_algo_config( algo_config; kwargs... )

    scal = initialize_var_scaler(mop, new_algo_config)

    x_vec = initialize_primal_site(mop, x0)
    x_dict = site_vec_to_dict(mop, x_vec )

    #=
    mop_eval = evaluate_at_unscaled_site( mop, dict )
    F = Base.promote_type( valtype(x_dict), valtype(mop_eval) )

    iterate = initialize_iterate(;
      unscaled_site = F.(x_vec),
      unscaled_dict = F.(x_dict),
      variable_scaler = scal
    )

    T = _precision(iterate)

    res_id = put_into_db # TODO

    iter_data = IterData(;
      iterate,
      evaluation_cache = T.(mop_eval),
      radius = T( delta_0(new_algo_config) ),
    )
    =#
    return x_dict
end
