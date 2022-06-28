function initialize_algo_config(
	algo_config;
	kwargs...
)
	if isempty(kwargs...)
		if isnothing(algo_config)
			return DEFAULT_CONFIG
		else
			return algo_config
		end
	else
		T = if haskey( kwargs, :_precision ) 
			getindex(kwargs, :_precision)
		else
			float_types = [ eltype(v) for v in values(kwargs) if v isa AbstractFloat ]
			isempty(float_types) ? MIN_PRECISION : Base.promote_type( float_types... )
		end
		ac_args = if isnothing(algo_config)
			kwargs
		else
			Dict( k => haskey(kwargs, k) ? getindex(kwargs, k) : getfield( algo_config, k ) for k = fieldnames(algo_config) )
		end
		return AlgorithmConfig{T}(; ac_args...)
	end
end

function initialize_var_scaler(mop :: AbstractMOP, ac :: AbstractConfig)
	lb, ub = bounds_vectors( mop )

	user_scaler = var_scaler( ac )
	variables = _variable_indices(mop)
	
	if user_scaler isa AbstractAffineScaler 
		vars_scaler = _variable_indices(user_scaler)
		if isempty(setdiff(vars_mop, vars_scaler))
			if variables == vars_scaler
				return user_scaler
			else
				sorting_index = [ findfirst(isequal(v), var_scaler) for v = variables ]
				return _init( FullScaler;
					variables,
					scaling_matrix = _scaling_matrix( user_scaler )[sorting_index, sorting_index],
					scaling_constants_vector = _scaling_constants_vector( user_scaler )[sorting_index]
				)
			end
		else
			@warn "Cannot use the provided variable scaler because the variable indices differ."
		end
	end

	lb_inf = isinf.(lb)
	ub_inf = isinf.(ub)
	w_inf = lb_inf .|| ub_inf

	if all(w_inf)
		return _init( NoVarScaling; 
			variables,
		)
	else
		w = ub .- lb
		w[w_inf] .= 1
		w_inv = 1 ./ w
		scaling_constants_vector = - lb .* w_inv
		scaling_constants_vector[w_inf] .= 0

		scaling_matrix = LinearAlgebra.Diagonal( w_inv )
		unscaling_matrix = LinearAlgebra.Diagonal( w )
		
		return _init( FullScaler;
			variables,
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

function _initialize_primal_site_fallback(mop, x0 :: Union{AbstractDict, AbstractDictionary})
	x_vec = ensure_precision( collect( getindices(mop, _variable_indices(mop) ) ) )
	return x_vec
end

_initialize_primal_site_fallback(mop, :: Nothing) = error("`mop` has no primal values and no initial site was provided.")

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

	num_objectives(mop) == 0 &&	error("`mop` has no objectives.")
	num_vars(mop) > 0 || error("There are no variables in `mop`.")
	n_vars = num_vars(mop)
	@assert length(x0) == n_vars "Number of variables $(n_vars) does not match length of `x0` ($(length(x0)))."

	@warn("TODO: Reset number of evaluations in `mop`.")

	new_algo_config = initialize_algo_config( algo_config; kwargs... )
	scal = initialize_var_scaler(mop, new_algo_config)
	
	x_vec = initialize_primal_site(mop, x0)
	x_dict = site_vec_to_dict( x_vec )

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
end


end