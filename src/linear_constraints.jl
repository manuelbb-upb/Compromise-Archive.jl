
# Helpers for bound constraints and linear constraints expressed 
# by arrays:

"""
    _intersect_bound_vec(x, b, dir; sense = :lb	)

Return the array `σ` of scalars such that it holds
for all indices `i` that `σ[i]` is the value for which
`b == x[i] + σ[i] * dir[i]` if `sense == :lb` or 
`x[i] + σ[i] * dir[i] == b` if `sense == :ub` 
(except for the case `x[i] == b[i]`).

!!! note
    This function is used in `_intersect_bounds` to determine 
	a *maximum* allowed stepsize, which results in the following 
	pecularities:
	    1) If `dir[i] == 0` then `σ[i] == typemax(eltype(x))`.
		2) If `x[i]` lies on the respective boundary, and `d[i]` 
		   points "away" from the boundary, then `σ[i] == typemax(eltype(x))`, 
		   else `σ[i] = 0`.
"""
function _intersect_bound_vec( 
    x :: AbstractVector{X}, 
	b :: AbstractVector{B}, 
	dir :: AbstractVector{D}, 
	dir_nz = .!(iszero.(dir));
    sense = :lb
) where {X,B,D}
    
	F = Base.promote_type(X,B,D)

    finf = typemax(F)
	σ = fill( finf, length(x) )

    isempty( b ) && return finf

    _d = dir[dir_nz]
	_σ = view(σ, dir_nz)

	# first: treat rows where we can move
	# in a positive direction along `dir`
    tmp = b[dir_nz] .- x[dir_nz]
    tmp_z = iszero.(tmp)
    tmp_nz = .!(tmp_z)
    _σ[tmp_nz] .= tmp[tmp_nz] ./ _d[tmp_nz]
    
	__d = _d[tmp_z]
    isempty( __d ) && return σ
	__σ = view( _σ, tmp_z )

	# second: if there are rows where 
	# `x` touches the boundary we can move
	# infintely (if direction points to interior) 
	# or not at all
    fzero = F(0)
	zero_ind = sense == :lb ? __d .< 0 : __d .> 0
    __σ[ zero_ind ] .= fzero 

	return σ
end

"""
	_stepsizes_interval( x, d;
		lb = [], ub = [],
		A_eq = Matrix(undef,0,0), b_eq = [],
		A_ineq = Matrix(undef,0,0), b_ineq = [],
	)

Return a tuple of stepsizes `(σ_min, σ_max)` such that 
`x .+ σ * d` conforms to the linear constraints 
* `lb .<= x .+ σ * d .<= ub`,
* `A_ineq * x .+ b_ineq .<= 0` and 
* `A_eq` * x .+ b_eq .== 0`.
for all `σ` in the closed intervall `[σ_min, σ_max]`.
If this is not possible, return `nothing`.

Else, `σ_min` is maximum negative stepsize possible and 
`σ_max` is the minimum positive stepsize.
"""
function _stepsizes_interval( 
	x :: AbstractVector{X}, 
	d :: AbstractVector{D}; 
	lb :: AbstractVector{LB} = X[], 
	ub :: AbstractVector{UB} = X[],
	A_eq :: AbstractMatrix{AEQ} = Matrix{X}(undef,0,length(x)),
	b_eq :: AbstractVector{BEQ} = X[], 
	A_ineq :: AbstractMatrix{AINEQ} = Matrix{X}(undef,0,length(x)),
	b_ineq :: AbstractVector{BINEQ} = X[],
) where {X,D,LB,UB,AEQ,BEQ,AINEQ,BINEQ}

    # TODO can we pass precalculated `Ax` values for `A_eq` and `A_ineq`
    n_vars = length(x)

    @assert length(lb) == n_vars "Dimension mismatch in `lb`."
    @assert length(ub) == n_vars "Dimension mismatch in `ub`."
    @assert size(A_eq, 2) == n_vars "Dimension mismatch in `A_eq`."
    @assert size(A_ineq, 2) == n_vars "Dimension mismatch in `A_ineq`."
    @assert size(A_eq, 1) == length(b_eq) "Dimension mismatch in `A_eq` and `b_eq`."
    @assert size(A_ineq, 1) == length(b_ineq) "Dimension mismatch in `A_ineq` and `b_ineq`."

    T = Base.promote_type(MIN_PRECISION,X,D,LB,UB,AEQ,BEQ,AINEQ,BINEQ)
    unsuc_ret_val = ( T(NaN), T(NaN) )

    if iszero(d)
        return (typemin(T), typemax(T)) :: Tuple{T,T}
    end

	if isempty( A_eq )
		# only inequality constraints
	
		d_zero_index = iszero.(d)
		d_nz = .!( d_zero_index )

		# lb <= x + σd - ε  ⇒  σ_i = (lb[i] - x[i] + ε) / d[i]
		σ_lb = _intersect_bound_vec( x, lb, d, d_nz; sense = :lb)

		# x + σ d + ε <= ub  ⇒  σ_i = (ub[i] - x[i] - ε) / d[i]
		σ_ub = _intersect_bound_vec( x, ub, d, d_nz; sense = :ub)

		# linear inequality constraint intersection
        σ_ineq = if isempty(A_ineq)
            T[] 
        else
			# A(x + σd) + b <= 0  ⇔  Ax + σ Ad <= - b
            ineq_bound = isempty( b_ineq ) ? zeros( T, n_vars ) : -b_ineq
		    _intersect_bound_vec( A_ineq*x, ineq_bound, A_ineq*d; sense = :ub ) 
        end

		σ = T[ σ_lb; σ_ub; σ_ineq ]
		# every entry in σ describes an allowed intervall of stepsizes
		# if `σ[i] >= 0`, then we can move along `d[i]` within `[0, σ[i]]`
		# if `σ[i] < 0`, then we can move along `d[i]` within `[σ[i], 0]`.
		
		σ_non_neg_index = σ .>= 0
		σ_neg_index = .!(σ_non_neg_index) #σ .< 0

		σ₊_array = σ[ σ_non_neg_index ]
		σ₋_array = σ[ σ_neg_index ]
		
		σ_pos = isempty( σ₊_array ) ? T(0) : minimum( σ₊_array )		
		σ_neg =	isempty( σ₋_array ) ? T(0) : maximum( σ₋_array )

		return (σ_neg, σ_pos) :: Tuple{T, T}
	
	else
		# there are equality constraints
		# they have to be all fullfilled and we loop through them one by one (rows of A_eq)
		num_eq = size(A_eq, 1)
		_b = isempty(b_eq) ? zeros(T, num_eq) : b_eq

		σ = T(Inf)
		for i = 1 : N
			a = A_eq[i, :]

            # a'(x + σd) + b = 0  ⇔  σ a'd = b - a'x  ⇔ σ = ( b - a'x )/a'd 
            ad = a'd
			if !iszero(ad) 
				σ_i = (_b[i] - a'x) / ad
			else
                # check for primal feasibility of `x`:
				if !(iszero(a'x + _b[i]))
					return unsuc_ret_val
				else
					continue
				end
			end
			
			if isinf(σ)
				σ = σ_i
			else
				if !(σ_i ≈ σ)
					return unsuc_ret_val
				end
			end
		end
		
		if isinf(σ)
			# only way this could happen:
			# ad == 0 for all i && x feasible w.r.t. eq const
            return _stepsizes_interval( 
				x, d; 
				lb, ub, A_ineq, b_ineq,
                A_eq = Matrix{T}(undef, 0, n_vars)
			) :: Tuple{T,T}
		end
			
		# check if x + σd is compatible with the other constraints
		x_trial = x + σ * d
		_b_ineq = isempty(b_ineq) ? zeros(T, n) : b_ineq
		
		lb_incompat = !isempty(lb) && any(x_trial .< lb )
		ub_incompat = !isempty(ub) && any(x_trial .> ub )
		ineq_incompat = !isempty(A_ineq) && any( A_ineq * x_trial .+ _b_ineq .> 0 )
		if lb_incompat || ub_incompat || ineq_incompat			
			return unsuc_ret_val
		else
			return (σ, σ)
		end					
	end
end

function _intersect_bounds(
	x, d;
	impossible_val :: Union{Real, Nothing} = 0,
	ret_mode = :pos, kwargs...
)
	s_interval = _stepsizes_interval( x, d; kwargs... )

	all(isnan.( s_interval )) && return impossible_val

	σ_neg, σ_pos = s_interval
	
	if ret_mode == :pos
		if σ_pos < 0 
			return impossible_val
		else
			return σ_pos 
		end
	elseif ret_mode == :neg
		if σ_neg > 0
			return impossible_val
		else
			return σ_neg
		end
	elseif ret_mode == :absmax
		if abs( σ_pos ) >= abs( σ_neg )
			return σ_pos
		else
			return σ_neg
		end
	elseif ret_mode == :both 
		if σ_pos < 0 || σ_neg > 0
			return impossible_val
		else
			return (σ_neg, σ_pos)
		end
	end
end

# TODO manually optimize this easy function
function intersect_box( x_scaled, d_scaled, lb_scaled, ub_scaled; return_vals = :absmax )
    return _intersect_bounds( x_scaled, d_scaled;
	 	lb = lb_scaled, 
		ub = ub_scaled,
		ret_mode = return_vals
	)
end

#=
# this should be safer than my handwritten function but also slower:
function _intersect_bounds_jump( x :: AbstractVector{R}, d, lb = [], ub = [], 
        A_eq = [], b_eq = [], A_ineq = [], b_ineq = []; 
        ret_mode = :pos, impossible_val = 0, _eps = -1.0 ) where R <: Real

    _lb = !isempty(lb) 
    _ub = !isempty(ub)
    _eq = !(isempty(A_eq) || isempty(b_eq))  # TODO warn if only one is supplied
    _ineq = !(isempty(A_ineq) || isempty(b_ineq))  # TODO warn if only one is supplied

    if !(_lb || _ub || _eq || _empty_ineq)
        ret_mode == :neg && return -MIN_PRECISION(Inf)
        return MIN_PRECISION(INF)
    end

    lp = JuMP.Model( LP_OPTIMIZER )

    #JuMP.set_optimizer_attribute( lp, "polish", true );
    JuMP.set_silent(lp)

    if ret_mode == :pos 
        JuMP.@variable(lp, σ >= 0)
        JuMP.@objective(lp, Max, σ )
    elseif ret_mode == :neg
        JuMP.@variable(lp, σ <= 0)
        JuMP.@objective(lp, Min, σ )
    else
        JuMP.@variable(lp, σ)
        JuMP.@variable(lp, abs_σ)
        JuMP.@constraint(lp, -abs_σ <= σ  <= abs_σ)
        JuMP.@objective(lp, Max, abs_σ)
    end
    
    _lb && JuMP.@constraint(lp, lb .<= x .+ σ * d)
    _ub && JuMP.@constraint(lp, x .+ σ * d .<= ub)
    _eq && JuMP.@constraint(lp, A_eq*(x .+ σ * d) .== b_eq )
    _ineq && JuMP.@constraint(lp, A_ineq*(x .+ σ * d) .== b_ineq )

    JuMP.optimize!( lp )
    
    σ_opt = JuMP.value(σ)
    if isnan(σ_opt) σ_opt = impossible_val end 
    
    return σ_opt
end
=#
