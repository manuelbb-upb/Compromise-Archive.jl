const Vec = AbstractVector{<:Real}
const VecVec = AbstractVector{<:AbstractVector}
const NumOrVec = Union{Real, Vec}
const VecOrNum = NumOrVec 
const Mat = AbstractMatrix{<:Real}

const VecF = AbstractVector{<:AbstractFloat}
const MatF = AbstractMatrix{<:AbstractFloat}
const NumOrVecF = Union{AbstractFloat, VecF}

const MIN_PRECISION = Float32
const INF = MIN_PRECISION(Inf)

# ### Enums

# These codes should be availabe everywhere and that is why we 
# define them here:
#=
@enum ITER_TYPE begin
    ACCEPTABLE     # accept trial point, shrink radius 
    SUCCESSFULL    # accept trial point, grow radius 
    MODELIMPROVING # reject trial point, keep radius 
    INACCEPTABLE   # reject trial point, shrink radius (much)
    RESTORATION    # apart from the above distinction: a restoration step has been computed and used as the next iterate
    FILTER_FAIL    # trial point is not acceptable for filter
    FILTER_ADD     # trial point acceptable to filter with large constraint violation
    EARLY_EXIT
#    CRIT_LOOP_EXIT
    INITIALIZATION
end

@enum STOP_CODE begin
    CONTINUE = 1
    MAX_ITER = 2
    BUDGET_EXHAUSTED = 3
    CRITICAL = 4
    TOLERANCE = 5 
    INFEASIBLE = 6
    STOP_VAL = 7
end

@enum RADIUS_UPDATE begin 
    LEAVE_UNCHANGED 
    GROW
    SHRINK
    SHRINK_MUCH 
end

const NLOPT_SUCCESS_CODES = [
    :SUCCESS, 
    :STOPVAL_REACHED, 
    :FTOL_REACHED,
    :XTOL_REACHED,
    :MAXEVAL_REACHED, 
    :MAXTIME_REACHED
]

const NLOPT_FAIL_CODES = [
    :FAILURE,
    :INVALID_ARGS,
    :OUT_OF_MEMORY,
    :ROUNDOFF_LIMITED,
    :FORCED_STOP
]
=#

abstract type AbstractOuterEvaluator{is_atomic} end
abstract type DiffFn end
abstract type AbstractInnerEvaluator end
abstract type AbstractSurrogateConfig end 
abstract type AbstractAffineScaler end
abstract type AbstractMOP{is_editable} end

abstract type AbstractConfig end

abstract type AbstractDescentConfig end