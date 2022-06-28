# depends on AbstractDifferentiation
# `wuf` stands for `WrappedUserFunc`, but the type is not available here

@with_kw struct FuncContainerBackend{
    G <: Union{AbstractVector{<:Function},Nothing}, 
    J <: Union{Function,Nothing}, 
    H <: Union{AbstractVector{<:Function},Nothing},
    F <: AD.AbstractBackend,
    F2 <: Union{Nothing,AD.HigherOrderBackend}
} 
    gradients :: G = nothing
    jacobian :: J = nothing
    hessians :: H = nothing
    fallback :: F = AD.FiniteDifferencesBackend()
    fallback2 :: F2 = nothing
end

function _gradient( ::Nothing, ::Nothing, fallback, wuf, x, output_number )
    return AD.gradient( 
        fallback, 
        ξ -> eval_at_vec( wuf, ξ, output_number ),
        x
    )[end]
end

function _gradient( :: Nothing, jacobian_handle, fallback, wuf, x, output_number )
    return vec( jacobian_handle(x)[ output_number, :] )
end

function _gradient( gradients_handles, jacobian_handle, fallback, wuf, x, output_number )
    return gradients_handles[output_number]( x )
end

function gradient( fcb :: FuncContainerBackend, wuf , x; output_number )
    return _gradient( fcb.gradients, fcb.jacobian, fcb.fallback, wuf, x, output_number )
end

function _jacobian( :: Nothing, :: Nothing, fallback, wuf, x )
    return AD.jacobian( 
        fallback, 
        ξ -> eval_at_vec( wuf, ξ ),
        x
    )[end]
end

function _jacobian( gradients_handles, :: Nothing, fallback, wuf, x )
    return transpose(
        hcat( 
            ( g(x) for g in gradients_handles )...
        )
    )
end

function _jacobian( gradients_handles, jacobian_handle, fallback, wuf, x)
    return jacobian_handle(x)
end

function jacobian( fcb :: FuncContainerBackend, wuf , x )
    return _jacobian( fcb.gradients, fcb.jacobian, fcb.fallback, wuf, x )
end

function _partial_jacobian( :: Nothing, :: Nothing, fallback, wuf, x, output_numbers )
    return AD.jacobian( 
        fallback, 
        ξ -> eval_at_vec( wuf, ξ, output_numbers ),
        x
    )
end

function _partial_jacobian( gradients_handles, :: Nothing, fallback, wuf, x, output_numbers )
    return transpose(
        hcat( 
            ( g(x) for g in gradients_handles[output_numbers] )...
        )
    )
end

function _partial_jacobian( gradients_handles, jacobian_handle, fallback, wuf, x, output_numbers )
    return jacobian_handle(x)[output_numbers, :]
end

function partial_jacobian( fcb :: FuncContainerBackend, wuf , x; output_numbers )
    return _partial_jacobian( fcb.gradients, fcb.jacobian, fcb.fallback, wuf, x, output_numbers )
end

function _hessian( hessians_handles, fallback, fallback2, wuf, x, output_number  )
    return hessians_handles[ output_number ]( x )
end

function _hessian( ::Nothing, fallback, fallback2, wuf, x, output_number  )
    return AD.hessian( 
        fallback2,
        ξ -> eval_at_vec(wuf, x, output_number),
        x
    )[end]
end

function _hessian( ::Nothing, fallback, ::Nothing, wuf, x, output_number  )
    return AD.jacobian(
        fallback,
        ξ -> gradient(fcb, wuf, ξ; output_number ),
        x
    )[end]
end


function hessian( fcb :: FuncContainerBackend, wuf , x; output_number )
    return _hessian( fcb.hessians, fcb.fallback, fcb.fallback2, wuf, x, output_number)
end
