# make a configuration broadcastable
Broadcast.broadcastable( sc::AbstractSurrogateConfig ) = Ref(sc);

# Methods to be implemented by each type inheriting from AbstractSurrogateConfig
max_evals( :: AbstractSurrogateConfig ) ::Int = typemax(Int)

combinable( :: AbstractSurrogateConfig ) :: Bool = false

needs_gradients( :: AbstractSurrogateConfig ) :: Bool = false
needs_hessians( :: AbstractSurrogateConfig ) :: Bool = false
