### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 2658f446-1d54-4140-9737-e5a22e2d3d8d
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
	using ModelingToolkit
	using ModelingToolkitStandardLibrary.Thermal, ModelingToolkitStandardLibrary.Blocks
	using ModelingToolkit: t_nounits as t, D_nounits as D
	using ModelingToolkitNeuralNets, Lux, StableRNGs
	using OrdinaryDiffEqTsit5
	using Plots, PlutoUI
	using SymbolicIndexingInterface
	using Optimization
	using OptimizationOptimJL
	using LineSearches
	using Statistics
	using SciMLSensitivity
	using SymbolicRegression
	import Zygote
end

# ╔═╡ 48e34a90-67cc-11f0-361f-d5caa033960d
md"""
# Exploring acasual model augmentation with neural networks

Sebastian Micluța-Câmpeanu¹ ², Fredrik Bagge Carlson¹


¹ JuliaHub
² University of Bucharest
"""

# ╔═╡ ef97932f-bbb6-4cc0-8102-5bdbd454b6a7
md"""
## Acasual modeling
"""

# ╔═╡ 3ac83699-240e-4dab-8493-7ac8b294a87d
md"""
With acasual modeling we want to represent the dynamics of the system as a collection of **components** and **connections** between them instead of a (large) system of differential algebraic equations (DAEs). The large system of DAEs *is still built* by ModelingToolkit as part of structural simplification (`mtkcompile`).

The unsimplified version of the system can be visualized as a block diagram, which becomes essential for understandig and debugging more complex system since the connections between the blocks are usually translated to various interactions between the components of the physical system.
"""

# ╔═╡ b1aaae5f-18fa-4ea5-bf86-c6dffd64bcc0
input_f(t) = (1+sin(0.005 * t^2))/2

# ╔═╡ 95c9ea4f-1972-4419-990f-a9e381d37989
md"""
![mtk connectors](https://docs.sciml.ai/ModelingToolkitStandardLibrary/stable/connectors/through_across.png)
"""

# ╔═╡ 18e43649-05a7-4405-89ed-0a58127230d6
md"""
## An example

`ModelingToolkitStandardLibrary` provies predefined (block) components for various domains, making it easier to build up complex systems by just connecting components.
"""

# ╔═╡ 0e8cd59c-39bd-4603-9625-42f205d448b0
@mtkmodel PotWithPlate begin
    @parameters begin
        C1 = 1
        C2 = 15
    end
    @components begin
        input = Blocks.TimeVaryingFunction(f=input_f)
        source = PrescribedHeatFlow(T_ref=373.15)
        plate = HeatCapacitor(C=C1, T=273.15)
        pot = HeatCapacitor(C=C2, T=273.15)
        conduction = ThermalConductor(G=1)
        air = ThermalConductor(G=0.1)
        env = FixedTemperature(T=293.15)
        Tsensor = TemperatureSensor()
    end
    @equations begin
        connect(input.output, :u, source.Q_flow)
        connect(source.port, plate.port)
        connect(plate.port, conduction.port_a)
        connect(conduction.port_b, pot.port)
        connect(pot.port, air.port_a)
        connect(air.port_b, env.port)
        connect(pot.port, Tsensor.port)
    end
end

# ╔═╡ e95b7a6a-4eac-4c14-8853-e14c688cd48b
md"""
## The Block Diagram
"""

# ╔═╡ ba0babf8-7589-48ed-aec0-de19af1c5c76
PlutoUI.LocalResource("pot_plate.png")

# ╔═╡ a9ff7aec-f824-4f74-9d2c-49f083f014f0
md"""
## The equations of the simplified system
"""

# ╔═╡ 8d596d7a-cafa-4eeb-ba2c-0392834efca9
md"""
All that is actually described by 2 ODEs
"""

# ╔═╡ 10f685c6-2717-4ff1-954b-99c02321cc4c
@mtkcompile sys1 = PotWithPlate()

# ╔═╡ 084a9648-1828-4d0f-b3e6-d4bef715d1cc
md"""
Which expands to
"""

# ╔═╡ ca3dfa42-62b4-4203-8d45-aff7fb5e3d67
full_equations(sys1)[1]

# ╔═╡ 4199655f-8326-497d-b7f9-c35c3f8fdc9c
full_equations(sys1)[2]

# ╔═╡ 55ad8c0a-ce04-424c-ab2d-7028cfbe883b
md"""
## Model augmentation

How do we usually represent model modifications in UDEs?

```math
\begin{align}
\frac{dx}{dt} &= ax + NN(x,y) \\
\frac{dy}{dt} &= -cy + NN(x,y)
\end{align}
```
"""

# ╔═╡ dc506208-65be-4d7e-bd56-6740389bcd80
md"""
As we can see, this represents the modification from the neural network at the system level, i.e. after structural simplification.

This no longer can be mapped back in terms of components, so we would loose the ability to reason about the system using the block diagram.

But what if we *could* represent the neural network in the block diagram as a first class citizen?
"""

# ╔═╡ 0053b7a7-b6e4-42c5-b5c4-0762bcc162d3
md"""
## Enter ModelingToolkitNeuralNets
"""

# ╔═╡ ea1005b5-ecfa-4993-b2e2-45e727d84488
md"""
#### What is `ModelingToolkitNeuralNets`?

A package that allows one to embed neural networks inside [`ModelingToolkit`](https://github.com/SciML/ModelingToolkit.jl) systems in order to formulate Universal Differential Equations.

The neural network is symbolically represented either as a block component or a callable parameter, so it can be added to any part of the equations in an `System`. In this talk we will mainly focus on the block component interface, `NeuralNetworkBlock`. What does this block component look like?
"""

# ╔═╡ c3e67ff5-4528-4bd6-9397-17b61bf8d456
md"""
```julia
function NeuralNetworkBlock(; n_input = 1, n_output = 1,
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        eltype = Float64,
        name)
    ca = ComponentArray{eltype}(init_params)

    @parameters p[1:length(ca)]=Vector(ca) [tunable = true]
    @parameters T::typeof(typeof(ca))=typeof(ca) [tunable = false]
    @parameters lux_model::typeof(chain)=chain [tunable = false]

    @variables inputs(t_nounits)[1:n_input] [input = true]
    @variables outputs(t_nounits)[1:n_output] [output = true]

    expected_outsz = only(outputsize(chain, inputs, rng))
    msg = "The outputsize of the given Lux network ($expected_outsz) does not match `n_output = $n_output`"
    @assert n_output==expected_outsz msg

    eqs = [outputs ~ stateless_apply(lux_model, inputs, lazyconvert(T, p))]

    ude_comp = System(
        eqs, t_nounits, [inputs, outputs], [lux_model, p, T]; name)
    return ude_comp
end
```
"""

# ╔═╡ 6e8ba831-9568-47af-90f8-d3303a28accc
md"""
## How can we use this?
"""

# ╔═╡ bf4790f3-c569-4d60-8d5a-aa64a424e636
md"""
The `NeuralNetworkBlock` can be viewed as just a "black box" with a given number of inputs and outputs, meaning that we can create connections to any part of the system from the inputs and the outputs... Almost any connections, since we still need to create a balanced system that can be structurally simplified.

This already shows that we have significant differences between how we operate with system level UDEs and component level UDEs.

- For system level UDEs, we can just modify the differential equations that will be solved, but we lose structural information
- For component based UDEs, the advantage is that we can leverage all the structural information, but the difficulty lies in formulating the UDE in a manner that is structurally and physically consistent.

The component UDE formulation also makes it easier to add new state dimensions to the system and help understand additional degrees of freedom. We can also increase the state dimension via system level UDEs too, but it's much harder to constrain how does the new state dimension interact with the rest of the system.
"""

# ╔═╡ f6748e68-b456-455a-8c0a-5a95920a8135
md"""
## An example
"""

# ╔═╡ d48f2b2c-9d1a-47b6-9412-6e3f517ed017
md"""
Let us consider the following thermal system
"""

# ╔═╡ dbb8d874-5f8b-4529-97a1-96e8edad05ce
@mtkmodel SimplePot begin
    @parameters begin
        C2 = 15
    end
    @components begin
        input = Blocks.TimeVaryingFunction(f=input_f)
        source = PrescribedHeatFlow(T_ref=373.15)
        pot = HeatCapacitor(C=C2, T=273.15)
        air = ThermalConductor(G=0.1)
        env = FixedTemperature(T=293.15)
        Tsensor = TemperatureSensor()
    end
    @equations begin
        connect(input.output, :u, source.Q_flow)
        connect(source.port, pot.port)
        connect(pot.port, Tsensor.port)
        connect(pot.port, air.port_a)
        connect(air.port_b, env.port)
    end
end

# ╔═╡ 4f4806e5-83e6-4927-8a2d-f79a5867d057
md"""
## Building the system
"""

# ╔═╡ 5bcb371c-f967-4b95-918e-a5e825743ca6
@named incomplete_model = SimplePot()

# ╔═╡ 72864a55-d8be-4788-843b-1cd9dc874f10
md"""
## Diagram
"""

# ╔═╡ 886f57b6-9aad-47e5-8177-5c0236b01514
PlutoUI.LocalResource("simple.png")

# ╔═╡ cc6581ae-8408-4353-9e90-9c6426b91308
md"""
## Structural simplification
"""

# ╔═╡ dbc609cb-9b9e-4b5b-8644-24f032326fa2
md"""
Let's compile the system
"""

# ╔═╡ 53cf73c7-2306-42c9-b0e8-5f292bc09869
sys2 = mtkcompile(incomplete_model)

# ╔═╡ 36827985-34fc-4ede-920e-f32c9d594b79
md"""
We can observe that this entire system is governed by a single equation:
"""

# ╔═╡ d53fd3be-e212-40c7-9e65-70ba52605b14
only(full_equations(sys2))

# ╔═╡ dcc1b7f8-ff8d-4ab1-a8b4-bb71123c74f7
md"""
## Comparing with the original system
"""

# ╔═╡ 2651bd18-d5b2-479e-bef1-0693b089b48e
md"""
Comparing the predictions from the above system with the prediction from the system from the beginning at the presentation, we notice a difference. Can we recover the structural change that is needed?
"""

# ╔═╡ 4e345a1c-3f00-4d9e-bb4b-022e2318c30c
begin
	prob1 = ODEProblem(sys1, Pair[], (0, 100.0))
	sol1 = solve(prob1, Tsit5(), reltol=1e-6)
	prob2 = ODEProblem(sys2, Pair[], (0, 100.0))
	sol2 = solve(prob2, Tsit5(), reltol=1e-6)
	plot(sol1, idxs=sys1.pot.T, label="pot.T in original system")
	plot!(sol2, idxs=sys1.pot.T, label="pot.T in simplified system")
end

# ╔═╡ 67e265b0-1729-45f6-8280-b86194fcdc02
md"""
## Adding the neural network block
"""

# ╔═╡ c44af106-a51e-4ce3-a173-1295c58c270c
md"""
We would like to add the neural network to the system roughly as follows
"""

# ╔═╡ a52d9a1f-0789-483f-8253-bccb0389cf08
PlutoUI.LocalResource("nn.png")

# ╔═╡ f6032e9f-35b0-4931-83f9-165e631086db
md"""
## The implementation - specialized component
"""

# ╔═╡ ca375671-4673-4b4b-b1aa-91e233cb1326
@mtkmodel ThermalNN begin
    begin
        n_input = 2
        n_output = 1
        chain = multi_layer_feed_forward(; n_input, n_output, depth=1, width=4, activation=Lux.swish)
    end
    @components begin
        port_a = HeatPort()
        port_b = HeatPort()
        nn = NeuralNetworkBlock(; n_input, n_output, chain, rng=StableRNG(1337))
    end
    @parameters begin
        T0 = 273.15
        T_range = 10
        C1 = 1
    end
    @variables begin
        dT(t), [guess = 0.0]
        Q_flow(t), [guess = 0.0]
        x(t) = T0
    end
    @equations begin
        dT ~ port_a.T - port_b.T
        port_a.Q_flow ~ Q_flow
        C1*D(x) ~ Q_flow - nn.outputs[1]
        port_a.T ~ x
        nn.outputs[1] + port_b.Q_flow ~ 0
        nn.inputs[1] ~ (x - T0) / T_range
        nn.inputs[2] ~ (port_b.T - T0) / T_range
    end
end

# ╔═╡ aff00e05-6816-4f44-94b5-765ae07f3e47
md"""
## Let's take a closer look
"""

# ╔═╡ 764aff04-e1ce-4a08-876b-766197e18584
md"""
```julia
begin
    n_input = 2
    n_output = 1
    chain = multi_layer_feed_forward(; n_input, n_output, depth=1, width=4, activation=Lux.swish)
end
```
"""

# ╔═╡ 7c7c887c-e27b-4a18-b2ca-f4ab9fc62093
md"""
We need to provdide the number of inputs an outputs to our neural network and of course the neural network itself, which can be any Lux.jl network.

`ModelingToolkitNeuralNets` offers an utility function for building simple chains:
```julia
multi_layer_feed_forward(; n_input, n_output, width::Int = 4, depth::Int = 1,
    activation = tanh, use_bias = true, initial_scaling_factor = 1e-8)
```

The `Chain` is built such that the last layer is:
```julia
Lux.Dense(width, n_output;
	init_weight = (rng, a...) -> initial_scaling_factor * Lux.kaiming_uniform(rng, a...), use_bias)
```

This means that by construction our neural network will will have the output scaled by `initial_scaling_factor`.
"""

# ╔═╡ bd3b6bcc-86db-40a4-a75f-64291917beb0
md"""
## Components

These essentially represent "the interface" for our special Thermal neural network component (`ThermalNN`).

```julia
@components begin
	port_a = HeatPort()
    port_b = HeatPort()
    nn = NeuralNetworkBlock(; n_input, n_output, chain, rng=StableRNG(1337))
end
@parameters begin
    T0 = 273.15
    T_range = 10
    C1 = 1
end
@variables begin
    dT(t), [guess = 0.0]
    Q_flow(t), [guess = 0.0]
    x(t) = T0
end
```
"""

# ╔═╡ 9b5f45f4-6b26-404a-b9ef-b3beb2d0a777
md"""
## How does this component work?

```julia
@equations begin
    dT ~ port_a.T - port_b.T
    port_a.Q_flow ~ Q_flow
    C1*D(x) ~ Q_flow - nn.outputs[1]
    port_a.T ~ x
	nn.outputs[1] + port_b.Q_flow ~ 0
	nn.inputs[1] ~ (x - T0) / T_range
	nn.inputs[2] ~ (port_b.T - T0) / T_range
end
```

### How would we write such components?

We would like this component to represent physically meaningful relations, not just to fit the data, so we need to incorporate physical knoweledge into how the neural network interacts with the system.

We want to add a new state for the system, which means that we will have to integrate the output from the neural network. We will denote the new state with `x`.
```julia
nn.inputs[1] ~ some_function(x)
nn.outputs[1] + port_b.Q_flow ~ 0
D(x) ~ some_other_function(nn.outputs[1])
```
"""

# ╔═╡ 0ca9c63b-415d-4059-91e4-70a8bc949545
md"""
## How does this component work?

```julia
@equations begin
    dT ~ port_a.T - port_b.T
    port_a.Q_flow ~ Q_flow
    C1*D(x) ~ Q_flow - nn.outputs[1]
    port_a.T ~ x
	nn.outputs[1] + port_b.Q_flow ~ 0
	nn.inputs[1] ~ (x - T0) / T_range
	nn.inputs[2] ~ (port_b.T - T0) / T_range
end
```

### How would we write such components?

In this particular case, one reasonable assumption to make would be that our component is conservative at steady state. What does this mean?

```julia
D(x) == 0 => port_a.Q_flow + port_b.Q_flow == 0
```
"""

# ╔═╡ 2e386eca-8efe-44dc-92f8-92160d404510
md"""
## How does this component work?

```julia
@equations begin
    dT ~ port_a.T - port_b.T
    port_a.Q_flow ~ Q_flow
    C1*D(x) ~ Q_flow - nn.outputs[1]
    port_a.T ~ x
	nn.outputs[1] + port_b.Q_flow ~ 0
	nn.inputs[1] ~ (x - T0) / T_range
	nn.inputs[2] ~ (port_b.T - T0) / T_range
end
```

### How would we write such components?

The input to the neural network should be normalized, so we can do something like
```julia
nn.inputs[1] ~ (x - T0) / T_range
```

where `T0` and `T_range` are user defined parameteres.
"""

# ╔═╡ 5c7919a0-4338-455e-8b32-2ce5e65e502b
md"""
## The actual implementation - putting it all together
"""

# ╔═╡ 49731e72-f9ff-4b26-8cc4-6059c1ca4c79
@mtkmodel NeuralPot begin
    @parameters begin
        C2 = 15
    end
    @components begin
        input = Blocks.TimeVaryingFunction(f=input_f)
        source = PrescribedHeatFlow(T_ref=373.15)
        pot = HeatCapacitor(C=C2, T=273.15)
        air = ThermalConductor(G=0.1)
        env = FixedTemperature(T=293.15)
        Tsensor = TemperatureSensor()
        thermal_nn = ThermalNN()
    end
    @equations begin
        connect(input.output, :u, source.Q_flow)
        connect(pot.port, Tsensor.port)
        connect(pot.port, air.port_a)
        connect(air.port_b, env.port)
        connect(source.port, thermal_nn.port_a)
        connect(thermal_nn.port_b, pot.port)
    end
end

# ╔═╡ 075c7787-983c-49fa-ad97-ed84c11759a8
md"""
## Building the full system
"""

# ╔═╡ 34e36ab8-b96d-4752-8bed-7960066a9c4c
@named model = NeuralPot()

# ╔═╡ f78cff87-01d4-4fcd-9901-245cc7fb5e90
md"""
## Structural Simplification
"""

# ╔═╡ ae2afb40-af2c-4755-8778-19505d945fa2
sys3 = mtkcompile(model)

# ╔═╡ 851e0e18-b7e4-48af-9d93-277796819038
md"""
Let's check that we can succesfully simulate the system in the initial state:
"""

# ╔═╡ 56db609d-72cd-4fbb-a7ba-ce90115daa5d
begin
	prob3 = ODEProblem(sys3, Pair[], (0, 100.0))
	sol3 = solve(prob3, Tsit5(), abstol=1e-6, reltol=1e-6)
	SciMLBase.successful_retcode(sol3)
end

# ╔═╡ 76e56ead-5644-45dc-86a5-57dd883be4bd
md"""
## Training the neural network - defining the loss
"""

# ╔═╡ f905fbd7-0ee0-4f63-a8f0-a2670d75553b
begin
	tp = Symbolics.scalarize(sys3.thermal_nn.nn.p)
	oop_update = setsym_oop(prob3, tp)
	x0 = prob3.ps[tp]
end

# ╔═╡ aec542ec-5a95-4eb6-afbf-f2a2d6299620
print_cb = (opt_state, loss) -> begin
    opt_state.iter % 1000 ≠ 0 && return false
    @info "step $(opt_state.iter), loss: $loss"
    false
end

# ╔═╡ 46c8a34c-aa91-4e45-af40-b0b96a815a47
function loss(x, opt_ps)
    prob, oop_update, data, ts, get_T = opt_ps

    u0, p = oop_update(prob, x)
    new_prob = remake(prob; u0, p)

    new_sol = solve(new_prob, Tsit5(), saveat=ts, abstol=1e-8, reltol=1e-8, 				verbose=false, sensealg=GaussAdjoint())

    !SciMLBase.successful_retcode(new_sol) && return Inf

    mean(abs2.(get_T(new_sol) .- data))
end

# ╔═╡ 4b5ad787-0838-46ba-b2ee-35c5736355dd
md"""
## Training the neural network - defining the `OptimizationProblem`
"""

# ╔═╡ 0fa6519d-e26a-4795-a097-01c4e43c3fc7
begin	
	of = OptimizationFunction(loss, AutoForwardDiff())
	
	data = sol1[sys1.pot.T]
	get_T = getsym(prob3, sys3.pot.T)
	opt_ps = (prob3, oop_update, data, sol1.t, get_T);
	
	op = OptimizationProblem(of, x0, opt_ps,)
end

# ╔═╡ a6d1a31c-2626-4e74-ac41-849b435948d3
md"""
## Training the neural network - the first stage
"""

# ╔═╡ c84d3c8a-e9e9-494e-bf4a-3d2f7b666c59
res = solve(op, Adam(); maxiters=10_000, callback=print_cb)

# ╔═╡ b430acbd-e2a2-49ad-946d-f294c91825ee
md"""
## Training the neural network - the second stage
"""

# ╔═╡ 63ac9ec2-526b-48f1-aaf7-8ca1833956c9
op2 = OptimizationProblem(of, res.u, opt_ps)

# ╔═╡ 128a33f3-ccd1-4fe2-9f00-cd0ab1dfd701
res2 = solve(op2, LBFGS(linesearch=BackTracking()); maxiters=2000, callback=print_cb)

# ╔═╡ 1418b939-543d-43ad-87b0-8ed92dac3bbc
md"""
## Evaluating the results
"""

# ╔═╡ 60416692-9e20-4c50-8c75-fb0e51a58073
begin
	(new_u0, new_p) = oop_update(prob3, res2.u)
	new_prob1 = remake(prob3, u0=new_u0, p=new_p)
	new_sol1 = solve(new_prob1, Tsit5(), abstol=1e-6, reltol=1e-6)
end

# ╔═╡ 6bdefe43-e703-4323-b636-22d324bda6e6
md"""
## Evaluating the results
"""

# ╔═╡ 36246ec3-c27f-48d4-860c-1b8f8a317289
begin
	plt = plot(new_sol1, layout=(2,3), idxs=[
	    sys3.thermal_nn.nn.inputs[1], sys3.thermal_nn.x,
	    sys3.thermal_nn.nn.outputs[1], sys3.thermal_nn.port_b.T,
	    sys3.pot.T, sys3.pot.port.Q_flow],
	    size=(850,700), lw=3)
	plot!(plt, sol1, idxs=[
	    (sys1.conduction.port_a.T-273.15)/10, sys1.conduction.port_a.T,
	    sys1.conduction.port_a.Q_flow, sys1.conduction.port_b.T,
	    sys1.pot.T, sys1.pot.port.Q_flow], ls=:dash, lw=3)
end

# ╔═╡ 1997fb13-3cff-4229-b43d-6345ecb6d405
md"""
## Evaluating the results

As we can see from the final plot, the neural network fits very well and not only the training data fits, but also the rest of the
predictions of the system match the original system. Let us also compare against the predictions of the incomplete system:
"""

# ╔═╡ 1b82ac97-d145-4f5a-8130-7cf71c1e566d
begin
	plot(sol1, label=["original sys: pot T" "original sys: plate T"], lw=3)
	plot!(sol3; idxs=[sys3.pot.T], label="untrained UDE", lw=2.5)
	plot!(sol2; idxs=[sys2.pot.T], label="incomplete sys: pot T", lw=2.5)
	plot!(new_sol1; idxs=[sys3.pot.T, sys3.thermal_nn.x], label="trained UDE", ls=:dash, lw=2.5)
end

# ╔═╡ ef8eab8e-d236-4f72-a7be-e4acc06364df
md"""
## What's next?
"""

# ╔═╡ fe8a3fbf-bf7d-41a6-b48a-4f211019bff3
begin
	lux_model = new_sol1.ps[sys3.thermal_nn.nn.lux_model]
	nn_p = new_sol1.ps[sys3.thermal_nn.nn.p]
	T = new_sol1.ps[sys3.thermal_nn.nn.T]
	
	sr_input = reduce(hcat, new_sol1[sys3.thermal_nn.nn.inputs])
	sr_output = LuxCore.stateless_apply(lux_model, sr_input, convert(T, nn_p))
end

# ╔═╡ 30b451f5-9889-40ea-a72a-8a94bfb4b1a3
md"""
## Symbolic Regression
"""

# ╔═╡ eb8d3d95-504f-4b81-848d-7746863b55e2
equation_search(sr_input, sr_output)

# ╔═╡ Cell order:
# ╟─48e34a90-67cc-11f0-361f-d5caa033960d
# ╟─2658f446-1d54-4140-9737-e5a22e2d3d8d
# ╟─ef97932f-bbb6-4cc0-8102-5bdbd454b6a7
# ╟─3ac83699-240e-4dab-8493-7ac8b294a87d
# ╟─b1aaae5f-18fa-4ea5-bf86-c6dffd64bcc0
# ╟─95c9ea4f-1972-4419-990f-a9e381d37989
# ╟─18e43649-05a7-4405-89ed-0a58127230d6
# ╠═0e8cd59c-39bd-4603-9625-42f205d448b0
# ╟─e95b7a6a-4eac-4c14-8853-e14c688cd48b
# ╟─ba0babf8-7589-48ed-aec0-de19af1c5c76
# ╟─a9ff7aec-f824-4f74-9d2c-49f083f014f0
# ╟─8d596d7a-cafa-4eeb-ba2c-0392834efca9
# ╠═10f685c6-2717-4ff1-954b-99c02321cc4c
# ╟─084a9648-1828-4d0f-b3e6-d4bef715d1cc
# ╟─ca3dfa42-62b4-4203-8d45-aff7fb5e3d67
# ╟─4199655f-8326-497d-b7f9-c35c3f8fdc9c
# ╟─55ad8c0a-ce04-424c-ab2d-7028cfbe883b
# ╟─dc506208-65be-4d7e-bd56-6740389bcd80
# ╟─0053b7a7-b6e4-42c5-b5c4-0762bcc162d3
# ╟─ea1005b5-ecfa-4993-b2e2-45e727d84488
# ╟─c3e67ff5-4528-4bd6-9397-17b61bf8d456
# ╟─6e8ba831-9568-47af-90f8-d3303a28accc
# ╟─bf4790f3-c569-4d60-8d5a-aa64a424e636
# ╟─f6748e68-b456-455a-8c0a-5a95920a8135
# ╟─d48f2b2c-9d1a-47b6-9412-6e3f517ed017
# ╠═dbb8d874-5f8b-4529-97a1-96e8edad05ce
# ╟─4f4806e5-83e6-4927-8a2d-f79a5867d057
# ╠═5bcb371c-f967-4b95-918e-a5e825743ca6
# ╟─72864a55-d8be-4788-843b-1cd9dc874f10
# ╟─886f57b6-9aad-47e5-8177-5c0236b01514
# ╟─cc6581ae-8408-4353-9e90-9c6426b91308
# ╟─dbc609cb-9b9e-4b5b-8644-24f032326fa2
# ╠═53cf73c7-2306-42c9-b0e8-5f292bc09869
# ╟─36827985-34fc-4ede-920e-f32c9d594b79
# ╠═d53fd3be-e212-40c7-9e65-70ba52605b14
# ╟─dcc1b7f8-ff8d-4ab1-a8b4-bb71123c74f7
# ╟─2651bd18-d5b2-479e-bef1-0693b089b48e
# ╟─4e345a1c-3f00-4d9e-bb4b-022e2318c30c
# ╟─67e265b0-1729-45f6-8280-b86194fcdc02
# ╟─c44af106-a51e-4ce3-a173-1295c58c270c
# ╟─a52d9a1f-0789-483f-8253-bccb0389cf08
# ╟─f6032e9f-35b0-4931-83f9-165e631086db
# ╠═ca375671-4673-4b4b-b1aa-91e233cb1326
# ╟─aff00e05-6816-4f44-94b5-765ae07f3e47
# ╟─764aff04-e1ce-4a08-876b-766197e18584
# ╟─7c7c887c-e27b-4a18-b2ca-f4ab9fc62093
# ╟─bd3b6bcc-86db-40a4-a75f-64291917beb0
# ╟─9b5f45f4-6b26-404a-b9ef-b3beb2d0a777
# ╟─0ca9c63b-415d-4059-91e4-70a8bc949545
# ╟─2e386eca-8efe-44dc-92f8-92160d404510
# ╟─5c7919a0-4338-455e-8b32-2ce5e65e502b
# ╠═49731e72-f9ff-4b26-8cc4-6059c1ca4c79
# ╟─075c7787-983c-49fa-ad97-ed84c11759a8
# ╠═34e36ab8-b96d-4752-8bed-7960066a9c4c
# ╟─f78cff87-01d4-4fcd-9901-245cc7fb5e90
# ╠═ae2afb40-af2c-4755-8778-19505d945fa2
# ╟─851e0e18-b7e4-48af-9d93-277796819038
# ╠═56db609d-72cd-4fbb-a7ba-ce90115daa5d
# ╟─76e56ead-5644-45dc-86a5-57dd883be4bd
# ╠═f905fbd7-0ee0-4f63-a8f0-a2670d75553b
# ╟─aec542ec-5a95-4eb6-afbf-f2a2d6299620
# ╠═46c8a34c-aa91-4e45-af40-b0b96a815a47
# ╟─4b5ad787-0838-46ba-b2ee-35c5736355dd
# ╠═0fa6519d-e26a-4795-a097-01c4e43c3fc7
# ╠═a6d1a31c-2626-4e74-ac41-849b435948d3
# ╠═c84d3c8a-e9e9-494e-bf4a-3d2f7b666c59
# ╟─b430acbd-e2a2-49ad-946d-f294c91825ee
# ╠═63ac9ec2-526b-48f1-aaf7-8ca1833956c9
# ╠═128a33f3-ccd1-4fe2-9f00-cd0ab1dfd701
# ╟─1418b939-543d-43ad-87b0-8ed92dac3bbc
# ╠═60416692-9e20-4c50-8c75-fb0e51a58073
# ╟─6bdefe43-e703-4323-b636-22d324bda6e6
# ╟─36246ec3-c27f-48d4-860c-1b8f8a317289
# ╟─1997fb13-3cff-4229-b43d-6345ecb6d405
# ╟─1b82ac97-d145-4f5a-8130-7cf71c1e566d
# ╟─ef8eab8e-d236-4f72-a7be-e4acc06364df
# ╠═fe8a3fbf-bf7d-41a6-b48a-4f211019bff3
# ╟─30b451f5-9889-40ea-a72a-8a94bfb4b1a3
# ╠═eb8d3d95-504f-4b81-848d-7746863b55e2
