import gpt as g

from gpt_ad_features import *

# alias for reverse autodiff in gpt
rad = g.ad.reverse

# setup rng
rng = g.random("seed")

# load a gauge field
config_name = "1200.config"
gauge_field_path = "/glurch/scratch/knd35666/ensembles/ens001/" + config_name
U = g.load(gauge_field_path)

# get grid of the field
grid = U[0].grid

# set paths as nearest neighbor hops
paths = [g.path(),] + [g.path().forward(mu) for mu in range(4)] + [g.path().backward(mu) for mu in range(4)]

# create the layers
ptc1 = ptc_layer(grid,
                 1,
                 1,
                 g.object_type.ot_vector_spin_color(4,3),
                 g.object_type.ot_matrix_spin(4),
                 U,
                 paths)

# define the network
def network(network_input):
    return ptc1([network_input,])[0]

# initialize the weights
for weight in ptc1.weights:
    weight.value = rng.normal(weight.value)

# wilson clover operator
fermion_p = {"csw_r": 1.0,
             "csw_t": 1.0,
             "mass": -0.55,
             "boundary_phases": [1,1,1,1],
             "isAnisotropic": False,
             "xi_0": 1.0,
             "nu": 1.0
            }
w = g.qcd.fermion.wilson_clover(U, fermion_p)

# optimization parameters
sample_size = 5
tolerance = 1e-9
iterations = 10000

# function to calculate cost and gradient
def cost_and_grad(weights):
    # set correct weights
    ptc1.weights = weights

    source = rng.normal([g.vspincolor(grid) for _ in range(sample_size)])
    normalizations = [g.norm2(se) for se in source]

    training_outputs = [g(se / norm**0.5) for se, norm in zip(source, normalizations)]
    training_inputs = [rad.node(g(w*oe)) for oe in training_outputs]

    # define cost
    cost = rad.node(0)
    for sample in range(sample_size):
        cost += 1/sample_size * g.norm2(network(training_inputs[sample]) - training_outputs[sample])

    cost /= 8*8*8*16

    cost_val = cost().real

    return cost_val

# define optimizer
opt = adam(ptc1.weights, cost_and_grad, alpha=1e-2)

_, (costs, _, _) = opt.optimize(tol=tolerance, maxiter=iterations, logging=True)

import matplotlib.pyplot as plt

plt.plot(range(len(costs)), costs)
plt.title("1h1l PTC")
plt.xlabel("iteration")
plt.yscale("log")
plt.ylabel("cost")
plt.savefig("1h1l_ptc_training.png")

f = open("1h1l_ptc_scores.dat", "w")
for k, score in enumerate(costs):
    f.write(f"{k} {score}\n")
f.close()

# save weights
weights_lattice = [g.mspin(grid) for _ in ptc1.weights]
for i in range(len(ptc1.weights)):
    weights_lattice[i][:] = ptc1.weights[i].value
g.save(f"weights/1h1l_ptc_{iterations}it", weights_lattice)

# get preconditioning matrix operator
def network_matrix(dst, src):
    dst @= network(src)(with_gradients=False)

prec = g.matrix_operator(lambda d, s: network_matrix(d, s))

# get iteration count gain
src = g.vspincolor(grid)
src[:] = g.vspincolor([[1,1,1], [1,1,1], [1,1,1], [1,1,1]])
src @= src / g.norm2(src)**0.5

inv = g.algorithms.inverter

slv = inv.fgmres({"eps": 1e-6, "maxiter": 1000, "restartlen": 30})
sol = slv(w)(src)
it = len(slv.history)
print(f"Unpreconditioned: {it}")
print(f"{slv.history[0]} - {slv.history[-1]}")

slv_prec = inv.fgmres({"eps": 1e-6, "maxiter": 1000, "restartlen": 30, "prec": lambda x: prec})
sol_prec = slv_prec(w)(src)
it_prec = len(slv_prec.history)
print(f"Preconditioned: {it_prec}")
print(f"{slv_prec.history[0]} - {slv_prec.history[-1]}")

print(f"Iteration count gain: {it/it_prec}")
