import gpt as g

from gpt_ad_features import *

# alias for reverse autodiff in gpt
rad = g.ad.reverse

# parameters
blocksize = [4,4,4,4]

# aliases
inv = g.algorithms.inverter

# setup rng
rng = g.random("seed")

# load a gauge field
config_name = "1200.config"
gauge_field_path = "/glurch/scratch/knd35666/ensembles/ens001/" + config_name
U = g.load(gauge_field_path)

# load weights
fine_weights = g.load("weights/1h1l_ptc")

# get grid and coarse grid
grid = U[0].grid
cgrid = g.block.grid(grid, blocksize)

# Wilson clover operator
fermion_p = {"csw_r": 1.0,
             "csw_t": 1.0,
             "mass": -0.55,
             "boundary_phases": [1,1,1,1],
             "isAnisotropic": False,
             "xi_0": 1.0,
             "nu": 1.0
            }
w = g.qcd.fermion.wilson_clover(U, fermion_p)

# load blocking vectors
block_vectors = g.load(f"multigrid_setup/{config_name}")
block_map = g.block.map(cgrid, block_vectors)

# identity "gauge field" on coarse grid
I = [g.identity(g.mcomplex(cgrid, 24)) for _ in range(4)]

# fine and coarse paths
fine_paths = [g.path(),] + [g.path().backward(mu) for mu in range(4)] + [g.path().forward(mu) for mu in range(4)]

# create the high mode model
fine_ptc = ptc_layer(grid,
                     1,
                     1,
                     g.object_type.ot_vector_spin_color(4,3),
                     g.object_type.ot_matrix_spin(4),
                     U,
                     fine_paths)
def fine_model(network_input):
    return fine_ptc(network_input)

# high-mode model matrix operator
def fine_network_matrix(dst, src):
    dst @= fine_model([src,])[0](with_gradients=False)
mh = g.matrix_operator(lambda d, s: fine_network_matrix(d, s))

# load weights of high-mode model
# set weights
for layer_weight, fine_weight in zip(fine_ptc.weights, fine_weights):
    layer_weight.value = fine_weight[0,0,0,0]

# create the layers
smoother = [ptc_layer(grid,
                      2,
                      2,
                      g.object_type.ot_vector_spin_color(4,3),
                      g.object_type.ot_matrix_spin(4),
                      U,
                      fine_paths) for _ in range(3)]
smoother.append(ptc_layer(grid,
                          2,
                          1,
                          g.object_type.ot_vector_spin_color(4,3),
                          g.object_type.ot_matrix_spin(4),
                          U,
                          fine_paths))

# define the network
def network(network_input):
    res = network_input
    for layer in smoother:
        res = layer(res)
    return res[0]

# initialize the weights
for layer in smoother:
    for weight in layer.weights:
        weight.value = rng.normal(weight.value, sigma=0.01)
    layer.weights[0].value += g.mspin([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

# define u_k+1 in iterative relaxation approach
def next_uk(uk, Mh, D, b):
    result = g.copy(b)
    result @= result - D*uk
    result @= Mh * result
    result @= uk + result
    return result

# optimization parameters
sample_size = 5
tolerance = 1e-12
iterations = 10000

# example null vector
null = g.vspincolor(grid)
null[:] = 0

# function to calculate cost and gradient
def cost_and_grad(weights):
    # set correct weights
    for i in range(len(weights)):
        layer_index = 0
        index_sum = 0
        while i >= index_sum + len(smoother[layer_index].weights):
            index_sum += len(smoother[layer_index].weights)
            layer_index += 1
        smoother[layer_index].weights[i - index_sum] = weights[i]

    source = rng.normal([g.vspincolor(grid) for _ in range(sample_size)])
    normalizations = [g.norm2(se) for se in source]
    source = [g(se / norm**0.5) for se, norm in zip(source, normalizations)]

    training_outputs = [g(next_uk(next_uk(se, mh, w, null), mh, w, null)) for se in source]
    training_inputs = [[rad.node(se), rad.node(null)] for se in source]

    # define cost
    cost = rad.node(0)
    for sample in range(sample_size):
        cost += 1/sample_size * g.norm2(network(training_inputs[sample]) - training_outputs[sample])

    cost /= 8*8*8*16

    cost_val = cost().real

    return cost_val

weight_list = list()
for layer in smoother:
    weight_list.extend(layer.weights)

# define optimizer
opt = adam(weight_list, cost_and_grad, alpha=1e-3)

_, (costs, _, _) = opt.optimize(tol=tolerance, maxiter=iterations, logging=True)

import matplotlib.pyplot as plt

plt.plot(range(len(costs)), costs)
plt.title("Smoother (1h4l PTC)")
plt.xlabel("iteration")
plt.yscale("log")
plt.ylabel("cost")
plt.savefig("1h4l_ptc_smoother_training.png")

f = open("1h4l_ptc_smoother_scores.dat", "w")
for k, score in enumerate(costs):
    f.write(f"{k} {score}\n")
f.close()

# save weights
weights_lattice = [[g.mspin(grid) for _ in layer.weights] for layer in smoother]
for i, layer in enumerate(smoother):
    for j, weight in enumerate(layer.weights):
        weights_lattice[i][j][:] = weight.value
g.save(f"weights/1h4l_ptc_smoother", weights_lattice)

# get preconditioning matrix operator
def network_matrix(dst, src):
    dst @= network([src, null])(with_gradients=False)

prec = g.matrix_operator(lambda d, s: network_matrix(d, s))

# get iteration count gain
src = g.vspincolor(grid)
src[:] = g.vspincolor([[1,1,1], [1,1,1], [1,1,1], [1,1,1]])
src @= src / g.norm2(src)**0.5

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

