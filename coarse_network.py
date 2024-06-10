import gpt as g

from gpt_ad_features import *

# monkey patch for vcomplex compatibility to ad
def infinitesimal_to_cartesian(self, A, dA):
    return dA

g.core.object_type.ot_vector_complex_additive_group.infinitesimal_to_cartesian = infinitesimal_to_cartesian
g.core.object_type.ot_matrix_complex_additive_group.infinitesimal_to_cartesian = infinitesimal_to_cartesian

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

# get grid and coarse grid
grid = U[0].grid
cgrid = g.block.grid(grid, blocksize)

# load blocking vectors
block_vectors = g.load(f"multigrid_setup/{config_name}")
block_map = g.block.map(cgrid, block_vectors)

# identity "gauge field" on coarse grid
I = [g.identity(g.mcomplex(cgrid, 24)) for _ in range(4)]

# wilson clover operator on fine and coarse grid
fermion_p = {"csw_r": 1.0,
             "csw_t": 1.0,
             "mass": -0.55,
             "boundary_phases": [1,1,1,1],
             "isAnisotropic": False,
             "xi_0": 1.0,
             "nu": 1.0
            }
w = g.qcd.fermion.wilson_clover(U, fermion_p)
cw = block_map.coarse_operator(w)

# set paths as non-redundant nearest neighbor hops
paths = [g.path(), g.path().backward(3)] + [g.path().forward(mu) for mu in range(4)]

# create the layers
lptc1 = lptc_layer(cgrid,
                   1,
                   1,
                   g.ot_vector_complex_additive_group(len(block_vectors)),
                   g.ot_matrix_complex_additive_group(len(block_vectors)),
                   I,
                   paths)

# define the network
def network(network_input):
    return lptc1(network_input)

# optimization parameters
sample_size = 5
iterations = 1000
alpha = 1

# example (coarse) source vector
src = g.vspincolor(grid)
src[:] = 1
csrc = block_map.project(src)

# training step
def training_step(alpha):
    source = rng.normal([csrc for _ in range(sample_size)])
    normalizations = [g.norm2(se) for se in source]

    training_outputs = [g(se / norm**0.5) for se, norm in zip(source, normalizations)]
    training_inputs = [rad.node(g(cw*oe)) for oe in training_outputs]

    # define cost
    cost = rad.node(0)
    for sample in range(sample_size):
        cost += 1/sample_size * g.norm2(network([training_inputs[sample],])[0] - training_outputs[sample])

    cost_val = cost()

    # update weights (this is simple gradient descent so far)
    for weight in lptc1.weights:
        weight.value -= alpha * weight.gradient

    return cost_val

costs = list()
for k in range(iterations):
    cc = training_step(alpha).real
    costs.append(cc)
    print(f"{k}: {cc}")

import matplotlib.pyplot as plt

plt.plot(range(iterations), [c.real for c in costs])
plt.title("1h1l coarse LPTC")
plt.xlabel("iteration")
plt.yscale("log")
plt.ylabel("cost")
plt.savefig("1h1l_coarse_lptc_training.png")

f = open("1h1l_ptc_scores.dat", "w")
for k, score in enumerate(costs):
    f.write(f"{k} {score}\n")
f.close()

# save weights
g.save(f"weights/1h1l_coarse_lptc_{iterations}it", [e.value for e in lptc1.weights])

# get preconditioning matrix operator
def network_matrix(dst, src):
    dst @= block_map.promote(network([block_map.project(src),])[0](with_gradients=False))

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
