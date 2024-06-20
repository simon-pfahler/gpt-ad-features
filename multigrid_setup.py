import gpt as g

# alias for reverse autodiff in gpt
rad = g.ad.reverse

# parameters
nbasisvectors = 12
blocksize = [4,4,4,4]

# aliases
inv = g.algorithms.inverter

# setup rng
rng = g.random("seed")

# load a gauge field
config_name = "1200.config"
gauge_field_path = "/glurch/scratch/knd35666/ensembles/ens001/" + config_name
U = g.load(gauge_field_path)

# get grid of the field
grid = U[0].grid

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

# define transitions between grids
def find_near_null_vectors(w, cgrid):
    slv = inv.fgmres({"eps": 1e-8, "maxiter": 1000, "restartlen": 25, "checkres": False})
    basis = g.orthonormalize(rng.cnormal([w.vector_space[0].lattice() for _ in range(12)]))
    null = g.lattice(basis[0])
    null[:] = 0
    for b in basis:
        slv(w)(b, null)
    #g.qcd.fermion.coarse.split_chiral(basis)
    bm = g.block.map(cgrid, basis)
    bm.orthonormalize()
    return basis

# calculate projection vectors
cgrid = g.block.grid(grid, blocksize)
vecs = find_near_null_vectors(w, cgrid)

# save projection vectors
g.save(f"multigrid_setup/{config_name}", vecs)
