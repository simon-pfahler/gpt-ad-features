import gpt as g

from ad_features import parallel_transport

rad = g.ad.reverse

class pt_layer:
    def __init__(self, grid, num_in, ot_in, U, paths):
        self.grid = grid  # underlying grid
        self.num_in = num_in  # number of input nodes
        self.num_out = num_in * len(paths)  # number of output nodes
        self.ot_in = ot_in  # object type of the input
        self.U = [rad.node(Umu) for Umu in U]  # gauge field
        self.paths = paths  # paths to use

    def __call__(self, layer_input):
        assert len(layer_input) == self.num_in

        # create output nodes
        result = [rad.node(g.lattice(self.grid, self.ot_in)) for _ in range(self.num_out)]

        # calculate content of output nodes
        for b, phi in enumerate(layer_input):
            for i, path in enumerate(self.paths):
                result[i+len(self.paths)*b] = parallel_transport(path, self.U, phi)

        return result
