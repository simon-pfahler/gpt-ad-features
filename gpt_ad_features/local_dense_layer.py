import gpt as g

rad = g.ad.reverse

class local_dense_layer:
    def __init__(self, grid, num_in, num_out, ot_in, ot_weights):
        self.grid = grid  # underlying grid
        self.num_in = num_in  # number of input nodes
        self.num_out = num_out  # number of output nodes
        self.num_weights = num_in * num_out  # number of weights
        self.ot_in = ot_in  # object type of the input
        self.ot_weights = ot_weights  # object type of the weights

        # create weights
        self.weights = [rad.node(g.lattice(self.grid, self.ot_weights)) for _ in range(self.num_weights)]
    
    def __call__(self, layer_input):
        assert len(layer_input) == self.num_in

        # create and initialize output nodes
        result = [rad.node(g.lattice(self.grid, self.ot_in)) for _ in range(self.num_out)]
        for a, _ in enumerate(result):
            result[a] *= 0

        # calculate content of output nodes
        for a, _ in enumerate(result):
            for b, phi in enumerate(layer_input):
                result[a] += self.weights[b+self.num_in*a] * phi

        return result
