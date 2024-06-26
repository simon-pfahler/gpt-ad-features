import gpt as g

from gpt_ad_features import pt_layer, dense_layer

rad = g.ad.reverse

class ptc_layer:
    def __init__(self, grid, num_in, num_out, ot_in, ot_weights, U, paths):
        self.grid = grid  # underlying grid
        self.num_in = num_in  # number of input nodes
        self.num_out = num_out  # number of output nodes
        self.ot_in = ot_in  # object type of the input
        self.ot_weights = ot_weights  # object type of the weights

        self.pt_layer = pt_layer(grid, num_in, ot_in, U, paths)
        self.dense_layer = dense_layer(grid, self.pt_layer.num_out, num_out, ot_in, ot_weights)
        
        self.weights = self.dense_layer.weights

    def __call__(self, layer_input):
        assert len(layer_input) == self.num_in

        return self.dense_layer(self.pt_layer(layer_input))

