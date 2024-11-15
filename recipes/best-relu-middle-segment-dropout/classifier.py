import speechbrain as sb


class Classifier(sb.nnet.containers.Sequential):
    """Block for linear layers."""
    def __init__(self, input_shape, layers, neurons, activation, dropout):
        super().__init__(input_shape=input_shape)

        for i in range(1, layers + 1):
            self.append(sb.nnet.linear.Linear, n_neurons=neurons, layer_name=f"fc{i}")
            self.append(activation(), layer_name=f"act{i}")
            self.append(dropout, layer_name=f"dropout{i}")
