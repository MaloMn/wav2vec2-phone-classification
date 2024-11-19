import torch
import speechbrain as sb


class KeepOneLayer(torch.nn.Module):
    def __init__(self, encoder, layer_id):
        super().__init__()
        self.encoder = encoder
        self.layer_id = layer_id

    def forward(self, wav, wav_lens=None):
        hidden_states = self.encoder(wav)
        # raise Exception(f"Shape is {hidden_states.shape}")
        # Keeping only the specified layers
        return hidden_states[self.layer_id, :, :, :]
    

class Classifier(sb.nnet.containers.Sequential):
    """Block for linear layers."""
    def __init__(self, input_shape, layers, neurons, activation, dropout):
        super().__init__(input_shape=input_shape)

        for i in range(1, layers + 1):
            self.append(sb.nnet.linear.Linear, n_neurons=neurons, layer_name=f"fc{i}")
            self.append(activation(), layer_name=f"act{i}")
            self.append(dropout, layer_name=f"dropout{i}")
