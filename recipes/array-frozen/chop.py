import torch


class ChopSSLLayers(torch.nn.Module):
    def __init__(self, encoder, layer_id):
        super().__init__()
        self.encoder = encoder
        self.layer_id = layer_id

    def forward(self, wav, wav_lens=None):
        hidden_states = self.encoder(wav)
        print(hidden_states.shape)
        # Keeping only the specified layers
        return hidden_states[self.layer_id, :, :, :]
