import torch
from transformers.models.t5.modeling_t5 import T5LayerNorm

# class LoRAT5Transformer:
#     def 
# class TransformerRewardModel(torch.nn.Module):


class WeightedPooling(torch.nn.Module):
    def __init__(self, n_channels: int, normalize_output=False):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(n_channels)/torch.sqrt(n_channels))
        self.normalize_output = normalize_output


    def forward(self, inputs):
        # inputs = (batch, channels, input_dim) (pooling done across channels)
        # outputs = (batch, input_dim)
        weighted_sum = torch.einsum('j,ijk->ik', self.W, self.inputs)
        if self.normalize_output:
            variance = weighted_sum.pow(2).sum(-1, keepdim=1)
            return weighted_sum*torch.rsqrt(variance + self.variance_epsilon)
        else:
            return weighted_sum


class AveragePooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        # inputs = (batch, channels, input_dim) (pooling done across channels)
        return torch.mean(inputs, dim=1)


class AttentionPooling(torch.nn.Module):
    def __init__(self, n_input_dim: int, n_inner_dim: int, n_output_dim: int = None):
        super().__init__()
        
        if n_output_dim is None:
            n_output_dim = n_input_dim

        # Using query as a module that takes the keys as inpput is possible
        # because q doesn't depend on inputs and is instead a single learnable
        # query vector
        self.n_input_dim = n_input_dim
        self.n_inner_dim = n_inner_dim
        self.n_output_dim = n_output_dim

        self.layer_norm = T5LayerNorm(n_input_dim)

        self.q = torch.nn.Linear(n_inner_dim, 1, bias=False)
        self.k = torch.nn.Linear(n_input_dim, n_inner_dim, bias=False)
        self.v = torch.nn.Linear(n_input_dim, n_inner_dim, bias=False)
        self.o = torch.nn.Linear(n_inner_dim, n_output_dim, bias=False)


    def forward(self, inputs: torch.Tensor):
        # inputs = (batch, channels, input_dim) (pooling done across channels)

        # inputs = (batch, n_channels, n_input_dim)
        # keys = (batch, n_channels, n_inner_dim)
        normed_inputs = self.layer_norm(inputs)
        keys = self.k(normed_inputs)

        # attentions = (batch, n_channels)
        attentions = torch.softmax(self.q(keys).squeeze(2), dim=1)

        # values = (batch, n_channels, n_value_dim)
        values = self.v(inputs)
        return self.o(torch.einsum('ij,ijk->ik', attentions, values))


