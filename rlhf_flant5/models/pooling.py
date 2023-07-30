import torch
from transformers.models.t5.modeling_t5 import T5LayerNorm

# class LoRAT5Transformer:
#     def 
# class TransformerRewardModel(torch.nn.Module):


class WeightedPooling(torch.nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        n_channels_float = torch.as_tensor(n_channels, dtype=torch.float32)
        self.W = torch.nn.Parameter(torch.randn(n_channels)/torch.sqrt(n_channels_float))


    def forward(self, inputs):
        # inputs = (batch, channels, input_dim) (pooling done across channels)
        # outputs = (batch, input_dim)
        weighted_sum = torch.einsum('j,ijk->ik', self.W, inputs)
        return weighted_sum


class AveragePooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        # inputs = (batch, channels, input_dim) (pooling done across channels)
        return torch.mean(inputs, dim=1)


class AttentionPooling(torch.nn.Module):
    def __init__(self, n_input_dim: int):
        super().__init__()
        
        # Using query as a module that takes the keys as inpput is possible
        # because q doesn't depend on inputs and is instead a single learnable
        # query vector
        self.n_input_dim = n_input_dim
        self.q = torch.nn.Linear(n_input_dim, 1, bias=False)
        # self.k = torch.nn.Linear(n_input_dim, n_inner_dim, bias=False)

    def forward(self, inputs: torch.Tensor):
        # inputs = (batch, channels, input_dim) (pooling done across channels)

        # inputs = (batch, n_channels, n_input_dim)
        # relevances = (batch, n_channels)
        relevances = self.q(inputs).squeeze(2)

        # attentions = (batch, n_channels)
        attentions = torch.softmax(relevances, dim=1)

        # (batch, 1, n_channels) x (batch, n_channels, n_value_dim) = (batch, 1, n_value_dim)
        # return (batch, n_input_dim)
        return torch.matmul(attentions.unsqueeze(-2), inputs).squeeze(-2)


