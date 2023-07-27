import torch
from transformers.models.t5.modeling_t5 import T5LayerNorm

class GeGLU(torch.nn.Module):
    def __init__(self, input_dim, output_dim, normalize_input=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_input = normalize_input

        if self.normalize_input:
            self.layer_norm = T5LayerNorm(input_dim)

        self.W = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.V = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.act = torch.nn.GELU(approximate='tanh')

        # self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, inputs):
        hidden_gelu = self.act(self.W(inputs))
        hidden_linear = self.V(inputs)
        hidden_states = hidden_gelu * hidden_linear
        return hidden_states
