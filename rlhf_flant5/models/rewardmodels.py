import torch
import os
from typing import List

# from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.t5.modeling_t5 import T5LayerSelfAttention
from transformers.models.t5.modeling_t5 import T5LayerNorm
from transformers.tokenization_utils_base import BatchEncoding

from rlhf_flant5.models.pooling import AttentionPooling
from rlhf_flant5.models.pooling import WeightedPooling
from rlhf_flant5.models.activations import GeGLU

# class LoRAT5Transformer:
#     def 
# class TransformerRewardModel(torch.nn.Module):

DEBUG = int(os.environ.get('RLHF_DEBUG', '0'))

class LayerStateAggregator(torch.nn.Module):
    def __init__(self, block: T5Block, attention_inner_dim, output_dim):
        super().__init__()

        def get_hidden_dim(block: T5Block):
            att_layer: T5LayerSelfAttention = block.layer[0]
            return att_layer.SelfAttention.d_model

        input_dim = get_hidden_dim(block)
        self.layer_norm = T5LayerNorm(input_dim)
        self.attention_pooling = AttentionPooling(n_input_dim=input_dim,
                                                  n_inner_dim=attention_inner_dim)
        self.readout = GeGLU(input_dim, output_dim)

    def forward(self, layer_output_states):
        """
        layer_output_states: should the output hidden states of the corresponding layer
        """
        return self.readout(self.attention_pooling(self.layer_norm(layer_output_states)))


class SimplePerceptronLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layer_norm = T5LayerNorm(input_dim)
        self.W = torch.nn.Linear(input_dim, output_dim)

    def forward(self, inputs):
        normed_inputs = self.layer_norm(inputs)
        outputs = self.W(normed_inputs)
        outputs = torch.nn.functional.gelu(outputs, approximate='tanh')
        return outputs


class RewardFromLayerwiseWeightedAttention(torch.nn.Module):
    def __init__(self, pretrained_encoder: T5Stack, attention_inner_dim: int, pooling_output_dim: int, readout_additional_layers: List[int] = []):
        super().__init__()

        self.pretrained_encoder = pretrained_encoder
        self.layerwise_aggregators = torch.nn.ModuleList([LayerStateAggregator(block, attention_inner_dim, pooling_output_dim) for block in pretrained_encoder.block])
        self.cross_layer_weighted_pooling = WeightedPooling(len(pretrained_encoder.block))

        readout_input_dim = pooling_output_dim
        readout_layers = []
        if readout_additional_layers:
            for lsize in readout_additional_layers:
                readout_layers.append(SimplePerceptronLayer(readout_input_dim, lsize, bias=False))
                readout_input_dim = lsize
        readout_layers.append(SimplePerceptronLayer(readout_input_dim, 1))
        self.readout_layers = torch.nn.ModuleList(readout_layers)

    def forward(self, tokenized_input: BatchEncoding):
        encoder_outputs = self.pretrained_encoder(tokenized_input.input_ids,
                                                  attention_mask=tokenized_input.attention_mask,
                                                  output_hidden_states=True)
        all_layer_hidden_states = encoder_outputs.hidden_states
        all_layerwise_pooled_states = torch.stack([layerwise_agg(x) for layerwise_agg, x
                                                   in zip(self.layerwise_aggregators, all_layer_hidden_states)], -2)
        final_pooled_output = self.cross_layer_weighted_pooling(all_layerwise_pooled_states)

        out = final_pooled_output
        for layer in self.readout_layers:
            out = layer(out)

        predicted_reward = out.squeeze(-1)
        return predicted_reward

    # # This affects parameters call as well
    # def named_parameters(self):
    #     for x in super(torch.nn.Module, self).named_parameters():
    #         if not x[0].startswith('pretrained_encoder'):
    #             yield x

    def train(self, mode: bool=True):
        super().train(mode)
        if mode:
            self.pretrained_encoder.eval()