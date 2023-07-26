import torch
import os
from rlhf_flant5.models.pooling import AttentionPooling
from rlhf_flant5.models.pooling import WeightedPooling
from itertools import chain

# from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.t5.modeling_t5 import T5LayerSelfAttention
from transformers.models.t5.modeling_t5 import T5LayerNorm

# class LoRAT5Transformer:
#     def 
# class TransformerRewardModel(torch.nn.Module):

DEBUG = int(os.environ.get('RLHF_DEBUG', '0'))

class LayerwiseWeightedAttentionPooling(torch.nn.Module):
    def __init__(self, pretrained_encoder: T5Stack, att_inner_dim: int, att_output_dim: int):
        super().__init__()

        def get_hidden_dim(block: T5Block):
            att_layer: T5LayerSelfAttention = block.layer[0]
            return att_layer.SelfAttention.d_model

        self.pretrained_encoder = pretrained_encoder
        self.layerwise_attention_pooling = [AttentionPooling(n_input_dim=get_hidden_dim(block),
                                                             n_inner_dim=att_inner_dim,
                                                             n_output_dim=att_output_dim) for block in pretrained_encoder.block]
        self.cross_layer_weighted_pooling = WeightedPooling(len(pretrained_encoder.block))
        self.readout = torch.nn.Linear(att_output_dim, 1)

    def forward(self, input_ids, **encoder_input_kwargs):
        # Note that pretrained_encoder is always in eval mode since it does not get trained
        encoder_outputs = self.pretrained_encoder.forward(input_ids, output_hidden_states=True)
        all_layer_hidden_states = encoder_outputs.hidden_states
        all_layerwise_pooled_states = torch.stack([attention_pooling(x) for attention_pooling, x
                                                   in zip(self.layerwise_attention_pooling, all_layer_hidden_states)], -2)
        final_pooled_output = self.cross_layer_weighted_pooling(all_layerwise_pooled_states)
        predicted_reward = self.readout(final_pooled_output).squeeze(-1)
        return predicted_reward

    # This affects parameters call as well
    def named_parameters(self):
        for x in super(torch.nn.Module, self).named_parameters():
            if not x[0].startswith('pretrained_encoder'):
                yield x

    def train(self, mode: bool=True):
        super().train(mode)
        if mode:
            self.pretrained_encoder.eval()