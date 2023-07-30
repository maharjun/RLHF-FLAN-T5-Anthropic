import torch
import os
from abc import abstractmethod
from typing import List, Optional

# from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.t5.modeling_t5 import T5LayerSelfAttention
from transformers.models.t5.modeling_t5 import T5LayerNorm
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.models.t5.modeling_t5 import T5Model
from transformers.tokenization_utils_base import BatchEncoding

from rlhf_flant5.models.pooling import AttentionPooling
from rlhf_flant5.models.pooling import WeightedPooling
from rlhf_flant5.models.basemodels import PretrainedEmbeddingsModel
from rlhf_flant5.models.activations import GeGLU

# class LoRAT5Transformer:
#     def 
# class TransformerRewardModel(torch.nn.Module):

DEBUG = int(os.environ.get('RLHF_DEBUG', '0'))

class AttentionWithGeGLU(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layer_norm = T5LayerNorm(input_dim)
        self.attention_pooling = AttentionPooling(n_input_dim=input_dim)
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
        self.W = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, inputs):
        normed_inputs = self.layer_norm(inputs)
        outputs = self.W(normed_inputs)
        outputs = torch.nn.functional.gelu(outputs, approximate='tanh')
        return outputs


class SimpleMLPWithGELU(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers: List[int]=[]):
        readout_input_dim = input_dim
        readout_layers = []
        if hidden_layers:
            for lsize in hidden_layers:
                readout_layers.append(SimplePerceptronLayer(readout_input_dim, lsize))
                readout_input_dim = lsize
        readout_layers.append(torch.nn.Linear(readout_input_dim, output_dim, bias=False))
        self.readout_layers = torch.nn.ModuleList(readout_layers)

    def forward(self, inputs):
        out = inputs
        for layer in self.readout_layers:
            out = layer(out)
        return out


class RewardFromLayerwiseWeightedAttention():
    def __init__(self, pretrained_transformer: T5Model, pooling_output_dim: int, readout_additional_layers: List[int] = [],
                 n_blocks_to_use: Optional[int] = None, use_pretrained_output=False):

        super().__init__()

        def get_hidden_dim(block: T5Block):
            att_layer: T5LayerSelfAttention = block.layer[0]
            return att_layer.SelfAttention.d_model

        self.n_blocks_to_use = n_blocks_to_use or len(pretrained_transformer.encoder.block)
        self.pretrained_encoder: T5Stack = pretrained_transformer.encoder
        self.layerwise_aggregators = torch.nn.ModuleList([AttentionWithGeGLU(get_hidden_dim(block), pooling_output_dim)
                                                          for block in self.pretrained_encoder.block[-self.n_blocks_to_use:]])
        self.cross_layer_weighted_pooling = WeightedPooling(len(self.pretrained_encoder.block))
        self.readout = SimpleMLPWithGELU(pooling_output_dim, 1, readout_additional_layers)


    def get_PTE(self, input_ids, attention_mask):
        encoder_outputs = self.pretrained_encoder(input_ids,
                                                  attention_mask=attention_mask,
                                                  output_hidden_states=True)
        all_layer_hidden_states = encoder_outputs.hidden_states[-self.n_blocks_to_use:]
        return {f'hidden_state_{i}': hidden_state for i, hidden_state in enumerate(all_layer_hidden_states)}


    def forward_with_PTE(self, *all_layer_hidden_states):
        all_layerwise_pooled_states = torch.stack([layerwise_agg(x) for layerwise_agg, x
                                                   in zip(self.layerwise_aggregators, all_layer_hidden_states)], -2)
        final_pooled_output = self.cross_layer_weighted_pooling(all_layerwise_pooled_states)
        predicted_reward = self.readout(final_pooled_output).squeeze(-1)
        return predicted_reward

    def train(self, mode: bool=True):
        super().train(mode)


class RewardFromDecoderOutput(PretrainedEmbeddingsModel):
    def __init__(self, pretrained_transformer: T5Model, readout_additional_layers: List[int] = [], use_pretrained_output=False):
        super().__init__(['decoder_output'], use_pretrained_output)

        self.pretrained_transformer = pretrained_transformer


        readout_input_dim = self.pretrained_transformer.shared.weight.shape[1]
        self.readout = SimpleMLPWithGELU(readout_input_dim, 1, readout_additional_layers)

    # This should have the same signature expected from forward
    def get_PTE(self, input_ids, attention_mask):
        decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
        transformer_outputs: Seq2SeqModelOutput = self.pretrained_transformer(input_ids, attention_mask, decoder_input_ids=decoder_input_ids)
        out = transformer_outputs.last_hidden_state.squeeze(-2)
        return {
            'decoder_output': out
        }

    def forward_with_PTE(self, decoder_output):
        return self.readout(decoder_output).squeeze(-1)

    def pretraining_output(self, input_ids, attention_mask):
        decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
        transformer_outputs: Seq2SeqModelOutput = self.pretrained_transformer(input_ids, attention_mask, decoder_input_ids=decoder_input_ids)
        out = transformer_outputs.last_hidden_state.squeeze(-2)
        return out


class RewardFromAttentionPooledEncoder(PretrainedEmbeddingsModel):
    def __init__(self, pretrained_transformer: T5Model, pooler_output_dim, readout_additional_layers: List[int] = [],
                 n_blocks_to_use: Optional[int] = None, use_pretrained_output=False):

        super().__init__(['encoder_hidden_states'], use_pretrained_output)

        def get_hidden_dim(block: T5Block):
            att_layer: T5LayerSelfAttention = block.layer[0]
            return att_layer.SelfAttention.d_model

        self.n_blocks_to_use = n_blocks_to_use or len(pretrained_transformer.encoder.block)
        self.pretrained_encoder = pretrained_transformer.encoder
        self.attention_pooling = AttentionWithGeGLU(get_hidden_dim(self.pretrained_encoder.block[0]), pooler_output_dim)

        readout_input_dim = self.pretrained_transformer.shared.weight.shape[1]
        self.readout = SimpleMLPWithGELU(readout_input_dim, 1, readout_additional_layers)

    # This should have the same signature expected from input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, inputs_embeds=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None
    def get_PTE(self, input_ids, attention_mask):
        encoder_outputs = self.pretrained_encoder.forward(input_ids, attention_mask, output_hidden_states=True)
        encoder_hidden_states = torch.stack([x[:, 0] for x in encoder_outputs.hidden_states[-self.n_blocks_to_use:]], dim=1)

        return {
            'encoder_hidden_states': encoder_hidden_states
        }

    def forward_with_PTE(self, encoder_hidden_states):
        attention_pool_out = self.attention_pooling(encoder_hidden_states)
        predicted_reward = self.readout(attention_pool_out).squeeze(-1)
        return predicted_reward

    def pretraining_output(self, input_ids, attention_mask):
        decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
        transformer_outputs: Seq2SeqModelOutput = self.pretrained_transformer(input_ids, attention_mask, decoder_input_ids=decoder_input_ids)
        out = transformer_outputs.last_hidden_state.squeeze(-2)
        return out


class RewardFromAttentionPooledDecoder(PretrainedEmbeddingsModel):
    def __init__(self, pretrained_transformer: T5Model, pooler_output_dim, readout_additional_layers: List[int] = [],
                 n_blocks_to_use: Optional[int] = None, use_pretrained_output=False):

        super().__init__(['encoder_hidden_states'], use_pretrained_output)

        def get_hidden_dim(block: T5Block):
            att_layer: T5LayerSelfAttention = block.layer[0]
            return att_layer.SelfAttention.d_model

        self.n_blocks_to_use = n_blocks_to_use or len(pretrained_transformer.encoder.block)
        self.pretrained_encoder = pretrained_transformer.encoder
        self.pretrained_decoder = pretrained_transformer.decoder
        self.attention_pooling = AttentionWithGeGLU(get_hidden_dim(self.pretrained_decoder.block[0]), pooler_output_dim)

        readout_input_dim = self.pretrained_transformer.shared.weight.shape[1]
        self.readout = SimpleMLPWithGELU(readout_input_dim, 1, readout_additional_layers)

    # This should have the same signature expected from input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, inputs_embeds=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None
    def get_PTE(self, input_ids, attention_mask):
        decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
        encoder_outputs = self.pretrained_encoder(input_ids, attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state
        decoder_outputs = self.pretrained_decoder(decoder_input_ids,
                                                  encoder_hidden_states=last_hidden_state,
                                                  encoder_attention_mask=attention_mask,
                                                  output_hidden_states=True)

        decoder_hidden_states = torch.stack([x[:, 0] for x in decoder_outputs.hidden_states[-self.n_blocks_to_use:]], dim=1)
        return {
            'decoder_hidden_states': decoder_hidden_states
        }

    def forward_with_PTE(self, decoder_hidden_states):
        attention_pool_out = self.attention_pooling(decoder_hidden_states)
        predicted_reward = self.readout(attention_pool_out).squeeze(-1)
        return predicted_reward

    def pretraining_output(self, input_ids, attention_mask):
        decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
        transformer_outputs: Seq2SeqModelOutput = self.pretrained_transformer(input_ids, attention_mask, decoder_input_ids=decoder_input_ids)
        out = transformer_outputs.last_hidden_state.squeeze(-2)
        return out


RewardModelNameMap = {
    'RewardFromLayerwiseWeightedAttention': RewardFromLayerwiseWeightedAttention,
    'RewardFromDecoderOutput': RewardFromDecoderOutput,
    'RewardFromAttentionPooledEncoder': RewardFromAttentionPooledEncoder,
    'RewardFromAttentionPooledDecoder': RewardFromAttentionPooledDecoder,
}