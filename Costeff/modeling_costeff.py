# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2022, Bowen Shen, NLP group of Institute of Information Engineering.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is adapted from HuggingFace Transformers
# https://github.com/huggingface/transformers

import logging
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.models.bert.modeling_bert import (ACT2FN, BertModel,
                                                    load_tf_weights_in_bert)

from Costeff.configuration_costeff import CosteffConfig

logger = logging.getLogger(__name__)


def entropy(x):
    with torch.no_grad():
        # x: torch.Tensor, logits BEFORE softmax
        # sotfmax normalized prob distribution
        x = torch.softmax(x, dim=-1)
        # entropy calculation on probs: -\sum(p \ln(p))
        return -torch.sum(x*torch.log(x), dim=-1)


class GradientRescaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        ctx.gd_scale_weight = weight
        output = input
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        input = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = ctx.gd_scale_weight * grad_outputs

        return grad_input, grad_weight


gradient_rescale = GradientRescaleFunction.apply


class CosteffEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # DIFF: Embedding low-rank factorization
        if not hasattr(config, 'emb_hidden_dim'):
            config.emb_hidden_dim = -1
        if config.emb_hidden_dim == -1:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        else:
            self.word_embeddings1 = nn.Embedding(
                config.vocab_size, config.emb_hidden_dim, padding_idx=config.pad_token_id)
            self.word_embeddings2 = nn.Linear(config.emb_hidden_dim, config.hidden_size, bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # DIFF: Embedding low-rank factorization
        if inputs_embeds is None:
            if self.config.emb_hidden_dim == -1:
                inputs_embeds = self.word_embeddings(input_ids)
            else:
                inputs_embeds = self.word_embeddings1(input_ids)
                inputs_embeds = self.word_embeddings2(inputs_embeds)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CosteffSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.prun_hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        # DIFF: hidden_size -> prun_hidden_size
        self.attention_head_size = int(config.prun_hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # the in-shape is not pruned
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_scores) if self.output_attentions else (
            context_layer,)  # FIXME: TinyBERT use attention_scores instead of attention_probs
        return outputs


class CosteffSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # DIFF: hidden_size -> prun_hidden_size
        # the out-shape is not pruned
        self.dense = nn.Linear(config.prun_hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CosteffAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = CosteffSelfAttention(config)
        self.output = CosteffSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        # Convert to set and remove already pruned heads
        heads = set(heads) - self.pruned_heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class CosteffIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # DIFF: intermediate_size -> prun_intermediate_size
        # the in-shape is not pruned
        self.dense = nn.Linear(config.hidden_size, config.prun_intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CosteffOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # DIFF: intermediate_size -> prun_intermediate_size
        # the out-shape is not pruned
        self.dense = nn.Linear(config.prun_intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CosteffLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CosteffAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = CosteffAttention(config)
        self.intermediate = CosteffIntermediate(config)
        self.output = CosteffOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class CosteffEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_layers = config.num_hidden_layers
        self.layer = nn.ModuleList([CosteffLayer(config) for _ in range(config.num_hidden_layers)])
        self.highway = nn.ModuleList([CosteffHighway(config, i)
                                      for i in range(config.num_hidden_layers-1)])
        self.early_exit_entropy = [-1 for _ in range(config.num_hidden_layers-1)]

    def init_highway_pooler(self, pooler):
        loaded_model = pooler.state_dict()
        for highway in self.highway:
            for name, param in highway.pooler.state_dict().items():
                param.copy_(loaded_model[name])

    def set_early_exit_entropy(self, x):
        if (type(x) is float) or (type(x) is int):
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x
        logger.info("  entropy: %s" % str(self.early_exit_entropy))

    def judge_early_exit(self, E):
        # maybe a more complex judgement with all the entropy in E
        i = len(E)-1
        if E[i] < self.early_exit_entropy[i]:
            return True
        return False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        highway_mode=False,
        exit_start=1,
    ):
        all_hidden_states = ()
        all_attentions = ()
        all_highway_exits = ()
        highway_entropy_list = []

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                encoder_attention_mask
            )
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                attention_map = layer_outputs[1]

            # rescale 1, HACK: i started from 0, so 1/(k-i) is actually 1/(k-(i-1))=1/(k-i+1)
            if self.training and highway_mode:
                hidden_states = gradient_rescale(hidden_states, 1.0 / (self.num_layers - i))

            # early exit
            if highway_mode and exit_start-1 <= i and i < self.num_layers-1:
                current_outputs = (hidden_states,)
                if self.output_hidden_states:
                    new_all_hiddens = all_hidden_states + (hidden_states,)
                    current_outputs = current_outputs + (new_all_hiddens,)
                if self.output_attentions:
                    new_all_attentions = all_attentions + (attention_map,)
                    current_outputs = current_outputs + (new_all_attentions,)
                highway_exit = self.highway[i](current_outputs)
                # logits, sequence_output (or pooled_output)
                if not self.training:
                    highway_logits = highway_exit[0]
                    highway_entropy = torch.max(entropy(highway_logits))  # TODO: simply using max for batch_size > 1?
                    highway_entropy_list.append(highway_entropy)
                    highway_exit = highway_exit + (highway_entropy,)  # logits, hidden_states(?), entropy
                    all_highway_exits = all_highway_exits + (highway_exit,)

                    if self.judge_early_exit(highway_entropy_list):
                        new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                        raise HighwayException(new_output, i+1)
                else:
                    all_highway_exits = all_highway_exits + (highway_exit,)

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            # rescale 2, HACK: i started from 0, so k-i-1 is actually k-(i-1)-1=k-i
            # the last layer has only the first rescale factor
            if i < self.num_layers-1:
                if self.training and highway_mode:
                    hidden_states = gradient_rescale(hidden_states, max(1, (self.num_layers - i - 1)))

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        outputs = outputs + (all_highway_exits,)
        # last-layer hidden state, (all hidden states), (all attentions), all_highway_exits
        return outputs


class CosteffPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.pooler_dense:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.hidden_mode = config.hidden_mode
            self.pooling_mode = config.pooling_mode
            self.last_n_hidden = config.last_n_hidden
        self.activation = nn.Tanh()

    def forward(self, all_hidden_states):
        if self.config.pooler_dense:
            hidden_states = all_hidden_states[-1]
            first_token_tensor = hidden_states[:, 0]
            pooled_output = self.dense(first_token_tensor)
            pooled_output = self.activation(pooled_output)
        else:
            if not isinstance(self.last_n_hidden, int) or self.last_n_hidden <= 0:
                raise ValueError("Invalid last_n_hidden %s. Expecting a positive int." % (self.last_n_hidden))

            if self.last_n_hidden == 1:
                # last 1 hidden
                hidden_states = all_hidden_states[-1]
            else:
                if self.hidden_mode == 'avg':
                    # average last n hidden, compatible when less than n layers
                    hidden_states = sum(all_hidden_states[-self.last_n_hidden:]
                                        )/len(all_hidden_states[-self.last_n_hidden:])
                elif self.hidden_mode == 'concat':
                    # concat last n hidden, compatible when less than n layers
                    if len(all_hidden_states) < self.last_n_hidden:
                        hidden_states = torch.cat(tuple(
                            [all_hidden_states[i] for i in range(len(all_hidden_states))]
                        ), dim=-1)
                    else:
                        hidden_states = torch.cat(tuple(
                            [all_hidden_states[i] for i in range(-self.last_n_hidden, 0)]
                        ), dim=-1)
                else:
                    raise ValueError("Invalid hidden_mode %s. Options: `avg`, `concat`." % (self.hidden_mode))

            if self.pooling_mode == 'avg':
                # average pooling seq_len
                first_token_tensor = hidden_states.mean(dim=1)  # [batch_size, hidn_size]
            elif self.pooling_mode == 'cls':
                # take the first token.
                first_token_tensor = hidden_states[:, 0, :]
            else:
                raise ValueError("Invalid pooling_mode %s. Options: `avg`, `cls`." % (self.pooling_mode))

            pooled_output = first_token_tensor
            pooled_output = self.activation(pooled_output)

        return pooled_output


class BertPreTrainedModel(PreTrainedModel):
    config_class = CosteffConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CosteffModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = CosteffEmbeddings(config)
        self.encoder = CosteffEncoder(config)
        self.pooler = CosteffPooler(config)

        self.init_weights()

    def init_highway_pooler(self):
        self.encoder.init_highway_pooler(self.pooler)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        highway_mode=False,
        exit_start=1,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            highway_mode=highway_mode,
            exit_start=exit_start
        )
        sequence_output = encoder_outputs[0]
        all_hidden_states = encoder_outputs[1]
        pooled_output = self.pooler(all_hidden_states)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions), highway exits
        return outputs


class HighwayException(Exception):
    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer  # start from 1!


class CosteffHighway(nn.Module):
    def __init__(self, config, i):
        super(CosteffHighway, self).__init__()
        self.pooler = CosteffPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.pooler_dense:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            if config.hidden_mode == 'concat':
                self.classifier = nn.Linear(config.hidden_size*min(config.last_n_hidden, i+2), config.num_labels)
            else:
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_outputs):
        sequence_output = encoder_outputs[0]
        all_hidden_states = encoder_outputs[1]
        pooled_output = self.pooler(all_hidden_states)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # logits = self.classifier(torch.relu(pooled_output))  # FIXME: TinyBERT use relu instead of dropout

        return logits, pooled_output


class TeacherBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(TeacherBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self._init_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        # sequence_output, pooled_output, x(hidden_states), x(attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # logits = self.classifier(torch.relu(pooled_output))  # FIXME: TinyBERT use relu instead of dropout
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None
        # loss, logits, sequence_output, (hidden_states), (attentions)
        outputs = (loss,) + (logits,) + (sequence_output,) + outputs[2:]
        return outputs


class CosteffForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(CosteffForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.bert = CosteffModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.pooler_dense:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            if config.hidden_mode == 'concat':
                self.classifier = nn.Linear(config.hidden_size*config.last_n_hidden, config.num_labels)
            else:
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        highway_mode=False,
        exit_start=1
    ):
        exit_layer = self.num_layers
        try:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                highway_mode=highway_mode,
                exit_start=exit_start
            )
            # sequence_output, pooled_output, (hidden_states), (attentions), highway exits
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            # logits = self.classifier(torch.relu(pooled_output))  # FIXME: TinyBERT use relu instead of dropout
            # add hidden states and attention if they are here
            # logits, sequence_output, pooled_output, (hidden_states), (attentions), highway exits
            outputs = (logits,) + (sequence_output,) + outputs[2:]
        except HighwayException as e:  # only when evaluating
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]

        if labels is not None:
            loss = None
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if highway_mode:
                # work with highway exits
                highway_loss = 0.0
                for highway_exit in outputs[-1]:
                    highway_logits = highway_exit[0]
                    if self.num_labels == 1:
                        #  We are doing regression
                        loss_fct = MSELoss()
                        highway_loss += loss_fct(highway_logits.view(-1), labels.view(-1))
                    else:
                        loss_fct = CrossEntropyLoss()
                        highway_loss += loss_fct(highway_logits.view(-1, self.num_labels), labels.view(-1))
                if loss is not None:  # HACK: add the last classifier!!!
                    highway_loss += loss
                outputs = (highway_loss,) + outputs
            else:
                outputs = (loss,) + outputs
        if not self.training:
            outputs = outputs + (exit_layer,)

        # loss, logits, sequence_output, (all_hidden_states), (all_attentions), highway_exit, (exit_layer)eval
        return outputs
