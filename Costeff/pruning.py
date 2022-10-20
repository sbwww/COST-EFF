import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.utils.prune as prune
from transformers.models.bert.tokenization_bert import BertTokenizer

from Costeff.configuration_costeff import CosteffConfig
from Costeff.modeling_costeff import CosteffForSequenceClassification

logger = logging.getLogger(__name__)


class PruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = "unstructured"

    def __init__(self, prun_ratio, score):
        self.prun_ratio = prun_ratio
        self.score = score

    def compute_mask(self, t, default_mask):
        tensor_size = t.nelement()
        nparams_toprune = prune._compute_nparams_toprune(self.prun_ratio, tensor_size)
        mask = default_mask.clone()
        topk = torch.topk(
            torch.abs(self.score).view(-1), k=nparams_toprune, largest=False
        )
        # topk will have .indices and .values
        mask.view(-1)[topk.indices] = torch.nan
        return mask


def unstructured_pruning(module, name, prun_ratio, score):
    PruningMethod.apply(module, name, prun_ratio, score)
    return module


def Taylor_pruning_structured(model, ffn_prun_ratio, orig_heads, importance_file, config):
    keep_heads = config.num_attention_heads
    keep_layers = config.num_hidden_layers
    emb_hidden_dim = config.emb_hidden_dim
    head_size = int(config.prun_hidden_size / keep_heads)
    # Get modules to prune
    intermedia_modules, attn_modules = [], []
    for i in range(keep_layers):
        intermedia_modules += ["bert.encoder.layer.%d.intermediate.dense" % i,
                               "bert.encoder.layer.%d.output.dense" % i]
        attn_modules += ["bert.encoder.layer.%d.attention.self.query" % i,
                         "bert.encoder.layer.%d.attention.self.key" % i,
                         "bert.encoder.layer.%d.attention.self.value" % i,
                         "bert.encoder.layer.%d.attention.output.dense" % i]
    # structured pruning prunes input or output neurons of a matrix
    prune_out, prune_in = [], []
    for i in range(keep_layers):
        prune_out += ["bert.encoder.layer.%d.attention.self.query" % i,
                      "bert.encoder.layer.%d.attention.self.key" % i,
                      "bert.encoder.layer.%d.attention.self.value" % i,
                      "bert.encoder.layer.%d.intermediate.dense" % i]
        prune_in += ["bert.encoder.layer.%d.attention.output.dense" % i,
                     "bert.encoder.layer.%d.output.dense" % i]
    # Getting scores
    logger.info("  Loading Taylor score from %s..." % importance_file)  # HACK: importance_file is in ./temp
    taylor_dict = torch.load(importance_file)
    intermedia_scores, attn_scores = [], []
    all_head_indices = []
    for i in range(len(model.bert.encoder.layer)):
        score_attn_q = taylor_dict["bert.encoder.layer.%d.attention.self.query.weight" % i]
        score_attn_k = taylor_dict["bert.encoder.layer.%d.attention.self.key.weight" % i]
        score_attn_v = taylor_dict["bert.encoder.layer.%d.attention.self.value.weight" % i]
        score_attn_output = taylor_dict["bert.encoder.layer.%d.attention.output.dense.weight" % i]
        score_attn = score_attn_output.sum(0)
        attn_score_chunks = torch.split(score_attn, head_size)
        score_attn = torch.tensor([chunk.sum() for chunk in attn_score_chunks])
        attn_scores.append(score_attn)

        score_inter_in = taylor_dict["bert.encoder.layer.%d.intermediate.dense.weight" % i]
        score_inter_out = taylor_dict["bert.encoder.layer.%d.output.dense.weight" % i]
        score_inter = score_inter_in.sum(1) + score_inter_out.sum(0)
        intermedia_scores.append(score_inter)

    with torch.no_grad():
        # SVD iterative
        if emb_hidden_dim != -1:
            logger.info("  Factorizing Embedding Matrix...")
            if hasattr(model.bert.embeddings, "word_embeddings"):
                emb = model.bert.embeddings.word_embeddings.weight.data.numpy()
                del model.bert.embeddings.word_embeddings
            else:
                emb1 = model.bert.embeddings.word_embeddings1.weight.data.numpy()
                emb2 = model.bert.embeddings.word_embeddings2.weight.data.t().numpy()
                emb = np.matmul(emb1, emb2)
            u, s, v = np.linalg.svd(emb)
            s = np.eye(emb.shape[1])*s
            temp = np.dot(u[:, :emb_hidden_dim], s[:emb_hidden_dim, :emb_hidden_dim])
            new_word_emb1 = torch.from_numpy(temp)
            new_word_emb2 = torch.from_numpy(v[:emb_hidden_dim])
            model.bert.embeddings.word_embeddings1 = torch.nn.Embedding(
                config.vocab_size, emb_hidden_dim, padding_idx=0)
            model.bert.embeddings.word_embeddings1.weight.data = new_word_emb1.clone()
            model.bert.embeddings.word_embeddings2 = torch.nn.Linear(
                emb_hidden_dim, config.hidden_size, bias=False)
            model.bert.embeddings.word_embeddings2.weight.data = new_word_emb2.t().clone()

        logger.info("  Pruning Attention and FFN...")
        layer_id = 0
        for name, module in model.named_modules():
            if (name in attn_modules) and (not keep_heads > orig_heads):
                # Attention
                if layer_id >= len(all_head_indices):
                    score_attn = attn_scores[layer_id]
                    # nn.Linear weight matrix size is [out, in]
                    attn_size = module.weight.size(0)/float(orig_heads) if name in prune_out \
                        else module.weight.size(1)/float(orig_heads)
                    _, head_indices = torch.topk(score_attn, keep_heads, sorted=False)
                    head_indices, _ = torch.sort(head_indices)
                    logger.info("  Keep heads of layer %d: %s" % (layer_id, str(head_indices.tolist())))
                    all_head_indices.append(head_indices.tolist())
                if name in prune_out:
                    weight_chunks = torch.split(module.weight.data, int(attn_size), dim=0)
                    module.weight.data = torch.cat([weight_chunks[i] for i in head_indices], dim=0)
                    bias_chunks = torch.split(module.bias.data, int(attn_size))
                    module.bias.data = torch.cat([bias_chunks[i] for i in head_indices])
                elif name in prune_in:
                    weight_chunks = torch.split(module.weight.data, int(attn_size), dim=1)
                    module.weight.data = torch.cat([weight_chunks[i] for i in head_indices], dim=1)

            if (name in intermedia_modules) and (not ffn_prun_ratio > 1):
                # FFN
                score_inter = intermedia_scores[layer_id]
                expand_score = score_inter.expand(config.hidden_size, score_inter.size(0))
                if name in prune_out:
                    # HACK: unstructured pruning seems nonsense here but it's right!
                    # the score is expanded from [3072] to [768, 3072], each line is the same score_inter
                    # so during unstructured pruning, some entire columns are dropped, and then resized
                    # this is exactly structured!!!
                    unstructured_pruning(module, "bias", 1-ffn_prun_ratio, score_inter)
                    unstructured_pruning(module, "weight", 1-ffn_prun_ratio, expand_score.t())
                    prune.remove(module, "bias")
                    prune.remove(module, "weight")
                    module.bias.data = module.bias.data.masked_select(~torch.isnan(module.bias.data))
                    module.weight.data = module.weight.data.masked_select(
                        ~torch.isnan(module.weight.data)).view(-1, config.hidden_size)
                elif name in prune_in:
                    unstructured_pruning(module, "weight", 1-ffn_prun_ratio, expand_score)
                    prune.remove(module, "weight")
                    module.weight.data = module.weight.data.masked_select(
                        ~torch.isnan(module.weight.data)).view(config.hidden_size, -1)
                    layer_id += 1
    return model, all_head_indices


def prune_command(depth_or_width, model_path, keep_heads, keep_layers, ffn_hidden_dim, emb_hidden_dim):
    torch.manual_seed(42)
    logger.info("  Loading model from %s..." % model_path)
    config = CosteffConfig.from_pretrained(model_path)
    model = CosteffForSequenceClassification.from_pretrained(model_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)

    if depth_or_width == "depth":
        # this is for pruning layers
        model.bert.encoder.layer = torch.nn.ModuleList([model.bert.encoder.layer[i] for i in range(keep_layers)])
        new_config = config
        new_config.num_hidden_layers = keep_layers
        temp_dir = os.path.join(model_path, "temp")
    elif depth_or_width == "width":
        if ffn_hidden_dim > config.prun_intermediate_size or \
                (emb_hidden_dim > config.emb_hidden_dim and config.emb_hidden_dim != -1):
            raise ValueError("Cannot prune the model to a larger size!")

        ffn_prun_ratio = ffn_hidden_dim/config.prun_intermediate_size
        logger.info("  Pruning to %d heads, %d layers, %d FFN hidden dim, %d emb hidden dim..." %
                    (keep_heads, keep_layers, ffn_hidden_dim, emb_hidden_dim))

        temp_dir = os.path.join(model_path, "temp")
        importance_file = os.path.join(temp_dir, "taylor.pkl")
        orig_heads = config.num_attention_heads
        head_size = int(config.prun_hidden_size / orig_heads)

        new_config = config
        new_config.num_attention_heads = keep_heads
        new_config.num_hidden_layers = keep_layers
        new_config.prun_hidden_size = int(keep_heads*head_size)
        new_config.prun_intermediate_size = ffn_hidden_dim
        new_config.emb_hidden_dim = emb_hidden_dim

        model, all_head_indices = Taylor_pruning_structured(model, ffn_prun_ratio, orig_heads,
                                                            importance_file, new_config)
        #       head_to_distill and all_head_indices for each layer
        #         [0,1,2,3,4,5] and [0,1,2,3,5] -> [0,1,2,3,5]
        #         [0,1,2,3,5]   and [0,2,3,4]   -> [0,2,3,5]
        #         [0,2,3,5]     and [1,2,3]     -> [2,3,5]
        #         [2,3,5]       and [0,2]       -> [2,5]
        new_config.head_to_distill = [
            [model.config.head_to_distill[i][j] for j in all_head_indices[i]]
            for i in range(keep_layers)
        ]

    model.config = new_config

    output_dir = os.path.join(temp_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger.info("  Saving model to %s" % output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("  Number of parameters: %.2fM" %
                (sum([model.state_dict()[key].nelement() for key in model.state_dict()])/1e6))


def main():
    parser = argparse.ArgumentParser(description="pruning.py")
    parser.add_argument("--depth_or_width",  type=str,
                        default=None)
    parser.add_argument("--model_path", type=str,
                        help="distill type")
    parser.add_argument("--task", type=str,
                        help="Name of the task")
    parser.add_argument("--keep_heads", type=int,
                        help="the number of attention heads to keep")
    parser.add_argument("--keep_layers", type=int,
                        help="the number of layers of the pruned model")
    parser.add_argument("--ffn_hidden_dim", type=int,
                        help="Hidden size of the FFN subnetworks.")
    parser.add_argument("--emb_hidden_dim", type=int,
                        help="Hidden size of embedding factorization. Do not factorize embedding if value==-1")
    args = parser.parse_args()

    prune_command(args.depth_or_width, args.model_path, args.keep_heads, args.keep_layers,
                  args.ffn_hidden_dim, args.emb_hidden_dim)


if __name__ == "__main__":
    main()
