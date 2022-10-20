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

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss, MSELoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              SubsetRandomSampler, TensorDataset)
from tqdm import tqdm, trange
from transformers.data.metrics import glue_compute_metrics as compute_metrics
from transformers.data.processors.glue import (ColaProcessor,
                                               MnliMismatchedProcessor,
                                               MnliProcessor, MrpcProcessor,
                                               QnliProcessor, QqpProcessor,
                                               RteProcessor, Sst2Processor,
                                               StsbProcessor, WnliProcessor)
from transformers.data.processors.glue import glue_output_modes as output_modes
from transformers.data.processors.utils import (DataProcessor, InputExample,
                                                InputFeatures)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.optimization import (AdamW,
                                       get_constant_schedule_with_warmup,
                                       get_linear_schedule_with_warmup)

from Costeff.configuration_costeff import CosteffConfig
from Costeff.modeling_costeff import (CosteffForSequenceClassification,
                                      TeacherBertForSequenceClassification)
from Costeff.pruning import prune_command
from transformers.tokenization_utils import PreTrainedTokenizer

csv.field_size_limit(sys.maxsize)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler("info.log")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

try:
    from tensorboardX import SummaryWriter
    _has_tensorboard = True
except ImportError:
    _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


acc_tasks = ["sst-2", "qnli", "rte"]  # HACK: "mnli", "mnli-mm" are seperatly used
f1_tasks = ["mrpc", "qqp"]
corr_tasks = ["sts-b"]
mcc_tasks = ["cola"]


processors = {
    "cola": ColaProcessor,
    "sst-2": Sst2Processor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "sts-b": StsbProcessor,
    "mrpc": MrpcProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor
}


# Prepare seed
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# Prepare loss functions
def softLabelLoss(predicts, targets):
    # CE
    student_prob = predicts
    teacher_prob = torch.softmax(targets, dim=-1)
    loss_fct = CrossEntropyLoss()
    # KLDiv
    # student_prob = torch.log_softmax(predicts, dim=-1)
    # teacher_prob = torch.softmax(targets, dim=-1)
    # loss_fct = KLDivLoss(reduction="batchmean")

    soft_loss = loss_fct(student_prob, teacher_prob)
    return soft_loss


def get_tensor_data(output_mode, features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if features[0].label is not None:
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        tensor_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    else:
        all_labels = None
        tensor_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    return tensor_data, all_labels


def result_to_file(result, task_name, file_name):
    with open(file_name, "a") as writer:
        logger.info("***** %s Eval results *****", task_name)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("\n")


def output_test_result(args, preds, label_list, output_test_file):
    if args.task_name == "sts-b":
        test_pred_label = [tr for tr in preds]
    else:
        test_pred_label = [label_list[tr] for tr in preds]
    test_pred = pd.DataFrame({"index": range(len(test_pred_label)), "prediction": test_pred_label})
    test_pred.to_csv(output_test_file, sep="\t", index=False)


def pruning_schedule(prun_times, max_prune_times,
                     orig_heads, orig_ffn_dim, orig_emb_dim,
                     final_ffn_dim, final_emb_dim):
    keep_heads = orig_heads - prun_times
    ffn_hidden_dim = orig_ffn_dim - (orig_ffn_dim - final_ffn_dim) * (prun_times / max_prune_times)
    emb_hidden_dim = -1 if final_emb_dim == -1 else \
        orig_emb_dim - (orig_emb_dim - final_emb_dim) * (prun_times / max_prune_times)
    return keep_heads, ffn_hidden_dim, emb_hidden_dim


def init_prun(args, tlayer_num, tatt_heads, num_train_examples):
    if args.depth_or_width == "depth":
        max_prune_times = tlayer_num - args.keep_layers
    elif args.depth_or_width == "width":
        max_prune_times = tatt_heads - args.keep_heads
    if max_prune_times == 0:
        prune_step = 0
    else:
        steps_per_epoch = int(num_train_examples/args.train_batch_size/args.gradient_accumulation_steps)
        prune_step = int(args.prun_period_proportion * steps_per_epoch)
    logger.info("  Max prune times: %d" % max_prune_times)
    logger.info("  Prune step: %d" % prune_step)
    return max_prune_times, prune_step


class TaylorGrad():
    def __init__(self, model):
        # Initialize accumulated grad
        self.data_accumulator_grad = {}
        for n, p in model.named_parameters():
            self.data_accumulator_grad[n] = torch.full_like(p.data, 0)

    def accumulate_grad(self, model):
        for n, p in model.named_parameters():
            if p.grad is not None:
                self.data_accumulator_grad[n].add_(p.grad.data.abs())

    def get_taylor(self, model):
        taylor = {}
        for n, p in model.named_parameters():
            taylor[n] = torch.abs(self.data_accumulator_grad[n].mul(p.data))
        return taylor

    def clear_accugrad(self, model):
        for n, p in model.named_parameters():
            self.data_accumulator_grad[n] = torch.full_like(p.data, 0)


def iterative_pruning(args, student_model, teacher_config, scheduler,
                      prune_step, max_prune_times, prun_times, global_step, taylor_score=None):
    if args.depth_or_width == "width":
        if prun_times < max_prune_times:
            keep_heads, ffn_hidden_dim, emb_hidden_dim = pruning_schedule(
                prun_times, max_prune_times,
                teacher_config.num_attention_heads, teacher_config.intermediate_size, teacher_config.hidden_size,
                args.ffn_hidden_dim, args.emb_hidden_dim
            )
        else:
            keep_heads = teacher_config.num_attention_heads-prun_times
            ffn_hidden_dim = args.ffn_hidden_dim
            emb_hidden_dim = args.emb_hidden_dim
        keep_layers = student_model.config.num_hidden_layers
    elif args.depth_or_width == "depth":
        keep_heads, ffn_hidden_dim, emb_hidden_dim = student_model.config.num_attention_heads, student_model.config.intermediate_size, -1
        keep_layers = student_model.config.num_hidden_layers - 1

    lr = scheduler.get_last_lr()[0]
    logger.info("  Learning rate = %.4e" % lr)

    score_dict = taylor_score

    temp_dir = os.path.join(args.output_dir, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    logger.info("  Saving Taylor score to %s before pruning..." % temp_dir)
    torch.save(score_dict, os.path.join(temp_dir, "taylor.pkl"))

    prune_command(args.depth_or_width, args.output_dir, int(keep_heads), int(
        keep_layers), int(ffn_hidden_dim), int(emb_hidden_dim))

    logger.info("  Loading model from %s after pruning..." % temp_dir)
    config = CosteffConfig.from_pretrained(temp_dir)
    student_model = CosteffForSequenceClassification.from_pretrained(temp_dir, config=config)
    student_model.to(args.device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}]
    lr = args.learning_rate*((1+args.lr_restore_factor)**prun_times)  # enlarge lr after pruning
    remain_steps = args.max_optimization_steps-global_step
    logger.info("  Learning rate = %.4e" % lr)
    logger.info("  Remain step = %d" % remain_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(
        10, int(prune_step*args.warmup_proportion)), num_training_steps=prune_step)
    return student_model, optimizer, scheduler, remain_steps


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info(f"Using label list {label_list} for task {task}")
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info(f"Using output mode {output_mode} for task {task}")

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info(f"guid: {example.guid}")
        logger.info(f"features: {features[i]}")

    return features


def build_dataloader(set_type, args, processor, label_list, tokenizer, output_mode):
    if set_type == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif set_type == "eval":
        examples = processor.get_dev_examples(args.data_dir)
    elif set_type == "test":
        examples = processor.get_test_examples(args.data_dir)
    features = convert_examples_to_features(
        examples=examples, tokenizer=tokenizer, max_length=args.max_seq_length, output_mode=output_mode,
        label_list=label_list)
    data, labels = get_tensor_data(output_mode, features)
    sampler = SequentialSampler(data) if set_type == "test" else RandomSampler(data)
    batch_size = args.eval_batch_size if set_type == "eval" or set_type == "test" else args.train_batch_size
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader, labels, data


def do_test(args, model, test_dataloader, highway_mode=False):
    model.eval()
    preds = None
    exit_layer_counter = {(i+1): 0 for i in range(model.num_layers)}
    for batch_ in tqdm(test_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(args.device) for t in batch_)
        with torch.no_grad():
            inputs = {"input_ids":      batch_[0],
                      "attention_mask": batch_[1],
                      "token_type_ids": batch_[2]}
            if not args.train_ft:
                inputs["highway_mode"] = highway_mode

            outputs = model(**inputs)
            if highway_mode:
                exit_layer_counter[outputs[-1]] += args.eval_batch_size
            logits = outputs[0]
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            new_preds = logits.detach().cpu().numpy()
            preds = np.append(preds, new_preds, axis=0)

    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    if highway_mode:
        logger.info("  Exit layer counter %s", exit_layer_counter)
        actual_cost = sum([l*c for l, c in exit_layer_counter.items()])
        full_cost = len(test_dataloader.dataset) * model.num_layers
        speedup = 1.0/(actual_cost/full_cost)
        logger.info("  Expected speed up %.2fx" % speedup)

    model.train()
    return preds


def do_test_prof(args, model, test_dataloader, highway_mode=False):
    # Calculate parameters size
    size = 0
    for n, p in model.named_parameters():
        padding = "-"*(60-len(n))
        logger.info("  %s %s %d" % (n, padding, p.nelement()))
        size += p.nelement()
    logger.info("  Total parameters: %.2fM" % (size/1e6))

    model.eval()
    forward_flops = 0
    preds = None
    exit_layer_counter = {(i+1): 0 for i in range(model.num_layers)}
    for batch_ in tqdm(test_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(args.device) for t in batch_)
        with torch.no_grad():
            inputs = {"input_ids":      batch_[0],
                      "attention_mask": batch_[1],
                      "token_type_ids": batch_[2]}
            if not args.train_ft:
                inputs["highway_mode"] = highway_mode

            if preds is not None and len(preds) < args.max_profile_samples:
                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:
                    outputs = model(**inputs)
                events = prof.events()
                forward_flops += sum([evt.flops for evt in events])
            else:
                outputs = model(**inputs)
            if highway_mode:
                exit_layer_counter[outputs[-1]] += args.eval_batch_size
            logits = outputs[0]
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            new_preds = logits.detach().cpu().numpy()
            preds = np.append(preds, new_preds, axis=0)

    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    test_num = min(len(test_dataloader.dataset), args.max_profile_samples)
    logger.info("  Forward FLOPs: %.2fM" % (forward_flops/test_num/1e6))

    if highway_mode:
        logger.info("  Exit layer counter %s", exit_layer_counter)
        actual_cost = sum([l*c for l, c in exit_layer_counter.items()])
        full_cost = len(test_dataloader.dataset) * model.num_layers
        speedup = 1.0/(actual_cost/full_cost)
        logger.info("  Expected speed up %.2fx" % speedup)

    model.train()
    return preds


def do_eval(args, model, eval_dataloader, highway_mode=False):
    # Calculate parameters size
    size = 0
    for n, p in model.named_parameters():
        padding = "-"*(60-len(n))
        logger.info("  %s %s %d" % (n, padding, p.nelement()))
        size += p.nelement()
    logger.info("  Total parameters: %.2fM" % (size/1e6))

    eval_loss = 0
    nb_eval_steps = 0
    preds = None
    exit_layer_counter = {(i+1): 0 for i in range(model.num_layers)}
    exit_layer_counter_correct = {(i+1): 0 for i in range(model.num_layers)}  # 正确结果数量统计
    model.eval()
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(args.device) for t in batch_)
        with torch.no_grad():
            inputs = {"input_ids":      batch_[0],
                      "attention_mask": batch_[1],
                      "token_type_ids": batch_[2],
                      "labels":         batch_[3]}

            if not args.train_ft:
                inputs["highway_mode"] = highway_mode
            outputs = model(**inputs)

            if highway_mode:
                exit_layer_counter[outputs[-1]] += args.eval_batch_size
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            if highway_mode:
                exit_layer_counter_correct[outputs[-1]] += sum(np.argmax(preds, axis=-1) == out_label_ids)
        else:
            new_preds = logits.detach().cpu().numpy()
            preds = np.append(preds, new_preds, axis=0)
            new_out_label_ids = inputs["labels"].detach().cpu().numpy()
            out_label_ids = np.append(out_label_ids, new_out_label_ids, axis=0)
            if highway_mode:
                exit_layer_counter_correct[outputs[-1]] += sum(np.argmax(new_preds, axis=-1) == new_out_label_ids)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(args.task_name, preds, out_label_ids)
    result["eval_loss"] = eval_loss

    if highway_mode:
        logger.info("  Exit layer counter %s", exit_layer_counter)
        layer_acc = {
            i: round(c2 / max(1, c1)*100, 2) for i, c1, c2 in
            zip(exit_layer_counter.keys(),
                exit_layer_counter.values(),
                exit_layer_counter_correct.values())}
        logger.info("  Exit layer acc %s", layer_acc)  # 输出正确百分比
        actual_cost = sum([l*c for l, c in exit_layer_counter.items()])
        full_cost = len(eval_dataloader.dataset) * model.num_layers
        speedup = 1.0/(actual_cost/full_cost)
        logger.info("  Expected speed up %.2fx" % speedup)
        result["speedup"] = speedup

    model.train()
    return result


def do_eval_layer(args, model, eval_dataloader):
    # Calculate parameters size
    size = 0
    for n, p in model.named_parameters():
        padding = "-"*(60-len(n))
        logger.info("  %s %s %d" % (n, padding, p.nelement()))
        size += p.nelement()
    logger.info("  Total parameters: %.2fM" % (size/1e6))

    model.eval()
    model.bert.encoder.set_early_exit_entropy(-1)

    preds = np.zeros(shape=(model.num_layers, len(eval_dataloader.dataset), args.num_labels))
    out_label_ids = np.zeros(shape=(len(eval_dataloader.dataset)))

    sample_idx = 0
    bs = eval_dataloader.batch_size
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(args.device) for t in batch_)
        with torch.no_grad():
            inputs = {"input_ids":      batch_[0],
                      "attention_mask": batch_[1],
                      "token_type_ids": batch_[2],
                      "labels":         batch_[3]}

            if not args.train_ft:
                inputs["highway_mode"] = True
            outputs = model(**inputs)

        all_highway_exits = outputs[-2]
        _, logits = outputs[:2]

        new_out_label_ids = inputs["labels"].detach().cpu().numpy()
        out_label_ids[sample_idx:sample_idx+bs] = new_out_label_ids

        for i, highway_exit in enumerate(all_highway_exits):
            highway_logits = highway_exit[0]
            new_preds = highway_logits.detach().cpu().numpy()
            preds[i, sample_idx:sample_idx+bs] = new_preds

        preds[-1, sample_idx:sample_idx+bs] = logits.detach().cpu().numpy()
        sample_idx += bs

    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=-1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    layer_avg = 0.0
    for i, layer_preds in enumerate(preds):
        if i < args.exit_start - 1:
            continue
        result = compute_metrics(args.task_name, layer_preds, out_label_ids)
        logger.info("  Layer %d: %s" % (i, str(result)))
        if args.task_name in ["mnli", "mnli-mm"]:
            layer_avg += result[args.task_name+"/acc"]
        elif args.task_name in acc_tasks:
            layer_avg += result["acc"]
        elif args.task_name in f1_tasks:
            layer_avg += result["f1"]
        elif args.task_name in corr_tasks:
            layer_avg += result["corr"]
        elif args.task_name in mcc_tasks:
            layer_avg += result["mcc"]
    model.train()
    layer_avg /= (len(preds) - args.exit_start + 1)

    return result, layer_avg


def do_eval_prof(args, model, eval_dataloader, highway_mode=False):
    # Calculate parameters size
    size = 0
    for n, p in model.named_parameters():
        padding = "-"*(60-len(n))
        logger.info("  %s %s %d" % (n, padding, p.nelement()))
        size += p.nelement()
    logger.info("  Total parameters: %.2fM" % (size/1e6))

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    forward_flops = 0
    preds = None
    exit_layer_counter = {(i+1): 0 for i in range(model.num_layers)}
    exit_layer_counter_correct = {(i+1): 0 for i in range(model.num_layers)}  # 正确结果数量统计
    for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ = tuple(t.to(args.device) for t in batch_)
        with torch.no_grad():
            inputs = {"input_ids":      batch_[0],
                      "attention_mask": batch_[1],
                      "token_type_ids": batch_[2],
                      "labels":         batch_[3]}

            if not args.train_ft:
                inputs["highway_mode"] = highway_mode

            if preds is not None and len(preds) < args.max_profile_samples:
                with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True) as prof:
                    outputs = model(**inputs)
                events = prof.events()
                forward_flops += sum([evt.flops for evt in events])
            else:
                outputs = model(**inputs)
            if highway_mode:
                exit_layer_counter[outputs[-1]] += args.eval_batch_size
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            if not args.train_ft:
                exit_layer_counter_correct[outputs[-1]] += sum(np.argmax(preds, axis=-1) == out_label_ids)
        else:
            new_preds = logits.detach().cpu().numpy()
            preds = np.append(preds, new_preds, axis=0)
            new_out_label_ids = inputs["labels"].detach().cpu().numpy()
            out_label_ids = np.append(out_label_ids, new_out_label_ids, axis=0)
            if not args.train_ft:
                exit_layer_counter_correct[outputs[-1]] += sum(np.argmax(new_preds, axis=-1) == new_out_label_ids)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(args.task_name, preds, out_label_ids)
    result["eval_loss"] = eval_loss

    test_num = min(len(eval_dataloader.dataset), args.max_profile_samples)
    logger.info("  Forward FLOPs: %.2fM" % (forward_flops/test_num/1e6))

    if highway_mode:
        logger.info("  Exit layer counter %s", exit_layer_counter)
        layer_acc = {
            i: round(c2 / max(1, c1)*100, 2) for i, c1, c2 in
            zip(exit_layer_counter.keys(),
                exit_layer_counter.values(),
                exit_layer_counter_correct.values())}
        logger.info("  Exit layer acc %s", layer_acc)  # 输出正确百分比
        actual_cost = sum([l*c for l, c in exit_layer_counter.items()])
        full_cost = len(eval_dataloader.dataset) * model.num_layers
        speedup = 1.0/(actual_cost/full_cost)
        logger.info("  Expected speed up %.2fx" % speedup)
        result["speedup"] = speedup

    model.train()
    return result


def taylor_fake_iteration(args, student_model_path, teacher_model, train_dataset, highway_mode):
    data_size = len(train_dataset)
    taylor_indices = np.random.choice(range(data_size), int(data_size*args.taylor_proportion))
    taylor_sampler = SubsetRandomSampler(taylor_indices)
    taylor_dataloader = DataLoader(train_dataset, sampler=taylor_sampler, batch_size=args.train_batch_size)

    logger.info("  Loading Model from %s..." % student_model_path)
    config = CosteffConfig.from_pretrained(student_model_path,
                                           num_labels=args.num_labels,
                                           finetuning_task=args.task_name,
                                           output_attentions=args.output_attentions,
                                           output_hidden_states=args.output_hidden_states)
    # disable dropout
    orig_hidden_dropout_prob = config.hidden_dropout_prob
    orig_attention_probs_dropout_prob = config.attention_probs_dropout_prob
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    student_model = CosteffForSequenceClassification.from_pretrained(student_model_path, config=config)
    student_model.bert.encoder.set_early_exit_entropy(args.early_exit_entropy)
    student_model.to(args.device)

    taylor_grad = TaylorGrad(student_model)

    tr_loss = 0.
    tr_cls_loss = 0.
    tr_rep_loss = 0.
    for step, batch in enumerate(tqdm(taylor_dataloader, desc="Iteration")):
        batch = tuple(t.to(args.device) for t in batch)
        if batch[0].size()[0] != args.train_batch_size:
            continue

        inputs = {"input_ids":      batch[0],
                  "attention_mask": batch[1],
                  "token_type_ids": batch[2],
                  "labels":         batch[3]}

        cls_loss = 0.
        rep_loss = 0.

        student_model.train()
        teacher_model.eval()

        if args.internal_loss:
            inputs["highway_mode"] = highway_mode
            inputs["exit_start"] = args.exit_start
        else:
            inputs["highway_mode"] = False
        outputs = student_model(**inputs)
        hard_label_loss = outputs[0]
        student_logits = outputs[1]
        if args.train_costeff or args.train_ta and highway_mode:
            student_highway = outputs[-1]
        if args.output_hidden_states:
            student_hidns = outputs[3]  # all_hidden_states
            if args.output_attentions:
                student_attns = outputs[4]  # all_attentions
        elif args.output_attentions:
            student_attns = outputs[3]  # all_attentions

        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            teacher_logits = teacher_outputs[1]
            if args.train_costeff and highway_mode:
                teacher_highway = teacher_outputs[-2]
            if args.output_hidden_states:
                teacher_hidns = teacher_outputs[3]  # all_hidden_states
                if args.output_attentions:
                    teacher_attns = teacher_outputs[4]  # all_attentions
            elif args.output_attentions:
                teacher_attns = teacher_outputs[3]  # all_attentions

        if not args.pred_distill and not args.repr_distill:
            cls_loss = hard_label_loss
            tr_cls_loss += cls_loss.item()
        # soft cls loss grad
        if args.pred_distill:
            if args.output_mode == "classification":
                cls_loss = softLabelLoss(student_logits / args.temperature,
                                         teacher_logits / args.temperature)
                if args.train_costeff and highway_mode:
                    for student_exit, teacher_exit in zip(student_highway, teacher_highway):
                        student_highway_logits = student_exit[0]
                        teacher_highway_logits = teacher_exit[0]
                        cls_loss += softLabelLoss(student_highway_logits / args.temperature,
                                                  teacher_highway_logits / args.temperature)
                    cls_loss = 0.9*cls_loss + 0.1*hard_label_loss  # FIXME: hard-coded
            elif args.output_mode == "regression":
                loss_fct = MSELoss()
                cls_loss = loss_fct(student_logits.view(-1), teacher_logits.view(-1))
            tr_cls_loss += cls_loss.item()

        # hidn loss grad
        if args.repr_distill:
            non_padding_masks = batch[1].float()
            # [batch_size, seq_len]
            batch_non_padding_len = non_padding_masks.sum(dim=-1)
            # [batch_size]
            for student_hidn, teacher_hidn in zip(student_hidns, teacher_hidns):
                # [batch_size, seq_len, hidden_size]
                non_padding_masks_hidn = non_padding_masks.unsqueeze(-1)
                student_hidn = student_hidn.mul(non_padding_masks_hidn)
                teacher_hidn = teacher_hidn.mul(non_padding_masks_hidn)
                loss_fct = MSELoss()
                mean_loss = loss_fct(student_hidn, teacher_hidn)
                rep_loss += mean_loss
            tr_rep_loss += rep_loss.item()

        loss = cls_loss + rep_loss

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        tr_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            taylor_grad.accumulate_grad(student_model)
            student_model.zero_grad()

    loss = tr_loss / (step + 1)
    cls_loss = tr_cls_loss / (step + 1)
    rep_loss = tr_rep_loss / (step + 1)
    result = {}
    result["cls_loss"] = cls_loss
    result["rep_loss"] = rep_loss
    result["loss"] = loss
    logger.info("  Before pruning\n%s", str(result))
    # restore dropout
    student_model.config.hidden_dropout_prob = orig_hidden_dropout_prob
    student_model.config.attention_probs_dropout_prob = orig_attention_probs_dropout_prob

    taylor_score = taylor_grad.get_taylor(student_model)
    del taylor_grad
    return taylor_score


# Joint training, for TA and COST-EFF training
def do_train_1t(args, train_dataloader, eval_dataloader, student_model, teacher_model, tokenizer):
    logger.info("  This is joint training")
    if args.n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
        if teacher_model is not None:
            teacher_model = torch.nn.DataParallel(teacher_model)

    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in student_model.named_parameters() if (not any(nd in n for nd in no_decay))],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in student_model.named_parameters() if (any(nd in n for nd in no_decay))],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.depth_or_width is not None:
        max_prune_times, prune_step = init_prun(
            args, teacher_model.config.num_hidden_layers, teacher_model.config.num_attention_heads,
            len(train_dataloader.dataset))
        prun_times = 0
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(prune_step*args.warmup_proportion),
            num_training_steps=args.max_optimization_steps)
    else:
        # save model if prun_times < max_prune_times
        prun_times, max_prune_times, prune_step = -1, -1, -1
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(args.max_optimization_steps*args.warmup_proportion),
            num_training_steps=args.max_optimization_steps)

    # Train and evaluate
    global_step = 0
    best_dev_score = 0.0
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    student_model.zero_grad()

    temp_pred_distill = args.pred_distill
    temp_repr_distill = args.repr_distill
    for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0.
        tr_rep_loss = 0.
        tr_cls_loss = 0.

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(args.device) for t in batch)
            if batch[0].size()[0] != args.train_batch_size:
                continue

            inputs = {"input_ids":      batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels":         batch[3]}

            if args.depth_or_width is not None \
                    and prune_step > 0 and (global_step+1) % prune_step == 0 \
                    and prun_times < max_prune_times:
                # Pruning
                model_to_save = student_model.module if hasattr(student_model, "module") else student_model
                output_dir = os.path.join(args.output_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                logger.info("  Saving model to %s before pruning..." % output_dir)
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info("  Compute taylor score before pruning\n")
                taylor_score = taylor_fake_iteration(
                    args, args.output_dir, teacher_model, train_dataloader.dataset, highway_mode=True)
                prun_times += 1
                student_model, optimizer, scheduler, remain_steps = iterative_pruning(
                    args, student_model, teacher_model.config, scheduler,
                    prune_step, max_prune_times, prun_times, global_step, taylor_score=taylor_score)

            rep_loss = 0.
            hidn_loss = 0.
            attn_loss = 0.
            cls_loss = 0.

            student_model.train()
            if teacher_model is not None:
                teacher_model.eval()

            inputs["highway_mode"] = True
            inputs["exit_start"] = args.exit_start
            outputs = student_model(**inputs)
            hard_label_loss = outputs[0]
            student_logits = outputs[1]
            if args.train_costeff or args.train_ta:
                student_highway = outputs[-1]
            if args.output_hidden_states:
                student_hidns = outputs[3]  # all_hidden_states
                if args.output_attentions:
                    student_attns = outputs[4]  # all_attentions
            elif args.output_attentions:
                student_attns = outputs[3]  # all_attentions

            if args.pred_distill or args.repr_distill:
                with torch.no_grad():
                    if not args.train_costeff:
                        inputs.pop("highway_mode", None)
                        inputs.pop("exit_start", None)
                    teacher_outputs = teacher_model(**inputs)
                    teacher_logits = teacher_outputs[1]
                    if args.train_costeff:
                        teacher_highway = teacher_outputs[-2]
                    if args.output_hidden_states:
                        teacher_hidns = teacher_outputs[3]  # all_hidden_states
                        if args.output_attentions:
                            teacher_attns = teacher_outputs[4]  # all_attentions
                    elif args.output_attentions:
                        teacher_attns = teacher_outputs[3]  # all_attentions

            if prune_step > 0 and prun_times < max_prune_times:
                if (global_step+1) % prune_step < int(args.repr_proportion*prune_step):
                    args.repr_distill = True
                    args.pred_distill = False
                else:
                    args.repr_distill = temp_repr_distill
                    args.pred_distill = temp_pred_distill

            if args.repr_distill:
                teacher_layer_num = teacher_model.config.num_hidden_layers
                student_layer_num = student_model.config.num_hidden_layers
                prun_layer_ratio = student_layer_num/teacher_layer_num
                layers_per_block = int(teacher_layer_num / student_layer_num)

                try:
                    assert teacher_layer_num % student_layer_num == 0
                    if args.output_hidden_states:
                        new_teacher_hidns = [teacher_hidns[i * layers_per_block] for i in range(student_layer_num + 1)]
                    if args.output_attentions:
                        new_teacher_attns = [teacher_attns[i * layers_per_block] for i in range(student_layer_num)]
                except AssertionError:
                    # Use mod to drop layers if not teacher_layer_num % student_layer_num == 0
                    if args.output_hidden_states:
                        new_teacher_hidns = [
                            teacher_hidns[i] for i in range(teacher_layer_num + 1)
                            if not(i + 1) % (1 / prun_layer_ratio) < 1e-5]
                    if args.output_attentions:
                        new_teacher_attns = [
                            teacher_attns[i] for i in range(teacher_layer_num)
                            if not(i + 1) % (1 / prun_layer_ratio) < 1e-5]

                non_padding_masks = batch[1].float()
                # [batch_size, seq_len]
                batch_non_padding_len = non_padding_masks.sum(dim=-1)
                # [batch_size]
                if args.output_hidden_states:
                    for student_hidn, teacher_hidn in zip(student_hidns, new_teacher_hidns):
                        # [batch_size, seq_len, hidden_size]
                        non_padding_masks_hidn = non_padding_masks.unsqueeze(-1)
                        student_hidn = student_hidn.mul(non_padding_masks_hidn)
                        teacher_hidn = teacher_hidn.mul(non_padding_masks_hidn)
                        # HACK: manual batchmean
                        loss_fct = MSELoss(reduction="none")
                        tmp_loss = loss_fct(student_hidn, teacher_hidn)
                        #           sum   hidden_size   seq_len
                        batch_loss = tmp_loss.mean(dim=-1).sum(dim=-1)
                        # mean batch, manual mean seq_len
                        mean_loss = torch.mean(batch_loss/batch_non_padding_len)
                        hidn_loss += mean_loss
                if args.output_attentions:
                    for i, (student_attn, teacher_attn) in enumerate(zip(student_attns, new_teacher_attns)):
                        # [batch_size, head_num, seq_len, seq_len]
                        teacher_attn = teacher_attn.index_select(1, torch.tensor(
                            student_model.config.head_to_distill[i]).to(args.device))
                        non_padding_masks_attn = non_padding_masks.unsqueeze(
                            -1).matmul(non_padding_masks.unsqueeze(1)).unsqueeze(1)
                        student_attn = student_attn.mul(non_padding_masks_attn)
                        teacher_attn = teacher_attn.mul(non_padding_masks_attn)
                        # HACK: manual batchmean
                        loss_fct = MSELoss(reduction="none")
                        tmp_loss = loss_fct(student_attn, teacher_attn)
                        #            sum  seq_len2     seq_len1   mean head_num
                        batch_loss = tmp_loss.sum(dim=-1).mean(dim=-1).mean(dim=-1)
                        mean_loss = torch.mean(batch_loss/batch_non_padding_len)
                        attn_loss += mean_loss
                rep_loss = hidn_loss+attn_loss
                tr_rep_loss += rep_loss.item()

            if not args.pred_distill and not args.repr_distill:
                cls_loss = hard_label_loss
                tr_cls_loss += cls_loss.item()
            if args.pred_distill:
                if args.output_mode == "classification":
                    cls_loss = softLabelLoss(student_logits / args.temperature,
                                             teacher_logits / args.temperature)
                    if args.train_costeff:
                        for student_exit, teacher_exit in zip(student_highway, teacher_highway):
                            student_highway_logits = student_exit[0]
                            teacher_highway_logits = teacher_exit[0]
                            cls_loss += softLabelLoss(student_highway_logits / args.temperature,
                                                      teacher_highway_logits / args.temperature)
                    elif args.train_ta:
                        for student_exit in student_highway:
                            student_highway_logits = student_exit[0]
                            cls_loss += softLabelLoss(student_highway_logits / args.temperature,
                                                      teacher_logits / args.temperature)
                        cls_loss = (cls_loss+hard_label_loss)/2
                elif args.output_mode == "regression":
                    loss_fct = MSELoss()
                    cls_loss = loss_fct(student_logits.view(-1), teacher_logits.view(-1))
                tr_cls_loss += cls_loss.item()

            loss = cls_loss + rep_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                student_model.zero_grad()
                global_step += 1

            if (global_step) % args.eval_step == 0 or \
                    global_step == args.max_optimization_steps:
                logger.info("***** Running evaluation *****")
                logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                logger.info("  Num examples = %d", len(eval_dataloader.dataset))
                logger.info("  Batch size = %d", args.eval_batch_size)
                lr = scheduler.get_last_lr()[0]
                logger.info("  Learning rate = %.4e", lr)

                for group in optimizer.param_groups:
                    for p in group["params"]:
                        optim_step = optimizer.state[p]["step"]
                        break
                    break

                loss = tr_loss / (step + 1)
                cls_loss = tr_cls_loss / (step + 1)
                rep_loss = tr_rep_loss / (step + 1)

                result, layer_avg = do_eval_layer(args, model=student_model, eval_dataloader=eval_dataloader)
                result["global_step"] = global_step
                result["cls_loss"] = cls_loss
                result["rep_loss"] = rep_loss
                result["loss"] = loss
                result["lr"] = lr
                result["optim_step"] = optim_step
                if args.tb_writer is not None:
                    for k, v in result.items():
                        args.tb_writer.add_scalar(k, v, global_step)

                result_to_file(result, args.task_name, output_eval_file)

                if args.depth_or_width is not None:
                    save_model = True  # always save when pruning
                elif args.pred_distill or (not (args.pred_distill or args.repr_distill)):
                    save_model = False
                    if layer_avg > best_dev_score:
                        best_dev_score = layer_avg
                        save_model = True  # save best when pred_distill
                else:
                    save_model = True
                if save_model:
                    model_to_save = student_model.module if hasattr(student_model, "module") else student_model

                    logger.info("  Saving model to %s after evaluation..." % args.output_dir)
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)


# 2-stage training, for fine-tuning
def do_train_2t(args, train_dataloader, eval_dataloader, student_model, teacher_model, tokenizer, highway_mode=False):
    logger.info("  This is 2 stage training")
    if args.n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
        if teacher_model is not None:
            teacher_model = torch.nn.DataParallel(teacher_model)

    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    for name, param in student_model.named_parameters():
        if "highway" in name:
            param.requires_grad = highway_mode
        elif "classifier" in name or "pooler" in name:
            param.requires_grad = True
        else:
            param.requires_grad = (not highway_mode)
    optimizer_grouped_parameters = [
        {"params": [p for n, p in student_model.named_parameters() if
                    (p.requires_grad) and (not any(nd in n for nd in no_decay))],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in student_model.named_parameters() if
                    (p.requires_grad) and (any(nd in n for nd in no_decay))],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.depth_or_width is not None:
        max_prune_times, prune_step = init_prun(
            args, teacher_model.config.num_hidden_layers, teacher_model.config.num_attention_heads,
            len(train_dataloader.dataset))
        prun_times = 0
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(prune_step*args.warmup_proportion),
            num_training_steps=args.max_optimization_steps)
    else:
        # save model if prun_times < max_prune_times
        prun_times, max_prune_times, prune_step = -1, -1, -1
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(args.max_optimization_steps*args.warmup_proportion),
            num_training_steps=args.max_optimization_steps)

    # Train and evaluate
    global_step = 0
    best_dev_score = 0.0
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    student_model.zero_grad()
    for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0.
        tr_rep_loss = 0.
        tr_cls_loss = 0.

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(args.device) for t in batch)
            if batch[0].size()[0] != args.train_batch_size:
                continue

            inputs = {"input_ids":      batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels":         batch[3]}

            if args.depth_or_width is not None \
                    and prune_step > 0 and (global_step+1) % prune_step == 0 \
                    and prun_times < max_prune_times and not highway_mode:
                # Pruning
                model_to_save = student_model.module if hasattr(student_model, "module") else student_model
                output_dir = os.path.join(args.output_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                logger.info("  Saving model to %s before pruning..." % output_dir)
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info("  Compute taylor score before pruning\n")
                taylor_score = taylor_fake_iteration(
                    args, args.output_dir, teacher_model, train_dataloader.dataset, highway_mode=True)
                prun_times += 1
                student_model, optimizer, scheduler, remain_steps = iterative_pruning(
                    args, student_model, teacher_model.config, scheduler,
                    prune_step, max_prune_times, prun_times, global_step, taylor_score=taylor_score)

            rep_loss = 0.
            hidn_loss = 0.
            attn_loss = 0.
            cls_loss = 0.

            student_model.train()
            if teacher_model is not None:
                teacher_model.eval()
            if not args.train_ft:
                inputs["highway_mode"] = highway_mode
            outputs = student_model(**inputs)
            hard_label_loss = outputs[0]
            student_logits = outputs[1]
            if highway_mode:
                student_highway = outputs[-1]
            if args.output_hidden_states:
                student_hidns = outputs[3]  # all_hidden_states
                if args.output_attentions:
                    student_attns = outputs[4]  # all_attentions
            elif args.output_attentions:
                student_attns = outputs[3]  # all_attentions

            if (args.pred_distill or args.repr_distill) and teacher_model is not None:
                teacher_model.eval()
                with torch.no_grad():
                    if not args.train_costeff:
                        inputs.pop("highway_mode", None)
                    teacher_outputs = teacher_model(**inputs)
                    teacher_logits = teacher_outputs[1]
                    if args.train_costeff:
                        teacher_highway = teacher_outputs[-2]
                    if args.output_hidden_states:
                        teacher_hidns = teacher_outputs[3]  # all_hidden_states
                        if args.output_attentions:
                            teacher_attns = teacher_outputs[4]  # all_attentions
                    elif args.output_attentions:
                        teacher_attns = teacher_outputs[3]  # all_attentions

            if not args.pred_distill and not args.repr_distill:
                cls_loss = hard_label_loss
                tr_cls_loss += cls_loss.item()
            if args.pred_distill:
                if args.output_mode == "classification":
                    cls_loss = softLabelLoss(student_logits / args.temperature,
                                             teacher_logits / args.temperature)
                    if highway_mode:
                        if args.train_costeff:
                            for student_exit, teacher_exit in zip(student_highway, teacher_highway):
                                student_highway_logits = student_exit[0]
                                teacher_highway_logits = teacher_exit[0]
                                cls_loss += softLabelLoss(student_highway_logits / args.temperature,
                                                          teacher_highway_logits / args.temperature)
                        elif args.train_ta:
                            for student_exit in student_highway:
                                student_highway_logits = student_exit[0]
                                cls_loss += softLabelLoss(student_highway_logits / args.temperature,
                                                          teacher_logits / args.temperature)
                            cls_loss = (cls_loss+hard_label_loss)/2
                    else:
                        cls_loss = (cls_loss+hard_label_loss)/2
                elif args.output_mode == "regression":
                    loss_fct = MSELoss()
                    cls_loss = loss_fct(student_logits.view(-1), teacher_logits.view(-1))
                tr_cls_loss += cls_loss.item()

            if args.repr_distill:
                teacher_layer_num = teacher_model.config.num_hidden_layers
                student_layer_num = student_model.config.num_hidden_layers
                prun_layer_ratio = student_layer_num/teacher_layer_num
                layers_per_block = int(teacher_layer_num / student_layer_num)

                try:
                    assert teacher_layer_num % student_layer_num == 0
                    if args.output_hidden_states:
                        new_teacher_hidns = [teacher_hidns[i * layers_per_block] for i in range(student_layer_num + 1)]
                    if args.output_attentions:
                        new_teacher_attns = [teacher_attns[i * layers_per_block] for i in range(student_layer_num)]
                except AssertionError:
                    # Use mod to drop layers if not teacher_layer_num % student_layer_num == 0
                    if args.output_hidden_states:
                        new_teacher_hidns = [
                            teacher_hidns[i] for i in range(teacher_layer_num + 1)
                            if not(i + 1) % (1 / prun_layer_ratio) < 1e-5]
                    if args.output_attentions:
                        new_teacher_attns = [
                            teacher_attns[i] for i in range(teacher_layer_num)
                            if not(i + 1) % (1 / prun_layer_ratio) < 1e-5]

                non_padding_masks = batch[1].float()
                # [batch_size, seq_len]
                batch_non_padding_len = non_padding_masks.sum(dim=-1)
                # [batch_size]
                if args.output_hidden_states:
                    for student_hidn, teacher_hidn in zip(student_hidns, new_teacher_hidns):
                        # [batch_size, seq_len, hidden_size]
                        non_padding_masks_hidn = non_padding_masks.unsqueeze(-1)
                        student_hidn = student_hidn.mul(non_padding_masks_hidn)
                        teacher_hidn = teacher_hidn.mul(non_padding_masks_hidn)
                        # HACK: manual batchmean
                        loss_fct = MSELoss(reduction="none")
                        tmp_loss = loss_fct(student_hidn, teacher_hidn)
                        #           sum   hidden_size   seq_len
                        batch_loss = tmp_loss.mean(dim=-1).sum(dim=-1)
                        # mean batch, manual mean seq_len
                        mean_loss = torch.mean(batch_loss/batch_non_padding_len)
                        hidn_loss += mean_loss
                if args.output_attentions:
                    for i, (student_attn, teacher_attn) in enumerate(zip(student_attns, new_teacher_attns)):
                        # [batch_size, head_num, seq_len, seq_len]
                        teacher_attn = teacher_attn.index_select(1, torch.tensor(
                            student_model.config.head_to_distill[i]).to(args.device))
                        non_padding_masks_attn = non_padding_masks.unsqueeze(
                            -1).matmul(non_padding_masks.unsqueeze(1)).unsqueeze(1)
                        student_attn = student_attn.mul(non_padding_masks_attn)
                        teacher_attn = teacher_attn.mul(non_padding_masks_attn)
                        # HACK: manual batchmean
                        loss_fct = MSELoss(reduction="none")
                        tmp_loss = loss_fct(student_attn, teacher_attn)
                        #            sum  seq_len2     seq_len1   mean head_num
                        batch_loss = tmp_loss.sum(dim=-1).mean(dim=-1).mean(dim=-1)
                        mean_loss = torch.mean(batch_loss/batch_non_padding_len)
                        attn_loss += mean_loss
                rep_loss = hidn_loss+attn_loss
                tr_rep_loss += rep_loss.item()

            loss = cls_loss + rep_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                student_model.zero_grad()
                global_step += 1

            if (global_step) % args.eval_step == 0 or \
                    global_step == args.max_optimization_steps:
                logger.info("***** Running evaluation *****")
                logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                logger.info("  Num examples = %d", len(eval_dataloader.dataset))
                logger.info("  Batch size = %d", args.eval_batch_size)
                lr = scheduler.get_last_lr()[0]
                logger.info("  Learning rate = %.4e", lr)

                for group in optimizer.param_groups:
                    for p in group["params"]:
                        optim_step = optimizer.state[p]["step"]
                        break
                    break

                loss = tr_loss / (step + 1)
                cls_loss = tr_cls_loss / (step + 1)
                rep_loss = tr_rep_loss / (step + 1)

                if highway_mode:
                    result, layer_avg = do_eval_layer(args, model=student_model, eval_dataloader=eval_dataloader)
                    curr_score = layer_avg
                else:
                    result = do_eval(args, model=student_model,
                                     eval_dataloader=eval_dataloader, highway_mode=False)
                    if args.task_name in ["mnli", "mnli-mm"]:
                        curr_score = result[args.task_name+"/acc"]
                    elif args.task_name in acc_tasks:
                        curr_score = result["acc"]
                    elif args.task_name in f1_tasks:
                        curr_score = result["f1"]
                    elif args.task_name in corr_tasks:
                        curr_score = result["corr"]
                    elif args.task_name in mcc_tasks:
                        curr_score = result["mcc"]
                result["global_step"] = global_step
                result["cls_loss"] = cls_loss
                result["rep_loss"] = rep_loss
                result["loss"] = loss
                result["lr"] = lr
                result["optim_step"] = optim_step
                if args.tb_writer is not None:
                    for k, v in result.items():
                        args.tb_writer.add_scalar(k, v, global_step)

                result_to_file(result, args.task_name, output_eval_file)

                if args.depth_or_width is not None:
                    save_model = True  # always save when pruning
                elif args.train_ft or args.pred_distill:
                    save_model = False
                    if curr_score > best_dev_score:
                        best_dev_score = curr_score
                        save_model = True  # save best when pred_distill
                else:
                    save_model = True
                if save_model:
                    model_to_save = student_model.module if hasattr(student_model, "module") else student_model

                    logger.info("  Saving model to %s after evaluation..." % args.output_dir)
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)


def load_model(args, model_name_or_path):
    logger.info("  Loading Model from %s..." % model_name_or_path)
    config = CosteffConfig.from_pretrained(model_name_or_path,
                                           num_labels=args.num_labels,
                                           finetuning_task=args.task_name,
                                           output_attentions=args.output_attentions,
                                           output_hidden_states=args.output_hidden_states)

    if args.pooler_dense:
        config.pooler_dense = True
    else:
        config.hidden_mode = args.hidden_mode
        config.pooling_mode = args.pooling_mode
        config.last_n_hidden = args.last_n_hidden
    model = CosteffForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    model.bert.encoder.set_early_exit_entropy(args.early_exit_entropy)
    if args.pooler_dense and args.train_ta:
        model.bert.init_highway_pooler()
    model.to(args.device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The teacher model dir.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        help="The student model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--overwrite_output_dir",
                        action="store_true",
                        help="Overwrite the content of the output directory.\n"
                             "Use this to continue training if output_dir points to a checkpoint directory.")
    parser.add_argument("--config_dir",
                        default=None,
                        type=str,
                        help="The directory of configuration file.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--lr_schedule",
                        default="none",
                        type=str,
                        help="Learning rate schedule.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_layer",
                        action="store_true",
                        help="Whether to eval each layer on the dev set.")
    parser.add_argument("--do_prof",
                        action="store_true",
                        help="Whether to run profiler.")
    parser.add_argument("--do_test",
                        action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--patience",
                        default=None,
                        type=int,
                        help="The max number of epoch without improvement. Used to perform early stop.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=1e-2,
                        type=float,
                        metavar="W",
                        help="weight decay")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_attentions",
                        action="store_true",
                        help="Set this flag if output all attentions.")
    parser.add_argument("--output_hidden_states",
                        action="store_true",
                        help="Set this flag if output all hidden states.")
    parser.add_argument("--prun_period_proportion",
                        type=float,
                        default=0.,
                        help="The proportion of pruning steps.")
    parser.add_argument("--keep_heads",
                        type=int,
                        default=None,
                        help="Number of attention heads to keep.")
    parser.add_argument("--ffn_hidden_dim",
                        type=int,
                        default=None,
                        help="Number of FFN hhidden dimension to keep.")
    parser.add_argument("--emb_hidden_dim",
                        type=int,
                        default=None,
                        help="Number of embedding hidden dim to keep.")
    parser.add_argument("--keep_layers",
                        type=int,
                        default=None,
                        help="Number of layers to keep.")
    parser.add_argument("--depth_or_width",
                        type=str,
                        default=None,
                        help="Compress model depth or width. None means no compression.")
    parser.add_argument("--eval_step",
                        type=int,
                        default=500,
                        help="Evaluate every `eval_step` training steps.")
    parser.add_argument("--pred_distill",
                        action="store_true",
                        help="Set this flag if use prediction distillation.")
    parser.add_argument("--repr_distill",
                        action="store_true",
                        help="Set this flag if use representation distillation.")
    parser.add_argument("--temperature",
                        type=float,
                        default=1.0,
                        help="Scaling factor of distillation.")
    parser.add_argument("--train_costeff",
                        action="store_true",
                        help="Set this flag if Student: COST-EFF, Teacher: TA ")
    parser.add_argument("--train_ta",
                        action="store_true",
                        help="Set this flag if Student: TA, Teacher: finetuned model ")
    parser.add_argument("--train_ft",
                        action="store_true",
                        help="Set this flag if Student: bert-base-uncased, Teacher: None ")
    parser.add_argument("--train_1t",
                        action="store_true",
                        help="Set this flag if Joint training.")
    parser.add_argument("--train_2t",
                        action="store_true",
                        help="Set this flag if 2-stage training.")
    parser.add_argument("--highway_mode",
                        action="store_true",
                        help="Set this flag if it's using highway")
    parser.add_argument("--early_exit_entropy",
                        type=float,
                        default=-1,
                        help="Entropy threshold for early exit.")
    parser.add_argument("--hidden_mode",
                        type=str,
                        default="concat",
                        help="How to use hidden layers. Options: `avg`, `concat`.")
    parser.add_argument("--pooling_mode",
                        type=str,
                        default="cls",
                        help="How to pool hidden states. Options: `avg`, `cls`.")
    parser.add_argument("--last_n_hidden",
                        type=int,
                        default=1,
                        help="How many hidden layers to concat or average. If value is 1, use the last hidden state.")
    parser.add_argument("--max_profile_samples",
                        type=int,
                        default=2000,
                        help="How many samples to profile (profiling is very slow)?")
    parser.add_argument("--taylor_proportion",
                        type=float,
                        default=1.0,
                        help="How many samples to compute taylor score.")
    parser.add_argument("--internal_loss",
                        action="store_true",
                        help="Set this flag if exit loss is used in compression.")
    parser.add_argument("--lr_restore_factor",
                        type=float,
                        default=0,
                        help="Factor of enlarging learning rate during iterative pruning.")
    parser.add_argument("--repr_proportion",
                        type=float,
                        default=0.7,
                        help="Repr-only distill propotion in iterative pruning.")
    parser.add_argument("--pooler_dense",
                        action="store_true",
                        help="Whether to use pooler with a Linear.")
    parser.add_argument("--exit_start",
                        type=int,
                        default=1,
                        help="Early exit starting layer.")
    args = parser.parse_args()
    logger.info("The args: {}".format(args))

    # Prepare devices
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(args.device, args.n_gpu))

    set_seed(args)

    args.task_name = args.task_name.lower()
    if not (args.do_eval or args.do_test):
        if (args.config_dir is not None) and (os.path.exists(args.config_dir)):
            config_file = open(args.config_dir, "r")
            config = json.load(config_file)
            config_file.close()
            for key in config[args.task_name]:
                vars(args)[key] = config[args.task_name][key]
        args.logging_dir = os.path.join(args.output_dir, "logging")

    # Prepare task settings
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if "logging_dir" in args:
        if not os.path.exists(args.logging_dir):
            os.makedirs(args.logging_dir)
        if is_tensorboard_available():
            args.tb_writer = SummaryWriter(log_dir=args.logging_dir)
        else:
            args.tb_writer = None

    if not (args.do_eval or args.do_test):
        fw_args = open(os.path.join(args.output_dir, "args.txt"), "w")
        fw_args.write(str(args))
        fw_args.close()

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained("./models/pretrained_model/bert-base-uncased",
                                              do_lower_case=args.do_lower_case)

    if not (args.do_eval or args.do_test):
        if args.train_ft:
            teacher_model = None
        elif args.train_ta:
            logger.info("  Loading Teacher from %s..." % args.teacher_model)
            teacher_config = BertConfig.from_pretrained(args.teacher_model,
                                                        num_labels=args.num_labels,
                                                        finetuning_task=args.task_name,
                                                        output_attentions=args.output_attentions,
                                                        output_hidden_states=args.output_hidden_states)
            teacher_model = TeacherBertForSequenceClassification.from_pretrained(
                args.teacher_model, config=teacher_config)
            teacher_model.to(args.device)
        else:
            if args.teacher_model is not None:
                teacher_model = load_model(args, args.teacher_model)
            else:
                teacher_model = None

    if args.train_ft:
        logger.info("  Loading Student from %s..." % args.student_model)
        student_config = BertConfig.from_pretrained(args.student_model,
                                                    num_labels=args.num_labels,
                                                    finetuning_task=args.task_name,
                                                    output_attentions=args.output_attentions,
                                                    output_hidden_states=args.output_hidden_states)
        student_model = TeacherBertForSequenceClassification.from_pretrained(
            args.student_model, config=student_config)
        student_model.to(args.device)
    else:
        student_model = load_model(args, args.student_model)

    if args.do_test:
        test_dataloader, _, test_data = build_dataloader(
            "test", args, processor, label_list, tokenizer, args.output_mode)
    else:
        if not args.do_eval:
            if args.gradient_accumulation_steps < 1:
                raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                    args.gradient_accumulation_steps))
            args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

            train_dataloader, train_labels, train_data = build_dataloader(
                "train", args, processor, label_list, tokenizer, args.output_mode)
            args.max_optimization_steps = int(
                len(train_data) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        eval_dataloader, eval_labels, eval_data = build_dataloader(
            "eval", args, processor, label_list, tokenizer, args.output_mode)

    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_data))
        logger.info("  Batch size = %d", args.eval_batch_size)

        if args.do_prof:
            result = do_eval_prof(args, student_model, eval_dataloader, highway_mode=args.highway_mode)
        elif args.eval_layer:
            result, layer_avg = do_eval_layer(args, student_model, eval_dataloader)
        else:
            result = do_eval(args, student_model, eval_dataloader, highway_mode=args.highway_mode)

        output_eval_file = os.path.join(args.output_dir, "pure_eval_results.txt")
        result_to_file(result, args.task_name, output_eval_file)
        if args.task_name == "mnli":
            args.task_name = "mnli-mm"
            processor = processors[args.task_name]()
            args.output_mode = output_modes[args.task_name]
            label_list = processor.get_labels()
            args.num_labels = len(label_list)
            eval_dataloader, eval_labels, eval_data = build_dataloader(
                "eval", args, processor, label_list, tokenizer, args.output_mode)
            if args.do_prof:
                result = do_eval_prof(args, student_model, eval_dataloader, highway_mode=args.highway_mode)
            elif args.eval_layer:
                result, layer_avg = do_eval_layer(args, student_model, eval_dataloader)
            else:
                result = do_eval(args, student_model, eval_dataloader, highway_mode=args.highway_mode)
            result_to_file(result, args.task_name, output_eval_file)
    elif args.do_test:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_data))
        logger.info("  Batch size = %d", args.eval_batch_size)

        if args.do_prof:
            preds = do_test_prof(args, student_model, test_dataloader, highway_mode=args.highway_mode)
        else:
            preds = do_test(args, student_model, test_dataloader, highway_mode=args.highway_mode)
        output_test_file = os.path.join("./test", "%s-%.2f.tsv" % (args.task_name.upper(), args.early_exit_entropy))
        output_test_result(args, preds, label_list, output_test_file)
        if args.task_name == "mnli":
            args.task_name = "mnli-mm"
            processor = processors[args.task_name]()
            args.output_mode = output_modes[args.task_name]
            label_list = processor.get_labels()
            args.num_labels = len(label_list)
            test_dataloader, _, test_data = build_dataloader(
                "test", args, processor, label_list, tokenizer, args.output_mode)
            if args.do_prof:
                preds = do_test_prof(args, student_model, test_dataloader, highway_mode=args.highway_mode)
            else:
                preds = do_test(args, student_model, test_dataloader)
            output_test_file = os.path.join("./test", args.task_name.upper()+"-"+args.early_exit_entropy+".tsv")
            output_test_result(args, preds, label_list, output_test_file)
    else:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.max_optimization_steps)

        if args.train_costeff:
            if args.train_1t:
                do_train_1t(args, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                            student_model=student_model, teacher_model=teacher_model, tokenizer=tokenizer)
            elif args.train_2t:
                do_train_2t(args, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                            student_model=student_model, teacher_model=teacher_model, tokenizer=tokenizer,
                            highway_mode=args.highway_mode)
        elif args.train_ta:
            if args.train_1t:
                do_train_1t(args, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                            student_model=student_model, teacher_model=teacher_model, tokenizer=tokenizer)
            elif args.train_2t:
                do_train_2t(args, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                            student_model=student_model, teacher_model=teacher_model, tokenizer=tokenizer,
                            highway_mode=args.highway_mode)
        elif args.train_ft:
            do_train_2t(args, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                        student_model=student_model, teacher_model=teacher_model, tokenizer=tokenizer,
                        highway_mode=False)
        else:
            if args.train_1t:
                do_train_1t(args, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                            student_model=student_model, teacher_model=teacher_model, tokenizer=tokenizer)
            elif args.train_2t:
                do_train_2t(args, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                            student_model=student_model, teacher_model=teacher_model, tokenizer=tokenizer,
                            highway_mode=args.highway_mode)

        if "tb_writer" in dir() and args.tb_writer is not None:
            args.tb_writer.close()


if __name__ == "__main__":
    main()
