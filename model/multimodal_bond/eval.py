# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model.multimodal_bond.data_utils import load_and_cache_examples, tag_to_id, get_chunks

logger = logging.getLogger(__name__)


def evaluate(args, model, tokenizer, labels, pad_token_label_id, best, mode, prefix="", verbose=True):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation %s %s *****", prefix, mode)
    if verbose:
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "video_embedding": batch[-2]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    out_id_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_id_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                preds_list[i].append(label_map[preds[i][j]])
                out_id_list[i].append(out_label_ids[i][j])
                preds_id_list[i].append(preds[i][j])

    correct_preds, correct_pred_pm, total_correct, total_preds = 0., 0., 0., 0.  # i variables
    stats_by_class = defaultdict(lambda: defaultdict(lambda: 0.))
    for ground_truth_id, predicted_id in zip(out_id_list, preds_id_list):
        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks = set(get_chunks(ground_truth_id, tag_to_id(args.data_dir)))
        lab_pred_chunks = set(get_chunks(predicted_id, tag_to_id(args.data_dir)))

        # Updating the i variables
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

        # getting partial counts by class
        for chunk_type, _, _ in lab_chunks:
            stats_by_class[chunk_type]["total_correct"] += 1
        for chunk_type, _, _ in lab_pred_chunks:
            stats_by_class[chunk_type]["total_preds"] += 1
        for chunk_type in stats_by_class.keys():
            stats_by_class[chunk_type]["correct_preds"] += len(set([c for c in lab_chunks if c[0] == chunk_type]) &
                                                               set([c for c in lab_pred_chunks if c[0] == chunk_type]))

        if correct_preds<total_correct:
            for pred_chunk_type, pred_chunk_start, pred_chunk_end in lab_pred_chunks:
                for gt_chunk_type, gt_chunk_start, gt_chunk_end in lab_chunks:
                    len_seq = gt_chunk_end-gt_chunk_start
                    if pred_chunk_type == gt_chunk_type and (min(pred_chunk_end, gt_chunk_end) - max(pred_chunk_start, gt_chunk_start) >= 0.75*len_seq):
                        correct_pred_pm += 1
                        stats_by_class[pred_chunk_type]["correct_pred_pm"] += 1
        else:
            correct_pred_pm += len(lab_chunks & lab_pred_chunks)
            for chunk_type in stats_by_class.keys():
                stats_by_class[chunk_type]["correct_pred_pm"] += len(set([c for c in lab_chunks if c[0] == chunk_type]) &
                                                               set([c for c in lab_pred_chunks if c[0] == chunk_type]))

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    new_F = 2 * p * r / (p + r) if correct_preds > 0 else 0

    p_pm = correct_pred_pm / total_preds if correct_pred_pm > 0 else 0
    r_pm = correct_pred_pm / total_correct if correct_pred_pm > 0 else 0
    new_F_pm = 2 * p_pm * r_pm / (p_pm + r_pm) if correct_pred_pm > 0 else 0

    is_updated = False
    if new_F > best[-1]:
        best = [p, r, new_F]
        is_updated = True

    results = {
        "loss": eval_loss,
        "em_p/r/f1": ' / '.join([str(round(n, 2)) for n in [p, r, new_F]]),
        "best_em_p/r/f1": ' / '.join([str(round(n, 2)) for n in [best[0], best[1], best[-1]]]),
        "pm_p/r/f1": ' / '.join([str(round(n, 2)) for n in [p_pm, r_pm, new_F_pm]]),
    }

    if len(stats_by_class) > 1:
        for chunk_type in stats_by_class.keys():
            p = stats_by_class[chunk_type]["correct_preds"] / stats_by_class[chunk_type]["total_preds"] \
                if stats_by_class[chunk_type]["correct_preds"] > 0 else 0
            r = stats_by_class[chunk_type]["correct_preds"] / stats_by_class[chunk_type]["total_correct"] \
                if stats_by_class[chunk_type]["correct_preds"] > 0 else 0
            new_F = 2 * p * r / (p + r) if stats_by_class[chunk_type]["correct_preds"] > 0 else 0

            p_pm = stats_by_class[chunk_type]["correct_pred_pm"] / stats_by_class[chunk_type]["total_preds"] if stats_by_class[chunk_type]["correct_preds"] > 0 else 0
            r_pm = stats_by_class[chunk_type]["correct_pred_pm"] / stats_by_class[chunk_type]["total_correct"] if stats_by_class[chunk_type]["correct_preds"] > 0 else 0
            new_F_pm = 2 * p_pm * r_pm / (p_pm + r_pm) if stats_by_class[chunk_type]["correct_preds"] > 0 else 0

            results[chunk_type + '_em_p/r/f1'] = ' / '.join([str(round(n, 2)) for n in [p, r, new_F]])
            results[chunk_type + '_pm_p/r/f1'] = ' / '.join([str(round(n, 2)) for n in [p_pm, r_pm, new_F_pm]])

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list, best, is_updated