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
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


logger = logging.getLogger(__name__)


class VideoPhraseDataset(Dataset):
    def __init__(self, text_feature_dataset, video_feature_files):
        self.text_feature_dataset = text_feature_dataset
        self.video_feature_files = video_feature_files

    def __len__(self):
        return len(self.video_feature_files)

    def _prepare_video_features(self, features):
        if features.shape[0] >= 40:
            features = features[:40, :]
            mask = [1] * 40
        else:
            pad_length = 40 - features.shape[0]
            dim = features.shape[1]
            mask = [1] * features.shape[0] + [0] * pad_length
            features = np.concatenate((features, np.zeros((pad_length, dim))), axis=0)
        return features, mask

    def __getitem__(self, idx):
        text_features = self.text_feature_dataset[idx]
        video_feature, video_mask = self._prepare_video_features(np.load(self.video_feature_files[idx]))
        video_feature = torch.tensor(video_feature, dtype=torch.float)
        video_mask = torch.tensor(video_mask, dtype=torch.long)
        # temp = text_features + (video_feature, video_mask)
        return text_features + (video_feature, video_mask,)


FEAT_DIR = '/home/code-base/scratch/data/video_features'
class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, label, video_feature_file):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.label = label
        self.video_feature_file = video_feature_file


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_ids, video_feature_file):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids
        self.video_feature_file = video_feature_file


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    guid_index = 1
    examples = []

    with open(file_path, 'r') as f:
        data = json.load(f)
        print("%s samples in file" % len(data))
        for item in data:
            words = item["str_words"]
            labels = item["tags"]
            if not any(labels):
                class_label = 0
            else:
                class_label = 1

            timestamp = float(item["timestamp"])
            target_name_1 = os.path.join(FEAT_DIR, item["video_id"].replace("/", "__").replace(".mp4",
                                                                                               ".%s.cut.mp4" % int(
                                                                                                   timestamp)))[
                            :-4] + '.npy'
            target_name_2 = os.path.join(FEAT_DIR, item["video_id"].replace("/", "__") + "%s.cut.mp4" % int(timestamp))[
                            :-4] + '.npy'
            target_name_3 = os.path.join(FEAT_DIR, item["video_id"].replace("/", "__").replace(".mp4",
                                                                                               ".%s.cut.%s.mp4" % (
                                                                                               int(timestamp), 10)))[
                            :-4] + '.npy'
            if os.path.exists(target_name_1):
                video_feat_file = target_name_1
            elif os.path.exists(target_name_2):
                video_feat_file = target_name_2
            elif os.path.exists(target_name_3):
                video_feat_file = target_name_3
            else:
                continue

            if "spans" not in item:
                continue

            for span in item["spans"]:
                examples.append(
                    InputExample(guid="%s-%d".format(mode, guid_index),
                                 words=words[:span[0]] + ['<phrase>'] + words[span[0]:span[1]] + ['</phrase>'] + words[span[1]:],
                                 label=class_label,
                                 video_feature_file=video_feat_file))
                guid_index += 1

    print("Read %s examples from file" % guid_index)
    return examples


def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        show_exnum=-1,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    extra_long_samples = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        for word in example.words:
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            extra_long_samples += 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]

        if cls_token_at_end:
            tokens += [cls_token]

        else:
            tokens = [cls_token] + tokens

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        if ex_index < show_exnum:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          label_ids=example.label,
                          video_feature_file=example.video_feature_file)
        )

    logger.info("Extra long example %d of %d", extra_long_samples, len(examples))
    return features


def load_and_cache_examples(args, tokenizer, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_binary_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_ids = torch.tensor([f for f in range(len(features))], dtype=torch.long)
    video_feature_files = [f.video_feature_file for f in features]

    text_feature_dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_ids)
    video_ner_dataset = VideoPhraseDataset(text_feature_dataset, video_feature_files)

    return video_ner_dataset


if __name__ == '__main__':
    save(args)