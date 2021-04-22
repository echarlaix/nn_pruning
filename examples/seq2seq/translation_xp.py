# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for text translation.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from datasets import load_metric, load_dataset
import json
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    default_data_collator,
)

from nn_pruning.hp_naming import TrialShortNamer
from examples.xp import XP, DataTrainingArguments, ModelArguments, XPTrainingArguments
from .seq2seq_sparse_train import Seq2SeqXPTrainingArguments
from .translation_train import TranslationTrainer

logger = logging.getLogger(__name__)

@dataclass
class TranslationDataTrainingArguments(DataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    source_lang: str = field(
        default=None,
        metadata={"help": "Source language id for translation."}
    )

    target_lang: str = field(
        default=None,
        metadata={"help": "Target language id for translation."}
    )

    max_source_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    max_train_samples: int = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: int = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: int = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    num_beams: int = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: str = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: str = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
                    "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
                    "needs to be the target language token.(Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension == "json", "`train_file` should be a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension == "json", "`validation_file` should be a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


class TranslationXP(XP):
    ARGUMENTS = {
        "model": ModelArguments,
        "data": TranslationDataTrainingArguments,
        "training": Seq2SeqXPTrainingArguments,
    }
    TRANSLATION_TRAINER_CLASS = TranslationTrainer
    SHORT_NAMER = TrialShortNamer

    @classmethod
    def _model_init(self, model_args, model_config):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=model_config,
            cache_dir=model_args.cache_dir,
        )
        return model

    def _preprocess_function(self, examples):
        data_args = self.data_args
        inputs = [ex[self.source_lang] for ex in examples["translation"]]
        targets = [ex[self.target_lang] for ex in examples["translation"]]
        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=data_args.max_source_length, padding=self.padding, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=data_args.max_target_length, padding=self.padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def prepare_column_names(self):
        training_args = self.training_args
        if training_args.do_train:
            column_names = self.datasets["train"].column_names
        else:
            column_names = self.datasets["validation"].column_names
        self.column_names = column_names

    def prepare_datasets(self):
        data_args = self.data_args
        dataset_cache_dir = Path(data_args.dataset_cache_dir).resolve()

        if not dataset_cache_dir.exists():
            dataset_cache_dir.mkdir()

        if self.training_args.do_train:
            train_dataset = self.datasets["train"]
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))
            self.train_dataset = train_dataset.map(
                self._preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_name=str(dataset_cache_dir / "train_dataset"),
            )

        if self.training_args.do_eval:
            validation_dataset = self.datasets["validation"]
            if data_args.max_val_samples is not None:
                validation_dataset = validation_dataset.select(range(data_args.max_val_samples))
            self.validation_dataset = validation_dataset.map(
                self._preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_name=str(dataset_cache_dir / "eval_dataset"),
            )

    def create_dataset(self):
        # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
        # source and target languages (unless you adapt what follows).
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        data_args = self.data_args
        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            self.datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        else:
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
            extension = data_args.train_file.split(".")[-1]
            self.datasets = load_dataset(extension, data_files=data_files)

        self.prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

        self.prepare_column_names()

        # Get the language codes for input/target.
        self.source_lang = data_args.source_lang.split("_")[0]
        self.target_lang = data_args.target_lang.split("_")[0]

        self.padding = "max_length" if data_args.pad_to_max_length else False
        self.prepare_datasets()

    def create_trainer(self):
        training_args = self.training_args
        data_args = self.data_args

        # Data collator
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        else:
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                label_pad_token_id=label_pad_token_id,
            )

        # Metric
        metric = load_metric("sacrebleu")

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [[label.strip()] for label in labels]

            return preds, labels

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            labels = p.label_ids
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": result["score"]}

            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result

        all_args = self.get_all_args(exclude_base=True)

        # Initialize our Trainer
        self.trainer = self.TRANSLATION_TRAINER_CLASS(
            model=None,
            args=training_args,
            train_dataset=self.train_dataset if training_args.do_train else None,
            eval_dataset=self.validation_dataset if training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            model_init=self.model_init,
            **all_args,
        )
        self.trainer._max_length = data_args.max_target_length
        self.trainer._num_beams = data_args.num_beams

    @classmethod
    def evaluate_model(cls, model_name_or_path, task, optimize_mode="dense", output_dir = None):
        if output_dir is None:
            output_dir = Path(model_name_or_path)
        else:
            output_dir = Path(output_dir)
        output_dir = output_dir.resolve()

        parameters = {
            "model_name_or_path": model_name_or_path,
            "dataset_name": task,
            "dataset_cache_dir": "dataset_cache_dir",
            "do_train": 0,
            "do_eval": 1,
            "per_device_eval_batch_size": 128,
            "max_seq_length": 128,
            "doc_stride": 128,
            "output_dir": str(output_dir),
            "logging_dir": str(output_dir),
            "overwrite_cache": 0,
            "overwrite_output_dir": 0,
            "optimize_model_before_eval": optimize_mode,
            "max_target_length": 128,
        }

        cls.run_from_dict(parameters)

        file_info = {"metrics": "eval_results"}

        ret = {}
        for k, v in file_info.items():
            with open(output_dir / "checkpoint-0" / (v + ".json")) as f:
                j = json.load(f)
                ret[k] = j

        return ret

