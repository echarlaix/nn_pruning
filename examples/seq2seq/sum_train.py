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
A subclass of `Trainer` specific to Summary tasks
"""

import json
import os

from .seq2seq_sparse_train import Seq2SeqXPTrainer

from transformers.utils import logging
from transformers.trainer_utils import (
    PredictionOutput,
)
import numpy as np
import datasets

logger = logging.get_logger(__name__)


class SumTrainer(Seq2SeqXPTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        self.model_args = kwargs.pop("model_args")
        self.data_args = kwargs.pop("data_args")
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            ignore_keys=ignore_keys,
        )

        checkpoint_dir = self.checkpoint_dir()

        metrics = output.metrics

        if self.is_world_process_zero():
            logger.info("***** Eval results *****")
            for key, value in metrics.items():
                logger.info(f"  {key} = {value}")

            filename = "eval_results.json"
            s = json.dumps(metrics, indent=4, sort_keys=True)
            with open(os.path.join(checkpoint_dir, filename), "w") as f:
                f.write(s)

            super().finish_evaluate(checkpoint_dir, metrics)

        return metrics


    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.test_dataloader if test_dataset is None else test_dataset

        output = self.prediction_loop(
            test_dataloader,
            description="Prediction",
            prediction_loss_only=True,
            ignore_keys=ignore_keys,
        )

        return output.predictions
