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
A subclass of `Trainer` specific to Question-Answering tasks
"""

import json
import os
from pathlib import Path
from transformers import Trainer, is_datasets_available, is_torch_tpu_available
from transformers.integrations import is_ray_available
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    HPSearchBackend,
    PredictionOutput,
)
from transformers.utils import logging

if is_ray_available():
    from ray import tune

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

logger = logging.get_logger(__name__)


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        self.model_args = kwargs.pop("model_args")
        self.data_args = kwargs.pop("data_args")
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def checkpoint_dir(self):
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        trial = self._trial
        if self.hp_search_backend is not None and trial is not None:
            run_id = (
                trial.number
                if self.hp_search_backend == HPSearchBackend.OPTUNA
                else tune.get_trial_id()
            )
            run_name = (
                self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            )
            checkpoint_dir = Path(self.args.output_dir) / run_name / checkpoint_folder
        else:
            checkpoint_dir = Path(self.args.output_dir) / checkpoint_folder

        return checkpoint_dir

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        if True:
            try:
                output = self.prediction_loop(
                    eval_dataloader,
                    description="Evaluation",
                    # No point gathering the predictions if there are no metrics, otherwise we defer to
                    # self.args.prediction_loss_only
                    prediction_loss_only=True if compute_metrics is None else None,
                    ignore_keys=ignore_keys,
                )
            finally:
                self.compute_metrics = compute_metrics

            # We might have removed columns from the dataset so we put them back.
            if isinstance(eval_dataset, datasets.Dataset):
                eval_dataset.set_format(
                    type=eval_dataset.format["type"],
                    columns=list(eval_dataset.features.keys()),
                )

            if (
                self.post_process_function is not None
                and self.compute_metrics is not None
            ):
                eval_preds = self.post_process_function(
                    eval_examples, eval_dataset, output.predictions
                )
                metrics = self.compute_metrics(eval_preds)

                log_metrics = {"eval_" + k: v for k, v in metrics.items()}
                self.log(log_metrics)
            else:
                metrics = {}

            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics
            )
        else:
            metrics = {"eval_loss": 1.03}

        if self.is_world_process_zero():
            checkpoint_dir = self.checkpoint_dir()
            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir()

            logger.info("***** Eval results *****")
            for key, value in metrics.items():
                logger.info(f"  {key} = {value}")

            filename = "eval_metrics.json"
            s = json.dumps(metrics, indent=4, sort_keys=True)
            with open(os.path.join(checkpoint_dir, filename), "w") as f:
                f.write(s)

            for k, v in self.__dict__.items():
                if k.endswith("_args") and k != "args":
                    filename = k + ".json"
                    s = json.dumps(v.__dict__, indent=4, sort_keys=True)
                    with open(os.path.join(checkpoint_dir, filename), "w") as f:
                        f.write(s)

        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        # We might have removed columns from the dataset so we put them back.
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        eval_preds = self.post_process_function(
            test_examples, test_dataset, output.predictions
        )
        metrics = self.compute_metrics(eval_preds)

        return PredictionOutput(
            predictions=eval_preds.predictions,
            label_ids=eval_preds.label_ids,
            metrics=metrics,
        )
