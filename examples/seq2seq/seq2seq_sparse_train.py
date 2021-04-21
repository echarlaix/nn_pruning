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
Sparse Fine-tuning the library models for sequence to sequence.
"""


from inspect import signature
from dataclasses import dataclass, field
import timeit
from pathlib import Path
import json
import os.path
from transformers import (
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
)
from examples.xp import XPTrainingArguments

@dataclass
class Seq2SeqXPTrainingArguments(Seq2SeqTrainingArguments):
    optimize_model_before_eval: str = field(
        default="disabled",
        metadata={
            "help": "Apply some optimization to model before evaluation (use nn_pruning.inference_model_patcher.InferencePatcher)."
                    "Valid values: disabled, block_sparse, dense"
        },
    )


class Seq2SeqXPTrainer(Seq2SeqTrainer):

    def checkpoint_dir(self):
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        checkpoint_dir = self.run_dir() / checkpoint_folder

        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir()

        return checkpoint_dir

    def run_dir(self):
        return Path(self.args.output_dir)

    def finish_evaluate(self, checkpoint_dir, metrics):
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        for k, v in self.__dict__.items():
            if k.endswith("_args") and k != "args":
                filename = k + ".json"
                s = json.dumps(v.__dict__, indent=4, sort_keys=True)
                with open(os.path.join(checkpoint_dir, filename), "w") as f:
                    f.write(s)
