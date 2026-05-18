from collections import namedtuple
from dataclasses import dataclass
from typing import List
import tensorflow as tf

@dataclass(frozen=True)
class ModelInfo:
    wandb_id: str
    lesion_start_epoch: int
    lesion_type: str
    model_type: str
    run_name: str 
    train_data: str
    test_data: List[str]
    mask_value: int
    lstm_units: int
    learning_rate: float
    batch_size: int
    frequency_scale_k: float
    epochs: int
    seed: int
    orth_features: int
    phon_features: int
    phon_max_length: int
    name: str
    bucket_name: str

# Refactored to a dataclass... we'll see!
#
# ModelInfo = namedtuple("ModelInfo", [
#     "wandb_id", "lesion_start_epoch", "lesion_type", "model_type", "run_name", 
#     "train_data", "test_data", "mask_value", "lstm_units", "learning_rate",
#     "batch_size", "frequency_scale_k", "epochs", "seed", "orth_features",
#     "phon_features", "phon_max_length", "name", "bucket_name"])

# I don't think there are model states saved in the gcloud other than weights,
# so this may not be necessary.
#
# @dataclass(frozen=True)
# class ModelState:
#     encoder_cell_state: tf.Tensor
#     encoder_hidden_state: tf.Tensor
#     decoder_cell_state: tf.Tensor
#     decoder_hidden_state: tf.Tensor
#     output: tf.Tensor
#     name: str
#     bucket_name: str

BaseModelState = namedtuple("ModelState", [
    "encoder_cell_state", "encoder_hidden_state", "decoder_cell_state",
    "decoder_hidden_state", "output", "name", "bucket_name"])


class ModelState(BaseModelState):
    __slots__ = ()
    def nitems(self):
        return self.encoder_cell.shape[0]

    def nunits(self):
        return self.encoder_cell.shape[-1]

    def phon_max_length(self):
        return self.output.shape[1]

