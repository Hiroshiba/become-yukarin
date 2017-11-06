from typing import NamedTuple

from .data_struct import AcousticFeature
from .param import Param


class DatasetConfig(NamedTuple):
    param: Param
    input_glob: str
    target_glob: str
    input_mean: AcousticFeature
    input_var: AcousticFeature
    target_mean: AcousticFeature
    target_var: AcousticFeature
    seed: int
    num_test: int


class Config(NamedTuple):
    dataset_config: DatasetConfig
