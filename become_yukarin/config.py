import json
from pathlib import Path
from typing import List
from typing import NamedTuple
from typing import Union

from .param import Param


class DatasetConfig(NamedTuple):
    param: Param
    input_glob: Path
    target_glob: Path
    input_mean_path: Path
    input_var_path: Path
    target_mean_path: Path
    target_var_path: Path
    features: List[str]
    train_crop_size: int
    global_noise: float
    local_noise: float
    seed: int
    num_test: int


class DiscriminatorModelConfig(NamedTuple):
    in_channels: int
    hidden_channels_list: List[int]
    last_channels: int


class ModelConfig(NamedTuple):
    in_channels: int
    conv_bank_out_channels: int
    conv_bank_k: int
    max_pooling_k: int
    conv_projections_hidden_channels: int
    highway_layers: int
    out_channels: int
    out_size: int
    aligner_out_time_length: int
    disable_last_rnn: bool
    enable_aligner: bool
    discriminator: DiscriminatorModelConfig


class LossConfig(NamedTuple):
    l1: float
    predictor_fake: float
    discriminator_true: float
    discriminator_fake: float


class TrainConfig(NamedTuple):
    batchsize: int
    gpu: int
    log_iteration: int
    snapshot_iteration: int


class ProjectConfig(NamedTuple):
    name: str
    tags: List[str]


class Config(NamedTuple):
    dataset: DatasetConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig
    project: ProjectConfig

    def save_as_json(self, path):
        d = _namedtuple_to_dict(self)
        json.dump(d, open(path, 'w'), indent=2, sort_keys=True, default=_default_path)


def _default_path(o):
    if isinstance(o, Path):
        return str(o)
    raise TypeError(repr(o) + " is not JSON serializable")


def _namedtuple_to_dict(o: NamedTuple):
    return {
        k: v if not hasattr(v, '_asdict') else _namedtuple_to_dict(v)
        for k, v in o._asdict().items()
    }


def create_from_json(s: Union[str, Path]):
    try:
        d = json.loads(s)
    except TypeError:
        d = json.load(open(s))

    discriminator_model_config = DiscriminatorModelConfig(
        in_channels=d['model']['discriminator']['in_channels'],
        hidden_channels_list=d['model']['discriminator']['hidden_channels_list'],
        last_channels=d['model']['discriminator']['last_channels'],
    )

    return Config(
        dataset=DatasetConfig(
            param=Param(),
            input_glob=Path(d['dataset']['input_glob']),
            target_glob=Path(d['dataset']['target_glob']),
            input_mean_path=Path(d['dataset']['input_mean_path']),
            input_var_path=Path(d['dataset']['input_var_path']),
            target_mean_path=Path(d['dataset']['target_mean_path']),
            target_var_path=Path(d['dataset']['target_var_path']),
            features=d['dataset']['features'],
            train_crop_size=d['dataset']['train_crop_size'],
            global_noise=d['dataset']['global_noise'],
            local_noise=d['dataset']['local_noise'],
            seed=d['dataset']['seed'],
            num_test=d['dataset']['num_test'],
        ),
        model=ModelConfig(
            in_channels=d['model']['in_channels'],
            conv_bank_out_channels=d['model']['conv_bank_out_channels'],
            conv_bank_k=d['model']['conv_bank_k'],
            max_pooling_k=d['model']['max_pooling_k'],
            conv_projections_hidden_channels=d['model']['conv_projections_hidden_channels'],
            highway_layers=d['model']['highway_layers'],
            out_channels=d['model']['out_channels'],
            out_size=d['model']['out_size'],
            aligner_out_time_length=d['model']['aligner_out_time_length'],
            disable_last_rnn=d['model']['disable_last_rnn'],
            enable_aligner=d['model']['enable_aligner'],
            discriminator=discriminator_model_config,
        ),
        loss=LossConfig(
            l1=d['loss']['l1'],
            predictor_fake=d['loss']['predictor_fake'],
            discriminator_true=d['loss']['discriminator_true'],
            discriminator_fake=d['loss']['discriminator_fake'],
        ),
        train=TrainConfig(
            batchsize=d['train']['batchsize'],
            gpu=d['train']['gpu'],
            log_iteration=d['train']['log_iteration'],
            snapshot_iteration=d['train']['snapshot_iteration'],
        ),
        project=ProjectConfig(
            name=d['project']['name'],
            tags=d['project']['tags'],
        )
    )
