import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

from become_yukarin.param import Param


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
    input_global_noise: float
    input_local_noise: float
    target_global_noise: float
    target_local_noise: float
    seed: int
    num_test: int


class ModelConfig(NamedTuple):
    in_channels: int
    out_channels: int
    generator_base_channels: int
    generator_extensive_layers: int
    discriminator_base_channels: int
    discriminator_extensive_layers: int
    weak_discriminator: bool


class LossConfig(NamedTuple):
    mse: float
    adversarial: float


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

    backward_compatible(d)

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
            input_global_noise=d['dataset']['input_global_noise'],
            input_local_noise=d['dataset']['input_local_noise'],
            target_global_noise=d['dataset']['target_global_noise'],
            target_local_noise=d['dataset']['target_local_noise'],
            seed=d['dataset']['seed'],
            num_test=d['dataset']['num_test'],
        ),
        model=ModelConfig(
            in_channels=d['model']['in_channels'],
            out_channels=d['model']['out_channels'],
            generator_base_channels=d['model']['generator_base_channels'],
            generator_extensive_layers=d['model']['generator_extensive_layers'],
            discriminator_base_channels=d['model']['discriminator_base_channels'],
            discriminator_extensive_layers=d['model']['discriminator_extensive_layers'],
            weak_discriminator=d['model']['weak_discriminator'],
        ),
        loss=LossConfig(
            mse=d['loss']['mse'],
            adversarial=d['loss']['adversarial'],
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


def backward_compatible(d: Dict):
    if 'input_global_noise' not in d['dataset']:
        d['dataset']['input_global_noise'] = d['dataset']['global_noise']
        d['dataset']['input_local_noise'] = d['dataset']['local_noise']

    if 'target_global_noise' not in d['dataset']:
        d['dataset']['target_global_noise'] = d['dataset']['global_noise']
        d['dataset']['target_local_noise'] = d['dataset']['local_noise']

    if 'generator_base_channels' not in d['model']:
        d['model']['generator_base_channels'] = 64
        d['model']['generator_extensive_layers'] = 8
        d['model']['discriminator_base_channels'] = 32
        d['model']['discriminator_extensive_layers'] = 5

    if 'weak_discriminator' not in d['model']:
        d['model']['weak_discriminator'] = False
