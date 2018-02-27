import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Union

from become_yukarin.param import Param


class SRDatasetConfig(NamedTuple):
    param: Param
    input_glob: Path
    train_crop_size: int
    input_global_noise: float
    input_local_noise: float
    blur_size_factor: float
    seed: int
    num_test: int


class SRModelConfig(NamedTuple):
    generator_base_channels: int
    generator_extensive_layers: int
    discriminator_base_channels: int
    discriminator_extensive_layers: int


class SRLossConfig(NamedTuple):
    mse: float
    adversarial: float


class SRTrainConfig(NamedTuple):
    batchsize: int
    gpu: int
    log_iteration: int
    snapshot_iteration: int


class SRProjectConfig(NamedTuple):
    name: str
    tags: List[str]


class SRConfig(NamedTuple):
    dataset: SRDatasetConfig
    model: SRModelConfig
    loss: SRLossConfig
    train: SRTrainConfig
    project: SRProjectConfig

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

    return SRConfig(
        dataset=SRDatasetConfig(
            param=Param(),
            input_glob=Path(d['dataset']['input_glob']),
            train_crop_size=d['dataset']['train_crop_size'],
            input_global_noise=d['dataset']['input_global_noise'],
            input_local_noise=d['dataset']['input_local_noise'],
            blur_size_factor=d['dataset']['blur_size_factor'],
            seed=d['dataset']['seed'],
            num_test=d['dataset']['num_test'],
        ),
        model=SRModelConfig(
            generator_base_channels=d['model']['generator_base_channels'],
            generator_extensive_layers=d['model']['generator_extensive_layers'],
            discriminator_base_channels=d['model']['discriminator_base_channels'],
            discriminator_extensive_layers=d['model']['discriminator_extensive_layers'],
        ),
        loss=SRLossConfig(
            mse=d['loss']['mse'],
            adversarial=d['loss']['adversarial'],
        ),
        train=SRTrainConfig(
            batchsize=d['train']['batchsize'],
            gpu=d['train']['gpu'],
            log_iteration=d['train']['log_iteration'],
            snapshot_iteration=d['train']['snapshot_iteration'],
        ),
        project=SRProjectConfig(
            name=d['project']['name'],
            tags=d['project']['tags'],
        )
    )


def backward_compatible(d: Dict):
    if 'blur_size_factor' not in d['dataset']:
        d['dataset']['blur_size_factor'] = 0
