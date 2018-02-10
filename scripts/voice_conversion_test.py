import argparse
import glob
import multiprocessing
import re
from functools import partial
from pathlib import Path

import librosa
import numpy

from become_yukarin import AcousticConverter
from become_yukarin.config.config import create_from_json as create_config

parser = argparse.ArgumentParser()
parser.add_argument('model_names', nargs='+')
parser.add_argument('-md', '--model_directory', type=Path, default=Path('/mnt/dwango/hiroshiba/become-yukarin/'))
parser.add_argument('-iwd', '--input_wave_directory', type=Path,
                    default=Path('/mnt/dwango/hiroshiba/become-yukarin/dataset/hiho-wave/hiho-pause-atr503-subset/'))
parser.add_argument('-it', '--iteration', type=int)
parser.add_argument('-g', '--gpu', type=int)
args = parser.parse_args()

model_directory = args.model_directory  # type: Path
input_wave_directory = args.input_wave_directory  # type: Path
it = args.iteration
gpu = args.gpu

paths_test = list(Path('./test_data/').glob('*.wav'))


def extract_number(f):
    s = re.findall("\d+", str(f))
    return int(s[-1]) if s else -1


def process(p: Path, acoustic_converter: AcousticConverter):
    try:
        if p.suffix in ['.npy', '.npz']:
            fn = glob.glob(str(input_wave_directory / p.stem) + '.*')[0]
            p = Path(fn)
        wave = acoustic_converter(p)
        librosa.output.write_wav(str(output / p.stem) + '.wav', wave.wave, wave.sampling_rate, norm=True)
    except:
        import traceback
        print('error!', str(p))
        traceback.format_exc()


for model_name in args.model_names:
    base_model = model_directory / model_name
    config = create_config(base_model / 'config.json')

    input_paths = list(sorted([Path(p) for p in glob.glob(str(config.dataset.input_glob))]))
    numpy.random.RandomState(config.dataset.seed).shuffle(input_paths)
    path_train = input_paths[0]
    path_test = input_paths[-1]

    if it is not None:
        model_path = base_model / 'predictor_{}.npz'.format(it)
    else:
        model_paths = base_model.glob('predictor_*.npz')
        model_path = list(sorted(model_paths, key=extract_number))[-1]
    print(model_path)
    acoustic_converter = AcousticConverter(config, model_path, gpu=gpu)

    output = Path('./output').absolute() / base_model.name
    output.mkdir(exist_ok=True)

    paths = [path_train, path_test] + paths_test

    process_partial = partial(process, acoustic_converter=acoustic_converter)
    if gpu is None:
        pool = multiprocessing.Pool()
        pool.map(process_partial, paths)
    else:
        list(map(process_partial, paths))
