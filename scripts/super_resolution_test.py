import argparse
import glob
import multiprocessing
import re
from functools import partial
from pathlib import Path

import librosa
import numpy

from become_yukarin import SuperResolution
from become_yukarin.config.sr_config import create_from_json as create_config
from become_yukarin.dataset.dataset import AcousticFeatureProcess
from become_yukarin.dataset.dataset import WaveFileLoadProcess

parser = argparse.ArgumentParser()
parser.add_argument('model_names', nargs='+')
parser.add_argument('-md', '--model_directory', type=Path, default=Path('/mnt/dwango/hiroshiba/become-yukarin/'))
parser.add_argument('-iwd', '--input_wave_directory', type=Path,
                    default=Path('/mnt/dwango/hiroshiba/become-yukarin/dataset/yukari-wave/yukari-news/'))
args = parser.parse_args()

model_directory = args.model_directory  # type: Path
input_wave_directory = args.input_wave_directory  # type: Path

paths_test = list(Path('./test_data_sr/').glob('*.wav'))


def extract_number(f):
    s = re.findall("\d+", str(f))
    return int(s[-1]) if s else -1


def process(p: Path, super_resolution: SuperResolution):
    param = config.dataset.param
    wave_process = WaveFileLoadProcess(
        sample_rate=param.voice_param.sample_rate,
        top_db=None,
    )
    acoustic_feature_process = AcousticFeatureProcess(
        frame_period=param.acoustic_feature_param.frame_period,
        order=param.acoustic_feature_param.order,
        alpha=param.acoustic_feature_param.alpha,
    )

    try:
        if p.suffix in ['.npy', '.npz']:
            p = glob.glob(str(input_wave_directory / p.stem) + '.*')[0]
            p = Path(p)
        input = acoustic_feature_process(wave_process(str(p)))
        wave = super_resolution(input.spectrogram, acoustic_feature=input, sampling_rate=param.voice_param.sample_rate)
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

    model_paths = base_model.glob('predictor*.npz')
    model_path = list(sorted(model_paths, key=extract_number))[-1]
    print(model_path)
    super_resolution = SuperResolution(config, model_path)

    output = Path('./output').absolute() / base_model.name
    output.mkdir(exist_ok=True)

    paths = [path_train, path_test] + paths_test

    process_partial = partial(process, super_resolution=super_resolution)
    pool = multiprocessing.Pool()
    pool.map(process_partial, paths)
