"""
extract low and high quality spectrogram data.
"""

import argparse
import multiprocessing
from pathlib import Path
from pprint import pprint

import numpy
import pysptk
import pyworld

from become_yukarin.dataset.dataset import AcousticFeatureProcess
from become_yukarin.dataset.dataset import WaveFileLoadProcess
from become_yukarin.param import AcousticFeatureParam
from become_yukarin.param import VoiceParam

base_voice_param = VoiceParam()
base_acoustic_feature_param = AcousticFeatureParam()

parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', '-i', type=Path)
parser.add_argument('--output_directory', '-o', type=Path)
parser.add_argument('--sample_rate', type=int, default=base_voice_param.sample_rate)
parser.add_argument('--top_db', type=float, default=base_voice_param.top_db)
parser.add_argument('--pad_second', type=float, default=base_voice_param.pad_second)
parser.add_argument('--frame_period', type=int, default=base_acoustic_feature_param.frame_period)
parser.add_argument('--order', type=int, default=base_acoustic_feature_param.order)
parser.add_argument('--alpha', type=float, default=base_acoustic_feature_param.alpha)
parser.add_argument('--enable_overwrite', action='store_true')
arguments = parser.parse_args()

pprint(dir(arguments))


def generate_file(path):
    out = Path(arguments.output_directory, path.stem + '.npy')
    if out.exists() and not arguments.enable_overwrite:
        return

    # load wave and padding
    wave_file_load_process = WaveFileLoadProcess(
        sample_rate=arguments.sample_rate,
        top_db=arguments.top_db,
        pad_second=arguments.pad_second,
    )
    wave = wave_file_load_process(path, test=True)

    # make acoustic feature
    acoustic_feature_process = AcousticFeatureProcess(
        frame_period=arguments.frame_period,
        order=arguments.order,
        alpha=arguments.alpha,
    )
    feature = acoustic_feature_process(wave, test=True).astype_only_float(numpy.float32)
    high_spectrogram = feature.spectrogram

    fftlen = pyworld.get_cheaptrick_fft_size(arguments.sample_rate)
    low_spectrogram = pysptk.mc2sp(
        feature.mfcc,
        alpha=arguments.alpha,
        fftlen=fftlen,
    )

    # save
    numpy.save(out.absolute(), {
        'low': low_spectrogram,
        'high': high_spectrogram,
    })
    print('saved!', out)


def main():
    paths = list(sorted(arguments.input_directory.glob('*')))
    arguments.output_directory.mkdir(exist_ok=True)

    pool = multiprocessing.Pool()
    pool.map(generate_file, paths)


if __name__ == '__main__':
    main()
