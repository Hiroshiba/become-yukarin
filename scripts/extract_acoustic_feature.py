"""
extract alignments voices.
"""

import argparse
import multiprocessing
from pathlib import Path

import numpy

from become_yukarin.dataset.dataset import AcousticFeatureProcess
from become_yukarin.dataset.dataset import Wave
from become_yukarin.dataset.dataset import WaveFileLoadProcess
from become_yukarin.dataset.utility import MFCCAligner
from become_yukarin.param import AcousticFeatureParam
from become_yukarin.param import VoiceParam

base_voice_param = VoiceParam()
base_acoustic_feature_param = AcousticFeatureParam()

parser = argparse.ArgumentParser()
parser.add_argument('--input1_directory', '-i1', type=Path)
parser.add_argument('--input2_directory', '-i2', type=Path)
parser.add_argument('--output1_directory', '-o1', type=Path)
parser.add_argument('--output2_directory', '-o2', type=Path)
parser.add_argument('--sample_rate', type=int, default=base_voice_param.sample_rate)
parser.add_argument('--top_db', type=float, default=base_voice_param.top_db)
parser.add_argument('--frame_period', type=int, default=base_acoustic_feature_param.frame_period)
parser.add_argument('--order', type=int, default=base_acoustic_feature_param.order)
parser.add_argument('--alpha', type=float, default=base_acoustic_feature_param.alpha)
arguments = parser.parse_args()


def make_feature(
        path,
        sample_rate,
        top_db,
        frame_period,
        order,
        alpha,
):
    wave = WaveFileLoadProcess(sample_rate=sample_rate, top_db=top_db)(path, test=True)
    feature = AcousticFeatureProcess(frame_period=frame_period, order=order, alpha=alpha)(wave, test=True)
    return feature


def process(path1, path2):
    # load wave and padding
    wave_file_load_process = WaveFileLoadProcess(
        sample_rate=arguments.sample_rate,
        top_db=arguments.top_db,
    )
    wave1 = wave_file_load_process(path1, test=True)
    wave2 = wave_file_load_process(path2, test=True)

    m = max(len(wave1.wave), len(wave2.wave))
    wave1 = Wave(wave=numpy.pad(wave1.wave, (0, m - len(wave1.wave)), mode='mean'), sampling_rate=wave1.sampling_rate)
    wave2 = Wave(wave=numpy.pad(wave2.wave, (0, m - len(wave2.wave)), mode='mean'), sampling_rate=wave2.sampling_rate)

    # make acoustic feature
    acoustic_feature_process = AcousticFeatureProcess(
        frame_period=arguments.frame_period,
        order=arguments.order,
        alpha=arguments.alpha,
    )
    f1 = acoustic_feature_process(wave1, test=True)
    f2 = acoustic_feature_process(wave2, test=True)

    # alignment
    aligner = MFCCAligner(f1.mfcc, f2.mfcc)

    f0_1, f0_2 = aligner.align(f1.f0, f2.f0)
    spectrogram_1, spectrogram_2 = aligner.align(f1.spectrogram, f2.spectrogram)
    aperiodicity_1, aperiodicity_2 = aligner.align(f1.aperiodicity, f2.aperiodicity)
    mfcc_1, mfcc_2 = aligner.align(f1.mfcc, f2.mfcc)

    # save
    path = Path(arguments.output1_directory, path1.stem + '.npy')
    numpy.save(path.absolute(), dict(f0=f0_1, spectrogram=spectrogram_1, aperiodicity=aperiodicity_1, mfcc=mfcc_1))
    print('saved!', path)

    path = Path(arguments.output2_directory, path2.stem + '.npy')
    numpy.save(path.absolute(), dict(f0=f0_2, spectrogram=spectrogram_2, aperiodicity=aperiodicity_2, mfcc=mfcc_2))
    print('saved!', path)


def main():
    paths1 = list(sorted(arguments.input1_directory.glob('*')))
    paths2 = list(sorted(arguments.input2_directory.glob('*')))
    assert len(paths1) == len(paths2)

    arguments.output1_directory.mkdir(exist_ok=True)
    arguments.output2_directory.mkdir(exist_ok=True)

    pool = multiprocessing.Pool()
    pool.starmap(process, zip(paths1, paths2))


if __name__ == '__main__':
    main()
