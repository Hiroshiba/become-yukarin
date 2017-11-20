"""
extract alignments voices.
"""

import argparse
import multiprocessing
from pathlib import Path

import numpy

from become_yukarin.data_struct import AcousticFeature
from become_yukarin.dataset.dataset import AcousticFeatureLoadProcess
from become_yukarin.dataset.dataset import AcousticFeatureProcess
from become_yukarin.dataset.dataset import AcousticFeatureSaveProcess
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
parser.add_argument('--disable_alignment', action='store_true')
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


def generate_feature(path1, path2):
    # load wave and padding
    wave_file_load_process = WaveFileLoadProcess(
        sample_rate=arguments.sample_rate,
        top_db=arguments.top_db,
    )
    wave1 = wave_file_load_process(path1, test=True)
    wave2 = wave_file_load_process(path2, test=True)

    # make acoustic feature
    acoustic_feature_process = AcousticFeatureProcess(
        frame_period=arguments.frame_period,
        order=arguments.order,
        alpha=arguments.alpha,
    )
    f1 = acoustic_feature_process(wave1, test=True)
    f2 = acoustic_feature_process(wave2, test=True)

    # alignment
    if not arguments.disable_alignment:
        aligner = MFCCAligner(f1.mfcc, f2.mfcc)

        f0_1, f0_2 = aligner.align(f1.f0, f2.f0)
        spectrogram_1, spectrogram_2 = aligner.align(f1.spectrogram, f2.spectrogram)
        aperiodicity_1, aperiodicity_2 = aligner.align(f1.aperiodicity, f2.aperiodicity)
        mfcc_1, mfcc_2 = aligner.align(f1.mfcc, f2.mfcc)
        voiced_1, voiced_2 = aligner.align(f1.voiced, f2.voiced)

        f1 = AcousticFeature(
            f0=f0_1,
            spectrogram=spectrogram_1,
            aperiodicity=aperiodicity_1,
            mfcc=mfcc_1,
            voiced=voiced_1,
        )
        f2 = AcousticFeature(
            f0=f0_2,
            spectrogram=spectrogram_2,
            aperiodicity=aperiodicity_2,
            mfcc=mfcc_2,
            voiced=voiced_2,
        )

        f1.validate()
        f2.validate()

    # save
    acoustic_feature_save_process = AcousticFeatureSaveProcess(validate=True)
    path = Path(arguments.output1_directory, path1.stem + '.npy')
    feature = AcousticFeature(
        f0=f1.f0,
        spectrogram=f1.spectrogram,
        aperiodicity=f1.aperiodicity,
        mfcc=f1.mfcc,
        voiced=f1.voiced,
    )
    acoustic_feature_save_process({'path': path, 'feature': feature})
    print('saved!', path)

    path = Path(arguments.output2_directory, path2.stem + '.npy')
    feature = AcousticFeature(
        f0=f2.f0,
        spectrogram=f2.spectrogram,
        aperiodicity=f2.aperiodicity,
        mfcc=f2.mfcc,
        voiced=f2.voiced,
    )
    acoustic_feature_save_process({'path': path, 'feature': feature})
    print('saved!', path)


def generate_mean_var(path_directory: Path):
    path_mean = Path(path_directory, 'mean.npy')
    var_mean = Path(path_directory, 'var.npy')
    if path_mean.exists():
        path_mean.unlink()
    if var_mean.exists():
        var_mean.unlink()

    acoustic_feature_load_process = AcousticFeatureLoadProcess(validate=True)
    acoustic_feature_save_process = AcousticFeatureSaveProcess(validate=False)

    f0_list = []
    spectrogram_list = []
    aperiodicity_list = []
    mfcc_list = []
    for path in path_directory.glob('*'):
        feature = acoustic_feature_load_process(path)
        f0_list.append(feature.f0[feature.voiced].ravel())  # remove unvoiced
        spectrogram_list.append(feature.spectrogram.ravel())
        aperiodicity_list.append(feature.aperiodicity.ravel())
        mfcc_list.append(feature.mfcc.ravel())

    f0_list = numpy.concatenate(f0_list)
    spectrogram_list = numpy.concatenate(spectrogram_list)
    aperiodicity_list = numpy.concatenate(aperiodicity_list)
    mfcc_list = numpy.concatenate(mfcc_list)

    mean = AcousticFeature(
        f0=numpy.mean(f0_list),
        spectrogram=numpy.mean(spectrogram_list),
        aperiodicity=numpy.mean(aperiodicity_list),
        mfcc=numpy.mean(mfcc_list),
        voiced=numpy.nan,
    )
    var = AcousticFeature(
        f0=numpy.var(f0_list),
        spectrogram=numpy.var(spectrogram_list),
        aperiodicity=numpy.var(aperiodicity_list),
        mfcc=numpy.var(mfcc_list),
        voiced=numpy.nan,
    )

    acoustic_feature_save_process({'path': path_mean, 'feature': mean})
    acoustic_feature_save_process({'path': var_mean, 'feature': var})


def main():
    paths1 = list(sorted(arguments.input1_directory.glob('*')))
    paths2 = list(sorted(arguments.input2_directory.glob('*')))
    assert len(paths1) == len(paths2)

    arguments.output1_directory.mkdir(exist_ok=True)
    arguments.output2_directory.mkdir(exist_ok=True)

    pool = multiprocessing.Pool()
    pool.starmap(generate_feature, zip(paths1, paths2))

    generate_mean_var(arguments.output1_directory)
    generate_mean_var(arguments.output2_directory)


if __name__ == '__main__':
    main()
