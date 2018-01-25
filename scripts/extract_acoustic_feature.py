"""
extract alignments voices.
"""

import argparse
import multiprocessing
from pathlib import Path
from pprint import pprint

import numpy

from become_yukarin.acoustic_converter import AcousticConverter
from become_yukarin.config.config import create_from_json as create_config
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
parser.add_argument('--pre_converter1_config', type=Path)
parser.add_argument('--pre_converter1_model', type=Path)
parser.add_argument('--sample_rate', type=int, default=base_voice_param.sample_rate)
parser.add_argument('--top_db', type=float, default=base_voice_param.top_db)
parser.add_argument('--pad_second', type=float, default=base_voice_param.pad_second)
parser.add_argument('--frame_period', type=int, default=base_acoustic_feature_param.frame_period)
parser.add_argument('--order', type=int, default=base_acoustic_feature_param.order)
parser.add_argument('--alpha', type=float, default=base_acoustic_feature_param.alpha)
parser.add_argument('--f0_estimating_method', type=str, default=base_acoustic_feature_param.f0_estimating_method)
parser.add_argument('--f0_floor1', type=float, default=71)
parser.add_argument('--f0_ceil1', type=float, default=800)
parser.add_argument('--f0_floor2', type=float, default=71)
parser.add_argument('--f0_ceil2', type=float, default=800)
parser.add_argument('--ignore_feature', nargs='+', default=['spectrogram', 'aperiodicity'])
parser.add_argument('--disable_alignment', action='store_true')
parser.add_argument('--enable_overwrite', action='store_true')
arguments = parser.parse_args()

pprint(dir(arguments))

pre_convert = arguments.pre_converter1_config is not None
if pre_convert:
    config = create_config(arguments.pre_converter1_config)
    pre_converter1 = AcousticConverter(config, arguments.pre_converter1_model)
else:
    pre_converter1 = None


def generate_feature(path1, path2):
    out1 = Path(arguments.output1_directory, path1.stem + '.npy')
    out2 = Path(arguments.output2_directory, path2.stem + '.npy')
    if out1.exists() and out2.exists() and not arguments.enable_overwrite:
        return

    # load wave and padding
    wave_file_load_process = WaveFileLoadProcess(
        sample_rate=arguments.sample_rate,
        top_db=arguments.top_db,
        pad_second=arguments.pad_second,
    )
    wave1 = wave_file_load_process(path1, test=True)
    wave2 = wave_file_load_process(path2, test=True)

    # make acoustic feature
    acoustic_feature_process1 = AcousticFeatureProcess(
        frame_period=arguments.frame_period,
        order=arguments.order,
        alpha=arguments.alpha,
        f0_estimating_method=arguments.f0_estimating_method,
        f0_floor=arguments.f0_floor1,
        f0_ceil=arguments.f0_ceil1,
    )
    acoustic_feature_process2 = AcousticFeatureProcess(
        frame_period=arguments.frame_period,
        order=arguments.order,
        alpha=arguments.alpha,
        f0_estimating_method=arguments.f0_estimating_method,
        f0_floor=arguments.f0_floor2,
        f0_ceil=arguments.f0_ceil2,
    )
    f1 = acoustic_feature_process1(wave1, test=True).astype_only_float(numpy.float32)
    f2 = acoustic_feature_process2(wave2, test=True).astype_only_float(numpy.float32)

    # pre convert
    if pre_convert:
        f1_ref = pre_converter1.convert_to_feature(f1)
    else:
        f1_ref = f1

    # alignment
    if not arguments.disable_alignment:
        aligner = MFCCAligner(f1_ref.mfcc, f2.mfcc)

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
    acoustic_feature_save_process = AcousticFeatureSaveProcess(validate=True, ignore=arguments.ignore_feature)
    acoustic_feature_save_process({'path': out1, 'feature': f1})
    print('saved!', out1)

    acoustic_feature_save_process({'path': out2, 'feature': f2})
    print('saved!', out2)


def generate_mean_var(path_directory: Path):
    path_mean = Path(path_directory, 'mean.npy')
    path_var = Path(path_directory, 'var.npy')
    if path_mean.exists():
        path_mean.unlink()
    if path_var.exists():
        path_var.unlink()

    acoustic_feature_load_process = AcousticFeatureLoadProcess(validate=False)
    acoustic_feature_save_process = AcousticFeatureSaveProcess(validate=False)

    f0_list = []
    spectrogram_list = []
    aperiodicity_list = []
    mfcc_list = []
    for path in path_directory.glob('*'):
        feature = acoustic_feature_load_process(path)
        f0_list.append(feature.f0[feature.voiced])  # remove unvoiced
        spectrogram_list.append(feature.spectrogram)
        aperiodicity_list.append(feature.aperiodicity)
        mfcc_list.append(feature.mfcc)

    def concatenate(arr_list):
        try:
            arr_list = numpy.concatenate(arr_list)
        except:
            pass
        return arr_list

    f0_list = concatenate(f0_list)
    spectrogram_list = concatenate(spectrogram_list)
    aperiodicity_list = concatenate(aperiodicity_list)
    mfcc_list = concatenate(mfcc_list)

    mean = AcousticFeature(
        f0=numpy.mean(f0_list, axis=0, keepdims=True),
        spectrogram=numpy.mean(spectrogram_list, axis=0, keepdims=True),
        aperiodicity=numpy.mean(aperiodicity_list, axis=0, keepdims=True),
        mfcc=numpy.mean(mfcc_list, axis=0, keepdims=True),
        voiced=numpy.nan,
    )
    var = AcousticFeature(
        f0=numpy.var(f0_list, axis=0, keepdims=True),
        spectrogram=numpy.var(spectrogram_list, axis=0, keepdims=True),
        aperiodicity=numpy.var(aperiodicity_list, axis=0, keepdims=True),
        mfcc=numpy.var(mfcc_list, axis=0, keepdims=True),
        voiced=numpy.nan,
    )

    acoustic_feature_save_process({'path': path_mean, 'feature': mean})
    acoustic_feature_save_process({'path': path_var, 'feature': var})


def main():
    paths1 = list(sorted(arguments.input1_directory.glob('*')))
    paths2 = list(sorted(arguments.input2_directory.glob('*')))
    assert len(paths1) == len(paths2)

    arguments.output1_directory.mkdir(exist_ok=True)
    arguments.output2_directory.mkdir(exist_ok=True)

    pool = multiprocessing.Pool()
    pool.starmap(generate_feature, zip(paths1, paths2), chunksize=16)

    generate_mean_var(arguments.output1_directory)
    generate_mean_var(arguments.output2_directory)


if __name__ == '__main__':
    main()
