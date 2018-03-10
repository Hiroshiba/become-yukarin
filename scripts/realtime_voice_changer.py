import librosa
import world4py

world4py._WORLD_LIBRARY_PATH = 'x64_world.dll'

from pathlib import Path
from typing import NamedTuple
from multiprocessing import Queue
from multiprocessing import Process

import numpy
import pyaudio

from become_yukarin import AcousticConverter
from become_yukarin import Vocoder
from become_yukarin import RealtimeVocoder
from become_yukarin import SuperResolution
from become_yukarin import VoiceChanger
from become_yukarin.config.config import Config
from become_yukarin.config.config import create_from_json as create_config
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from become_yukarin.data_struct import Wave
from become_yukarin.data_struct import AcousticFeature
from become_yukarin.voice_changer import VoiceChangerStream
from become_yukarin.voice_changer import VoiceChangerStreamWrapper


class AudioConfig(NamedTuple):
    rate: int
    frame_period: float
    audio_chunk: int
    convert_chunk: int
    vocoder_buffer_size: int
    out_norm: float
    silent_threshold: float


def encode_worker(
        config: Config,
        wrapper: VoiceChangerStreamWrapper,
        audio_config: AudioConfig,
        queue_input: Queue,
        queue_output: Queue,
):
    wrapper.voice_changer_stream.vocoder = Vocoder(
        acoustic_feature_param=config.dataset.param.acoustic_feature_param,
        out_sampling_rate=audio_config.rate,
    )

    start_time = 0
    time_length = audio_config.convert_chunk / audio_config.rate

    while True:
        wave = queue_input.get()

        w = Wave(wave=wave, sampling_rate=audio_config.rate)
        wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=w)
        start_time += time_length

        feature = wrapper.pre_convert_next(time_length=time_length)
        queue_output.put(feature)


def convert_worker(
        config: Config,
        wrapper: VoiceChangerStreamWrapper,
        acoustic_converter: AcousticConverter,
        super_resolution: SuperResolution,
        audio_config: AudioConfig,
        queue_input: Queue,
        queue_output: Queue,
):
    wrapper.voice_changer_stream.voice_changer = VoiceChanger(
        super_resolution=super_resolution,
        acoustic_converter=acoustic_converter,
    )

    start_time = 0
    time_length = audio_config.convert_chunk / audio_config.rate
    while True:
        in_feature: AcousticFeature = queue_input.get()
        wrapper.voice_changer_stream.add_in_feature(
            start_time=start_time,
            feature=in_feature,
            frame_period=audio_config.frame_period,
        )
        start_time += time_length

        out_feature = wrapper.convert_next(time_length=time_length)
        queue_output.put(out_feature)


def decode_worker(
        config: Config,
        wrapper: VoiceChangerStreamWrapper,
        audio_config: AudioConfig,
        queue_input: Queue,
        queue_output: Queue,
):
    wrapper.voice_changer_stream.vocoder = RealtimeVocoder(
        acoustic_feature_param=config.dataset.param.acoustic_feature_param,
        out_sampling_rate=audio_config.rate,
        buffer_size=audio_config.vocoder_buffer_size,
        number_of_pointers=16,
    )
    # vocoder.warm_up(audio_config.vocoder_buffer_size / config.dataset.param.voice_param.sample_rate)

    start_time = 0
    time_length = audio_config.convert_chunk / audio_config.rate
    wave_fragment = numpy.empty(0)
    while True:
        feature: AcousticFeature = queue_input.get()
        wrapper.voice_changer_stream.add_out_feature(
            start_time=start_time,
            feature=feature,
            frame_period=audio_config.frame_period,
        )
        start_time += time_length

        wave = wrapper.post_convert_next(time_length=time_length).wave

        wave_fragment = numpy.concatenate([wave_fragment, wave])
        if len(wave_fragment) >= audio_config.audio_chunk:
            wave, wave_fragment = wave_fragment[:audio_config.audio_chunk], wave_fragment[audio_config.audio_chunk:]

            power = librosa.core.power_to_db(numpy.abs(librosa.stft(wave)) ** 2).mean()
            if power >= audio_config.silent_threshold:
                queue_output.put(wave)


def main():
    print('model loading...', flush=True)

    queue_input_wave = Queue()
    queue_input_feature = Queue()
    queue_output_feature = Queue()
    queue_output_wave = Queue()

    model_path = Path('./trained/pp-weakD-innoise01-tarnoise001/predictor_120000.npz')
    config_path = Path('./trained/pp-weakD-innoise01-tarnoise001/config.json')
    config = create_config(config_path)
    acoustic_converter = AcousticConverter(config, model_path, gpu=0)
    print('model 1 loaded!', flush=True)

    model_path = Path('./trained/sr-noise3/predictor_180000.npz')
    config_path = Path('./trained/sr-noise3/config.json')
    sr_config = create_sr_config(config_path)
    super_resolution = SuperResolution(sr_config, model_path, gpu=0)
    print('model 2 loaded!', flush=True)

    audio_instance = pyaudio.PyAudio()
    audio_config = AudioConfig(
        rate=config.dataset.param.voice_param.sample_rate,
        frame_period=config.dataset.param.acoustic_feature_param.frame_period,
        audio_chunk=config.dataset.param.voice_param.sample_rate,
        convert_chunk=config.dataset.param.voice_param.sample_rate,
        vocoder_buffer_size=config.dataset.param.voice_param.sample_rate // 16,
        out_norm=2.5,
        silent_threshold=-99.0,
    )

    voice_changer_stream = VoiceChangerStream(
        sampling_rate=audio_config.rate,
        frame_period=config.dataset.param.acoustic_feature_param.frame_period,
        order=config.dataset.param.acoustic_feature_param.order,
        in_dtype=numpy.float32,
    )

    wrapper = VoiceChangerStreamWrapper(
        voice_changer_stream=voice_changer_stream,
        extra_time_pre=0.2,
        extra_time=0.1,
    )

    process_encoder = Process(target=encode_worker, kwargs=dict(
        config=config,
        wrapper=wrapper,
        audio_config=audio_config,
        queue_input=queue_input_wave,
        queue_output=queue_input_feature,
    ))
    process_encoder.start()

    process_converter = Process(target=convert_worker, kwargs=dict(
        config=config,
        wrapper=wrapper,
        acoustic_converter=acoustic_converter,
        super_resolution=super_resolution,
        audio_config=audio_config,
        queue_input=queue_input_feature,
        queue_output=queue_output_feature,
    ))
    process_converter.start()

    process_decoder = Process(target=decode_worker, kwargs=dict(
        config=config,
        wrapper=wrapper,
        audio_config=audio_config,
        queue_input=queue_output_feature,
        queue_output=queue_output_wave,
    ))
    process_decoder.start()

    audio_stream = audio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=audio_config.rate,
        frames_per_buffer=audio_config.audio_chunk,
        input=True,
        output=True,
    )

    # process_converter.join()

    while True:
        # input audio
        in_data = audio_stream.read(audio_config.audio_chunk)
        wave = numpy.fromstring(in_data, dtype=numpy.float32)
        print('input', len(wave), flush=True)
        queue_input_wave.put(wave)

        print('queue_input_wave', queue_input_wave.qsize(), flush=True)
        print('queue_input_feature', queue_input_feature.qsize(), flush=True)
        print('queue_output_feature', queue_output_feature.qsize(), flush=True)
        print('queue_output_wave', queue_output_wave.qsize(), flush=True)

        # output
        try:
            wave = queue_output_wave.get_nowait()
        except:
            wave = None

        if wave is not None:
            print('output', len(wave), flush=True)
            wave *= audio_config.out_norm
            b = wave.astype(numpy.float32).tobytes()
            audio_stream.write(b)


if __name__ == '__main__':
    main()
