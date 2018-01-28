import queue
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy
import pyaudio

from become_yukarin import AcousticConverter
from become_yukarin import RealtimeVocoder
from become_yukarin import SuperResolution
from become_yukarin import VoiceChanger
from become_yukarin.config.config import create_from_json as create_config
from become_yukarin.config.sr_config import create_from_json as create_sr_config
from become_yukarin.data_struct import Wave
from become_yukarin.voice_changer import VoiceChangerStream
from become_yukarin.voice_changer import VoiceChangerStreamWrapper


class AudioConfig(NamedTuple):
    rate: int
    chunk: int
    vocoder_buffer_size: int
    out_norm: float


queue_input_wave = queue.Queue()
queue_output_wave = queue.Queue()
queue_output_fragment_wave = queue.Queue(maxsize=1)


def convert_worker(audio_config: AudioConfig, wrapper: VoiceChangerStreamWrapper):
    start_time = 0
    time_length = audio_config.chunk / audio_config.rate
    while True:
        wave = queue_input_wave.get()
        wave = Wave(wave=wave, sampling_rate=audio_config.rate)
        wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=wave)
        start_time += len(wave.wave) / wave.sampling_rate

        wave = wrapper.convert_next(time_length=time_length)
        queue_output_wave.put(wave.wave)
        wrapper.remove_previous_wave()


def input_callback(in_data, frame_count, time_info, status_flags, audio_config: AudioConfig):
    print('input', status_flags, flush=True)
    wave = numpy.fromstring(in_data, dtype=numpy.float32)
    queue_input_wave.put(wave)
    return None, pyaudio.paContinue


def output_callback(_, frame_count, time_info, status_flags, audio_config: AudioConfig):
    print('output', status_flags, flush=True)
    try:
        wave = queue_output_fragment_wave.get_nowait()
    except:
        wave = numpy.empty(0)

    while len(wave) < audio_config.chunk:
        wave_next = queue_output_wave.get()
        wave = numpy.concatenate([wave, wave_next])

    wave, wave_fragment = wave[:audio_config.chunk], wave[audio_config.chunk:]
    queue_output_fragment_wave.put(wave_fragment)

    wave *= audio_config.out_norm
    b = wave.astype(numpy.float32).tobytes()
    return b, pyaudio.paContinue


def main():
    print('model loading...', flush=True)

    model_path = Path('./trained/mfcc8-preconvert-innoise03/predictor_350000.npz')
    config_path = Path('./trained/mfcc8-preconvert-innoise03/config.json')
    config = create_config(config_path)
    acoustic_converter = AcousticConverter(config, model_path, gpu=0)
    print('model 1 loaded!', flush=True)

    model_path = Path('./trained/sr-noise3/predictor_70000.npz')
    config_path = Path('./trained/sr-noise3/config.json')
    sr_config = create_sr_config(config_path)
    super_resolution = SuperResolution(sr_config, model_path, gpu=0)
    print('model 2 loaded!', flush=True)

    audio_instance = pyaudio.PyAudio()
    audio_config = AudioConfig(
        rate=config.dataset.param.voice_param.sample_rate,
        chunk=config.dataset.param.voice_param.sample_rate,
        vocoder_buffer_size=config.dataset.param.voice_param.sample_rate // 16,
        out_norm=4.5,
    )

    vocoder = RealtimeVocoder(
        acoustic_feature_param=config.dataset.param.acoustic_feature_param,
        out_sampling_rate=audio_config.rate,
        buffer_size=audio_config.vocoder_buffer_size,
        number_of_pointers=16,
    )
    vocoder.warm_up(audio_config.vocoder_buffer_size / config.dataset.param.voice_param.sample_rate)

    voice_changer = VoiceChanger(
        super_resolution=super_resolution,
        acoustic_converter=acoustic_converter,
        vocoder=vocoder,
    )

    voice_changer_stream = VoiceChangerStream(
        voice_changer=voice_changer,
        sampling_rate=audio_config.rate,
        in_dtype=numpy.float32,
    )

    wrapper = VoiceChangerStreamWrapper(
        voice_changer_stream=voice_changer_stream,
        extra_time=0.2,
    )

    input_audio_stream = audio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=audio_config.rate,
        frames_per_buffer=audio_config.chunk,
        input=True,
        stream_callback=partial(input_callback, audio_config=audio_config)
    )

    output_audio_stream = audio_instance.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=audio_config.rate,
        frames_per_buffer=audio_config.chunk,
        output=True,
        stream_callback=partial(output_callback, audio_config=audio_config)
    )

    convert_worker(audio_config, wrapper)


if __name__ == '__main__':
    main()
