import world4py

world4py._WORLD_LIBRARY_PATH = 'x64_world.dll'

from functools import partial
from pathlib import Path
import signal
import time
from typing import NamedTuple
from multiprocessing import Queue
from multiprocessing import Process

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
    audio_chunk: int
    convert_chunk: int
    vocoder_buffer_size: int
    out_norm: float


def convert_worker(
        config,
        acoustic_converter,
        super_resolution,
        audio_config: AudioConfig,
        queue_input_wave,
        queue_output_wave,
):
    vocoder = RealtimeVocoder(
        acoustic_feature_param=config.dataset.param.acoustic_feature_param,
        out_sampling_rate=audio_config.rate,
        buffer_size=audio_config.vocoder_buffer_size,
        number_of_pointers=16,
    )
    # vocoder.warm_up(audio_config.vocoder_buffer_size / config.dataset.param.voice_param.sample_rate)

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
        extra_time=0.1,
    )

    start_time = 0
    wave = numpy.zeros(audio_config.convert_chunk * 2, dtype=numpy.float32)
    wave = Wave(wave=wave, sampling_rate=audio_config.rate)
    wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=wave)
    start_time += len(wave.wave) / wave.sampling_rate
    wave = wrapper.convert_next(time_length=1)

    time_length = audio_config.convert_chunk / audio_config.rate
    wave_fragment = numpy.empty(0)
    while True:
        wave = queue_input_wave.get()
        w = Wave(wave=wave, sampling_rate=audio_config.rate)
        wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=w)
        start_time += time_length

        b = time.time()
        wave = wrapper.convert_next(time_length=time_length).wave
        print('time', time.time()-b, flush=True)
        wrapper.remove_previous_wave()
        print('converted wave', len(wave), flush=True)

        wave_fragment = numpy.concatenate([wave_fragment, wave])
        if len(wave_fragment) >= audio_config.audio_chunk:
            wave, wave_fragment = wave_fragment[:audio_config.audio_chunk], wave_fragment[audio_config.audio_chunk:]
            queue_output_wave.put(wave)


def main():
    print('model loading...', flush=True)

    queue_input_wave = Queue()
    queue_output_wave = Queue()

    model_path = Path('./trained/harvest-innoise03/predictor_1390000.npz')
    config_path = Path('./trained/harvest-innoise03/config.json')
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
        audio_chunk=config.dataset.param.voice_param.sample_rate,
        convert_chunk=config.dataset.param.voice_param.sample_rate,
        vocoder_buffer_size=config.dataset.param.voice_param.sample_rate // 16,
        out_norm=2.5,
    )

    process_converter = Process(target=convert_worker, kwargs=dict(
        config=config,
        audio_config=audio_config,
        acoustic_converter=acoustic_converter,
        super_resolution=super_resolution,
        queue_input_wave=queue_input_wave,
        queue_output_wave=queue_output_wave,
    ))
    process_converter.start()

    signal.signal(signal.SIGINT, lambda signum, frame: process_converter.terminate())

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
