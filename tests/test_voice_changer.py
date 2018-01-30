from pathlib import Path
from typing import NamedTuple

import librosa
import numpy

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


model_base_path = Path('~/trained/')
test_data_path = Path('tests/test-deep-learning-yuduki-yukari.wav')
test_output_path = Path('tests/output.wav')

print('model loading...', flush=True)

model_path = model_base_path / Path('harvest-innoise03/predictor_1340000.npz')
config_path = model_base_path / Path('harvest-innoise03/config.json')
config = create_config(config_path)
acoustic_converter = AcousticConverter(config, model_path, gpu=0)
print('model 1 loaded!', flush=True)

model_path = model_base_path / Path('sr-noise3/predictor_165000.npz')
config_path = model_base_path / Path('sr-noise3/config.json')
sr_config = create_sr_config(config_path)
super_resolution = SuperResolution(sr_config, model_path, gpu=0)
print('model 2 loaded!', flush=True)

audio_config = AudioConfig(
    rate=config.dataset.param.voice_param.sample_rate,
    chunk=config.dataset.param.voice_param.sample_rate // 4,
    vocoder_buffer_size=config.dataset.param.voice_param.sample_rate // 16,
    out_norm=4.5,
)

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
    extra_time=0.2,
)

raw_wave, _ = librosa.load(str(test_data_path), sr=audio_config.rate)
wave_out_list = []

start_time = 0
for i in range(0, len(raw_wave), audio_config.chunk):
    wave_in = Wave(wave=raw_wave[i:i + audio_config.chunk], sampling_rate=audio_config.rate)
    wrapper.voice_changer_stream.add_wave(start_time=start_time, wave=wave_in)
    start_time += len(wave_in.wave) / wave_in.sampling_rate

    wave_out = wrapper.convert_next(time_length=audio_config.chunk / audio_config.rate)
    wave_out_list.append(wave_out)
    wrapper.remove_previous_wave()

out_wave = numpy.concatenate([w.wave for w in wave_out_list]).astype(numpy.float32)
librosa.output.write_wav(str(test_output_path), out_wave, sr=audio_config.rate)
