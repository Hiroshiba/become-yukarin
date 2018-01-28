from typing import List
from typing import NamedTuple

import numpy

from .acoustic_converter import AcousticConverter
from .data_struct import AcousticFeature
from .data_struct import Wave
from .super_resolution import SuperResolution
from .vocoder import Vocoder


class VoiceChanger(object):
    def __init__(
            self,
            acoustic_converter: AcousticConverter,
            super_resolution: SuperResolution,
            vocoder: Vocoder,
            output_sampling_rate: int = None,
    ) -> None:
        if output_sampling_rate is None:
            output_sampling_rate = super_resolution.config.dataset.param.voice_param.sample_rate

        self.acoustic_converter = acoustic_converter
        self.super_resolution = super_resolution
        self.vocoder = vocoder
        self.output_sampling_rate = output_sampling_rate

    def convert_from_wave_path(self, wave_path: str):
        w_in = self.acoustic_converter._wave_process(wave_path)
        return self.convert_from_wave(w_in)

    def convert_from_wave(self, wave: Wave):
        f_in = self.acoustic_converter._feature_process(wave)
        f_high = self.convert_from_acoustic_feature(f_in)
        wave = self.vocoder.decode(f_high)
        return wave

    def convert_from_acoustic_feature(self, f_in: AcousticFeature):
        f_low = self.acoustic_converter.convert_to_feature(f_in)
        s_high = self.super_resolution.convert(f_low.spectrogram.astype(numpy.float32))
        f_high = self.super_resolution.convert_to_feature(s_high, f_low)
        return f_high


class Segment(NamedTuple):
    start_time: float
    wave: Wave

    @property
    def time_length(self):
        return len(self.wave.wave) / self.wave.sampling_rate

    @property
    def end_time(self):
        return self.time_length + self.start_time


class VoiceChangerStream(object):
    def __init__(
            self,
            voice_changer: VoiceChanger,
            sampling_rate: int,
            in_dtype=numpy.float32,
    ):
        self.voice_changer = voice_changer
        self.sampling_rate = sampling_rate
        self.in_dtype = in_dtype
        self._data_stream = []  # type: List[Segment]

    @property
    def vocoder(self):
        return self.voice_changer.vocoder

    def add_wave(self, start_time: float, wave: Wave):
        # validation
        assert wave.sampling_rate == self.sampling_rate
        assert wave.wave.dtype == self.in_dtype

        segment = Segment(start_time=start_time, wave=wave)
        self._data_stream.append(segment)

    def remove_wave(self, end_time: float):
        self._data_stream = list(filter(lambda s: s.end_time > end_time, self._data_stream))

    def convert(self, start_time: float, time_length: float):
        end_time = start_time + time_length
        buffer_list = []
        stream = filter(lambda s: not (end_time < s.start_time or s.end_time < start_time), self._data_stream)

        start_time_buffer = start_time
        remaining_time = time_length
        for segment in stream:
            # padding
            if segment.start_time > start_time_buffer:
                pad = numpy.zeros(
                    shape=int((segment.start_time - start_time_buffer) * self.sampling_rate),
                    dtype=self.in_dtype,
                )
                buffer_list.append(pad)
                start_time_buffer = segment.start_time

            if remaining_time > segment.end_time - start_time_buffer:
                one_time_length = segment.end_time - start_time_buffer
            else:
                one_time_length = remaining_time

            first_index = int((start_time_buffer - segment.start_time) * self.sampling_rate)
            last_index = int(first_index + one_time_length * self.sampling_rate)
            one_buffer = segment.wave.wave[first_index:last_index]
            buffer_list.append(one_buffer)

            start_time_buffer += one_time_length
            remaining_time -= one_time_length

            if start_time_buffer >= end_time:
                break
        else:
            # last padding
            pad = numpy.zeros(shape=int((end_time - start_time_buffer) * self.sampling_rate), dtype=self.in_dtype)
            buffer_list.append(pad)

        buffer = numpy.concatenate(buffer_list)
        print('buffer', len(buffer), flush=True)
        in_wave = Wave(wave=buffer, sampling_rate=self.sampling_rate)
        in_feature = self.vocoder.encode(in_wave)
        out_feature = self.voice_changer.convert_from_acoustic_feature(in_feature)
        return out_feature

    def convert_with_extra_time(self, start_time: float, time_length: float, extra_time: float):
        """
        :param extra_time: 音声変換時に余分に使うデータの時間長。ゼロパディングを防ぐ。
        """
        frame_period = self.vocoder.acoustic_feature_param.frame_period

        start_time -= extra_time
        time_length += extra_time * 2

        extra_feature = self.convert(start_time=start_time, time_length=time_length)

        pad = int(extra_time / (frame_period / 1000))
        feature = AcousticFeature(
            f0=extra_feature.f0[pad:-pad],
            spectrogram=extra_feature.spectrogram[pad:-pad],
            aperiodicity=extra_feature.aperiodicity[pad:-pad],
            mfcc=extra_feature.mfcc[pad:-pad],
            voiced=extra_feature.voiced[pad:-pad],
        )

        out_wave = self.vocoder.decode(
            acoustic_feature=feature,
        )
        return out_wave


class VoiceChangerStreamWrapper(object):
    def __init__(
            self,
            voice_changer_stream: VoiceChangerStream,
            extra_time: float = 0.0
    ):
        self.voice_changer_stream = voice_changer_stream
        self.extra_time = extra_time
        self._current_time = 0

    def convert_next(self, time_length: float):
        out_wave = self.voice_changer_stream.convert_with_extra_time(
            start_time=self._current_time,
            time_length=time_length,
            extra_time=self.extra_time,
        )
        self._current_time += time_length
        return out_wave

    def remove_previous_wave(self):
        self.voice_changer_stream.remove_wave(end_time=self._current_time - self.extra_time)
