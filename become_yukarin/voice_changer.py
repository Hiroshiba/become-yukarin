from abc import ABCMeta, abstractproperty, abstractmethod
from typing import List, Callable, Any
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
            output_sampling_rate: int = None,
    ) -> None:
        if output_sampling_rate is None:
            output_sampling_rate = super_resolution.config.dataset.param.voice_param.sample_rate

        self.acoustic_converter = acoustic_converter
        self.super_resolution = super_resolution
        self.output_sampling_rate = output_sampling_rate

    def convert_from_acoustic_feature(self, f_in: AcousticFeature):
        f_low = self.acoustic_converter.convert_to_feature(f_in)
        s_high = self.super_resolution.convert(f_low.spectrogram.astype(numpy.float32))
        f_high = self.super_resolution.convert_to_feature(s_high, f_low)
        return f_high


class BaseSegment(ABCMeta):
    start_time: float

    @property
    @abstractmethod
    def time_length(self) -> float:
        pass

    @property
    @abstractmethod
    def end_time(self) -> float:
        pass


class FeatureSegment(NamedTuple, BaseSegment):
    start_time: float
    feature: AcousticFeature
    frame_period: float

    @property
    def time_length(self):
        return len(self.feature.f0) * self.frame_period / 1000

    @property
    def end_time(self):
        return self.time_length + self.start_time


class WaveSegment(NamedTuple, BaseSegment):
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
            sampling_rate: int,
            frame_period: float,
            in_dtype=numpy.float32,
    ):
        self.sampling_rate = sampling_rate
        self.frame_period = frame_period
        self.in_dtype = in_dtype

        self.voice_changer: VoiceChanger = None
        self.vocoder: Vocoder = None
        self._data_stream = []  # type: List[WaveSegment]
        self._in_feature_stream = []  # type: List[FeatureSegment]
        self._out_feature_stream = []  # type: List[FeatureSegment]

    def add_wave(self, start_time: float, wave: Wave):
        # validation
        assert wave.sampling_rate == self.sampling_rate
        assert wave.wave.dtype == self.in_dtype

        segment = WaveSegment(start_time=start_time, wave=wave)
        self._data_stream.append(segment)

    def add_in_feature(self, start_time: float, feature: AcousticFeature, frame_period: float):
        # validation
        assert frame_period == self.frame_period
        assert feature.f0.dtype == self.in_dtype

        segment = FeatureSegment(start_time=start_time, feature=feature, frame_period=self.frame_period)
        self._in_feature_stream.append(segment)

    def add_out_feature(self, start_time: float, feature: AcousticFeature, frame_period: float):
        # validation
        assert frame_period == self.frame_period

        segment = FeatureSegment(start_time=start_time, feature=feature, frame_period=self.frame_period)
        self._out_feature_stream.append(segment)

    def remove(self, end_time: float):
        self._data_stream = list(filter(lambda s: s.end_time > end_time, self._data_stream))
        self._in_feature_stream = list(filter(lambda s: s.end_time > end_time, self._in_feature_stream))
        self._out_feature_stream = list(filter(lambda s: s.end_time > end_time, self._out_feature_stream))

    @staticmethod
    def fetch(
            start_time: float,
            time_length: float,
            data_stream: List[BaseSegment],
            rate: float,
            pad_function: Callable[[int], Any],
            pick_function: Callable[[Any, int, int], Any],
            concat_function: Callable[[List], Any],
            extra_time: float = 0,
    ):
        start_time -= extra_time
        time_length += extra_time * 2

        end_time = start_time + time_length
        buffer_list = []
        stream = filter(lambda s: not (end_time < s.start_time or s.end_time < start_time), data_stream)

        start_time_buffer = start_time
        remaining_time = time_length
        for segment in stream:
            # padding
            if segment.start_time > start_time_buffer:
                length = int((segment.start_time - start_time_buffer) * rate)
                pad = pad_function(length)
                buffer_list.append(pad)
                start_time_buffer = segment.start_time

            if remaining_time > segment.end_time - start_time_buffer:
                one_time_length = segment.end_time - start_time_buffer
            else:
                one_time_length = remaining_time

            first_index = int((start_time_buffer - segment.start_time) * rate)
            last_index = int(first_index + one_time_length * rate)
            one_buffer = pick_function(segment, first_index, last_index)
            buffer_list.append(one_buffer)

            start_time_buffer += one_time_length
            remaining_time -= one_time_length

            if start_time_buffer >= end_time:
                break
        else:
            # last padding
            length = int((end_time - start_time_buffer) * rate)
            pad = pad_function(length)
            buffer_list.append(pad)

        buffer = concat_function(buffer_list)
        return buffer

    def pre_convert(self, start_time: float, time_length: float, extra_time: float):
        wave = self.fetch(
            start_time=start_time,
            time_length=time_length,
            extra_time=extra_time,
            data_stream=self._data_stream,
            rate=self.sampling_rate,
            pad_function=lambda length: numpy.zeros(shape=length, dtype=self.in_dtype),
            pick_function=lambda segment, first, last: segment.wave.wave[first:last],
            concat_function=numpy.concatenate,
        )
        in_wave = Wave(wave=wave, sampling_rate=self.sampling_rate)
        in_feature = self.vocoder.encode(in_wave)

        pad = int(extra_time / (self.vocoder.acoustic_feature_param.frame_period / 1000))
        in_feature = in_feature.pick(pad, -pad)
        return in_feature

    def convert(self, start_time: float, time_length: float, extra_time: float):
        order = self.voice_changer.acoustic_converter.config.dataset.param.acoustic_feature_param.order
        sizes = AcousticFeature.get_sizes(sampling_rate=self.sampling_rate, order=order)
        keys = ['f0', 'aperiodicity', 'mfcc', 'voiced']
        in_feature = self.fetch(
            start_time=start_time,
            time_length=time_length,
            extra_time=extra_time,
            data_stream=self._in_feature_stream,
            rate=1000 / self.frame_period,
            pad_function=lambda length: AcousticFeature.silent(length, sizes=sizes, keys=keys),
            pick_function=lambda segment, first, last: segment.feature.pick(first, last),
            concat_function=lambda buffers: AcousticFeature.concatenate(buffers, keys=keys),
        )
        out_feature = self.voice_changer.convert_from_acoustic_feature(in_feature)

        pad = int(extra_time * 1000 / self.frame_period)
        out_feature = out_feature.pick(pad, -pad)
        return out_feature

    def post_convert(self, start_time: float, time_length: float):
        order = self.voice_changer.acoustic_converter.config.dataset.param.acoustic_feature_param.order
        sizes = AcousticFeature.get_sizes(sampling_rate=self.sampling_rate, order=order)
        keys = ['f0', 'aperiodicity', 'spectrogram', 'voiced']
        out_feature = self.fetch(
            start_time=start_time,
            time_length=time_length,
            data_stream=self._out_feature_stream,
            rate=1000 / self.frame_period,
            pad_function=lambda length: AcousticFeature.silent(length, sizes=sizes, keys=keys),
            pick_function=lambda segment, first, last: segment.feature.pick(first, last),
            concat_function=lambda buffers: AcousticFeature.concatenate(buffers, keys=keys),
        )

        out_wave = self.vocoder.decode(
            acoustic_feature=out_feature,
        )
        return out_wave


class VoiceChangerStreamWrapper(object):
    def __init__(
            self,
            voice_changer_stream: VoiceChangerStream,
            extra_time_pre: float = 0.0,
            extra_time: float = 0.0,
    ):
        self.voice_changer_stream = voice_changer_stream
        self.extra_time_pre = extra_time_pre
        self.extra_time = extra_time
        self._current_time_pre = 0
        self._current_time = 0
        self._current_time_post = 0

    def pre_convert_next(self, time_length: float):
        in_feature = self.voice_changer_stream.pre_convert(
            start_time=self._current_time_pre,
            time_length=time_length,
            extra_time=self.extra_time_pre,
        )
        self._current_time_pre += time_length
        return in_feature

    def convert_next(self, time_length: float):
        out_feature = self.voice_changer_stream.convert(
            start_time=self._current_time,
            time_length=time_length,
            extra_time=self.extra_time,
        )
        self._current_time += time_length
        return out_feature

    def post_convert_next(self, time_length: float):
        out_wave = self.voice_changer_stream.post_convert(
            start_time=self._current_time_post,
            time_length=time_length,
        )
        self._current_time_post += time_length
        return out_wave

    def remove_previous(self):
        end_time = min(
            self._current_time_pre - self.extra_time_pre,
            self._current_time - self.extra_time,
            self._current_time_post,
        )
        self.voice_changer_stream.remove(end_time=end_time)
