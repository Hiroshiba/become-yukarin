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
            output_sampling_rate: int = None,
    ) -> None:
        if output_sampling_rate is None:
            output_sampling_rate = super_resolution.config.dataset.param.voice_param.sample_rate

        self.acoustic_converter = acoustic_converter
        self.super_resolution = super_resolution
        self.output_sampling_rate = output_sampling_rate

    # def convert_from_wave_path(self, wave_path: str):
    #     w_in = self.acoustic_converter._wave_process(wave_path)
    #     return self.convert_from_wave(w_in)
    #
    # def convert_from_wave(self, wave: Wave):
    #     f_in = self.acoustic_converter._feature_process(wave)
    #     f_high = self.convert_from_acoustic_feature(f_in)
    #     wave = self.vocoder.decode(f_high)
    #     return wave

    def convert_from_acoustic_feature(self, f_in: AcousticFeature):
        f_low = self.acoustic_converter.convert_to_feature(f_in)
        s_high = self.super_resolution.convert(f_low.spectrogram.astype(numpy.float32))
        f_high = self.super_resolution.convert_to_feature(s_high, f_low)
        return f_high


class FeatureSegment(NamedTuple):
    start_time: float
    feature: AcousticFeature
    frame_period: float

    @property
    def time_length(self):
        return len(self.feature.f0) * self.frame_period / 1000

    @property
    def end_time(self):
        return self.time_length + self.start_time


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
            sampling_rate: int,
            frame_period: float,
            in_dtype=numpy.float32,
    ):
        self.sampling_rate = sampling_rate
        self.frame_period = frame_period
        self.in_dtype = in_dtype

        self.voice_changer: VoiceChanger = None
        self.vocoder: Vocoder = None
        self._data_stream = []  # type: List[Segment]
        self._in_feature_stream = []  # type: List[FeatureSegment]
        self._out_feature_stream = []  # type: List[FeatureSegment]

    def add_wave(self, start_time: float, wave: Wave):
        # validation
        assert wave.sampling_rate == self.sampling_rate
        assert wave.wave.dtype == self.in_dtype

        segment = Segment(start_time=start_time, wave=wave)
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

    def pre_convert(self, start_time: float, time_length: float, extra_time: float):
        start_time -= extra_time
        time_length += extra_time * 2

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
        in_wave = Wave(wave=buffer, sampling_rate=self.sampling_rate)
        in_feature = self.vocoder.encode(in_wave)

        pad = int(extra_time / (self.vocoder.acoustic_feature_param.frame_period / 1000))
        in_feature = AcousticFeature(
            f0=in_feature.f0[pad:-pad],
            spectrogram=in_feature.spectrogram[pad:-pad],
            aperiodicity=in_feature.aperiodicity[pad:-pad],
            mfcc=in_feature.mfcc[pad:-pad],
            voiced=in_feature.voiced[pad:-pad],
        )
        return in_feature

    def convert(self, start_time: float, time_length: float, extra_time: float):
        start_time -= extra_time
        time_length += extra_time * 2

        order = self.voice_changer.acoustic_converter.config.dataset.param.acoustic_feature_param.order

        end_time = start_time + time_length
        f0_buffer_list = []
        mfcc_buffer_list = []
        ap_buffer_list = []
        voiced_buffer_list = []
        stream = filter(lambda s: not (end_time < s.start_time or s.end_time < start_time), self._in_feature_stream)

        start_time_buffer = start_time
        remaining_time = time_length
        for segment in stream:
            # padding
            if segment.start_time > start_time_buffer:
                pad_size = int((segment.start_time - start_time_buffer) * 1000 / self.frame_period)
                dims = AcousticFeature.get_sizes(self.sampling_rate, order)

                f0_buffer_list.append(numpy.zeros(shape=[pad_size, 1], dtype=self.in_dtype))
                mfcc_buffer_list.append(numpy.zeros(shape=[pad_size, dims['mfcc']], dtype=self.in_dtype))
                ap_buffer_list.append(numpy.zeros(shape=[pad_size, dims['aperiodicity']], dtype=self.in_dtype))
                voiced_buffer_list.append(numpy.zeros(shape=[pad_size, 1], dtype=numpy.bool))

                start_time_buffer = segment.start_time
            if remaining_time > segment.end_time - start_time_buffer:
                one_time_length = segment.end_time - start_time_buffer
            else:
                one_time_length = remaining_time

            first_index = int((start_time_buffer - segment.start_time) * 1000 / self.frame_period)
            last_index = int(first_index + one_time_length * 1000 / self.frame_period)

            f0_buffer_list.append(segment.feature.f0[first_index:last_index])
            mfcc_buffer_list.append(segment.feature.mfcc[first_index:last_index])
            ap_buffer_list.append(segment.feature.aperiodicity[first_index:last_index])
            voiced_buffer_list.append(segment.feature.voiced[first_index:last_index])

            start_time_buffer += one_time_length
            remaining_time -= one_time_length

            if start_time_buffer >= end_time:
                break
        else:
            # last padding
            pad_size = int((end_time - start_time_buffer) * 1000 / self.frame_period)
            dims = AcousticFeature.get_sizes(self.sampling_rate, order)

            f0_buffer_list.append(numpy.zeros(shape=[pad_size, 1], dtype=self.in_dtype))
            mfcc_buffer_list.append(numpy.zeros(shape=[pad_size, dims['mfcc']], dtype=self.in_dtype))
            ap_buffer_list.append(numpy.zeros(shape=[pad_size, dims['aperiodicity']], dtype=self.in_dtype))
            voiced_buffer_list.append(numpy.zeros(shape=[pad_size, 1], dtype=numpy.bool))

        f0 = numpy.concatenate(f0_buffer_list)
        mfcc = numpy.concatenate(mfcc_buffer_list)
        aperiodicity = numpy.concatenate(ap_buffer_list)
        voiced = numpy.concatenate(voiced_buffer_list)
        in_feature = AcousticFeature(
            f0=f0,
            spectrogram=numpy.nan,
            aperiodicity=aperiodicity,
            mfcc=mfcc,
            voiced=voiced,
        )

        out_feature = self.voice_changer.convert_from_acoustic_feature(in_feature)

        pad = int(extra_time * 1000 / self.frame_period)
        out_feature= AcousticFeature(
            f0=out_feature.f0[pad:-pad],
            spectrogram=out_feature.spectrogram[pad:-pad],
            aperiodicity=out_feature.aperiodicity[pad:-pad],
            mfcc=out_feature.mfcc[pad:-pad],
            voiced=out_feature.voiced[pad:-pad],
        )
        return out_feature

    def post_convert(self, start_time: float, time_length: float):
        end_time = start_time + time_length
        f0_buffer_list = []
        sp_buffer_list = []
        ap_buffer_list = []
        voiced_buffer_list = []
        stream = filter(lambda s: not (end_time < s.start_time or s.end_time < start_time), self._out_feature_stream)

        start_time_buffer = start_time
        remaining_time = time_length
        for segment in stream:
            # padding
            if segment.start_time > start_time_buffer:
                pad_size = int((segment.start_time - start_time_buffer) * 1000 / self.frame_period)
                dims = AcousticFeature.get_sizes(self.sampling_rate, self.vocoder.acoustic_feature_param.order)

                f0_buffer_list.append(numpy.zeros(shape=[pad_size, 1], dtype=self.in_dtype))
                sp_buffer_list.append(numpy.zeros(shape=[pad_size, dims['spectrogram']], dtype=self.in_dtype))
                ap_buffer_list.append(numpy.zeros(shape=[pad_size, dims['aperiodicity']], dtype=self.in_dtype))
                voiced_buffer_list.append(numpy.zeros(shape=[pad_size, 1], dtype=numpy.bool))

                start_time_buffer = segment.start_time

            if remaining_time > segment.end_time - start_time_buffer:
                one_time_length = segment.end_time - start_time_buffer
            else:
                one_time_length = remaining_time

            first_index = int((start_time_buffer - segment.start_time) * 1000 / self.frame_period)
            last_index = int(first_index + one_time_length * 1000 / self.frame_period)

            f0_buffer_list.append(segment.feature.f0[first_index:last_index])
            sp_buffer_list.append(segment.feature.spectrogram[first_index:last_index])
            ap_buffer_list.append(segment.feature.aperiodicity[first_index:last_index])
            voiced_buffer_list.append(segment.feature.voiced[first_index:last_index])

            start_time_buffer += one_time_length
            remaining_time -= one_time_length

            if start_time_buffer >= end_time:
                break
        else:
            # last padding
            pad_size = int((end_time - start_time_buffer) * 1000 / self.frame_period)
            dims = AcousticFeature.get_sizes(self.sampling_rate, self.vocoder.acoustic_feature_param.order)

            f0_buffer_list.append(numpy.zeros(shape=[pad_size, 1], dtype=self.in_dtype))
            sp_buffer_list.append(numpy.zeros(shape=[pad_size, dims['spectrogram']], dtype=self.in_dtype))
            ap_buffer_list.append(numpy.zeros(shape=[pad_size, dims['aperiodicity']], dtype=self.in_dtype))
            voiced_buffer_list.append(numpy.zeros(shape=[pad_size, 1], dtype=self.in_dtype))

        f0 = numpy.concatenate(f0_buffer_list)
        spectrogram = numpy.concatenate(sp_buffer_list)
        aperiodicity = numpy.concatenate(ap_buffer_list)
        voiced = numpy.concatenate(voiced_buffer_list)
        out_feature = AcousticFeature(
            f0=f0,
            spectrogram=spectrogram,
            aperiodicity=aperiodicity,
            mfcc=numpy.nan,
            voiced=voiced,
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
