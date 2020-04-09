
import os
import re
import math
import random
import numpy as np

from multiprocessing import Queue, Process
from .audio import gain_db_to_ratio, max_dbfs, normalize_audio, AUDIO_TYPE_NP, AUDIO_TYPE_PCM, AUDIO_TYPE_OPUS
from .helpers import int_range, float_range, pick_value_from_range, MEGABYTE

SPEC_PARSER = re.compile(r'^([a-z]+)(\[(.*)\])?$')
BUFFER_SIZE = 1 * MEGABYTE


def _enqueue_overlay_samples(sample_source, queue, buffering=BUFFER_SIZE):
    # preventing cyclic import problems
    from .sample_collections import samples_from_source  # pylint: disable=import-outside-toplevel
    samples = samples_from_source(sample_source, buffering=buffering, labeled=False)
    while True:
        for sample in samples:
            queue.put(sample)


class Overlay:
    """See "Overlay augmentation" in TRAINING.rst"""
    def __init__(self, source, p=1.0, snr=3.0, layers=1):
        self.source = source
        self.probability = float(p)
        self.snr = float_range(snr)
        self.layers = int_range(layers)
        self.queue = Queue(max(1, math.floor(self.probability * self.layers[1] * os.cpu_count())))
        self.current_sample = None
        self.enqueue_process = None

    def start(self, buffering=BUFFER_SIZE):
        self.enqueue_process = Process(target=_enqueue_overlay_samples,
                                       args=(self.source, self.queue),
                                       kwargs={'buffering': buffering})
        self.enqueue_process.start()

    def apply(self, sample, clock):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        n_layers = pick_value_from_range(self.layers, clock=clock)
        audio = sample.audio
        overlay_data = np.zeros_like(audio)
        for _ in range(n_layers):
            overlay_offset = 0
            while overlay_offset < len(audio):
                if self.current_sample is None:
                    next_overlay_sample = self.queue.get()
                    next_overlay_sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
                    self.current_sample = next_overlay_sample.audio
                n_required = len(audio) - overlay_offset
                n_current = len(self.current_sample)
                if n_required >= n_current:  # take it completely
                    overlay_data[overlay_offset:overlay_offset + n_current] += self.current_sample
                    overlay_offset += n_current
                    self.current_sample = None
                else:  # take required slice from head and keep tail for next layer or sample
                    overlay_data[overlay_offset:overlay_offset + n_required] += self.current_sample[0:n_required]
                    overlay_offset += n_required
                    self.current_sample = self.current_sample[n_required:]
        snr_db = pick_value_from_range(self.snr, clock=clock)
        orig_dbfs = max_dbfs(audio)
        overlay_gain = orig_dbfs - max_dbfs(overlay_data) - snr_db
        audio += overlay_data * gain_db_to_ratio(overlay_gain)
        sample.audio = normalize_audio(audio, dbfs=orig_dbfs)

    def stop(self):
        if self.enqueue_process is not None:
            self.enqueue_process.terminate()


class Reverb:
    """See "Reverb augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, delay=20.0, decay=10.0):
        self.probability = float(p)
        self.delay = float_range(delay)
        self.decay = float_range(decay)

    def apply(self, sample, clock):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        audio = np.array(sample.audio, dtype=np.float64)
        orig_dbfs = max_dbfs(audio)
        delay = pick_value_from_range(self.delay, clock=clock)
        decay = pick_value_from_range(self.decay, clock=clock)
        decay = gain_db_to_ratio(-decay)
        result = np.copy(audio)
        primes = [17, 19, 23, 29, 31]
        for delay_prime in primes:  # primes to minimize comb filter interference
            layer = np.copy(audio)
            n_delay = math.floor(delay * (delay_prime / primes[0]) * sample.audio_format.rate / 1000.0)
            n_delay = max(16, n_delay)  # 16 samples minimum to avoid performance trap and risk of division by zero
            for w_index in range(0, math.floor(len(audio) / n_delay)):
                w1 = w_index * n_delay
                w2 = (w_index + 1) * n_delay
                width = min(len(audio) - w2, n_delay)  # last window could be smaller
                layer[w2:w2 + width] += decay * layer[w1:w1 + width]
            result += layer
        audio = normalize_audio(result, dbfs=orig_dbfs)
        sample.audio = np.array(audio, dtype=np.float32)


class Resample:
    """See "Resample augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, rate=8000):
        self.probability = float(p)
        self.rate = int_range(rate)

    def apply(self, sample, clock):
        # late binding librosa and its dependencies
        from librosa.core import resample  # pylint: disable=import-outside-toplevel
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        rate = pick_value_from_range(self.rate, clock=clock)
        audio = sample.audio
        orig_len = len(audio)
        audio = np.swapaxes(audio, 0, 1)
        audio = resample(audio, sample.audio_format.rate, rate)
        audio = resample(audio, rate, sample.audio_format.rate)
        audio = np.swapaxes(audio, 0, 1)[0:orig_len]
        sample.audio = audio


class Codec:
    """See "Codec augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, bitrate=3200):
        self.probability = float(p)
        self.bitrate = int_range(bitrate)

    def apply(self, sample, clock):
        bitrate = pick_value_from_range(self.bitrate, clock=clock)
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_PCM)  # decoding to ensure it has to get encoded again
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_OPUS, bitrate=bitrate)  # will get decoded again downstream


class Gaps:
    """See "Gaps augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, n=1, size=50.0):
        self.probability = float(p)
        self.n_gaps = int_range(n)
        self.size = float_range(size)

    def apply(self, sample, clock):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        audio = sample.audio
        n_gaps = pick_value_from_range(self.n_gaps, clock=clock)
        for _ in range(n_gaps):
            size = pick_value_from_range(self.size, clock=clock)
            size = int(size * sample.audio_format.rate / 1000.0)
            size = min(size, len(audio) // 10)  # a gap should never exceed 10 percent of the audio
            offset = random.randint(0, max(0, len(audio) - size - 1))
            audio[offset:offset + size] = 0
        sample.audio = audio


class Volume:
    """See "Volume augmentation" in TRAINING.rst"""
    def __init__(self, p=1.0, dbfs=3.0103):
        self.probability = float(p)
        self.target_dbfs = float_range(dbfs)

    def apply(self, sample, clock):
        sample.change_audio_type(new_audio_type=AUDIO_TYPE_NP)
        target_dbfs = pick_value_from_range(self.target_dbfs, clock=clock)
        sample.audio = normalize_audio(sample.audio, dbfs=target_dbfs)


def parse_augmentation(augmentation_spec):
    """
    Parses an augmentation specification.

    Parameters
    ----------
    augmentation_spec : str
        Augmentation specification like "reverb[delay=20.0,decay=-20]".

    Returns
    -------
    Instance of an augmentation class from util.signal_augmentations.*.
    """
    match = SPEC_PARSER.match(augmentation_spec)
    if not match:
        raise ValueError('Augmentation specification has wrong format')
    cls_name = match.group(1)[0].upper() + match.group(1)[1:]
    if cls_name not in globals():
        raise ValueError('Unknown augmentation: {}'.format(cls_name))
    augmentation_cls = globals()[cls_name]
    parameters = [] if match.group(3) is None else match.group(3).split(',')
    args = []
    kwargs = {}
    for parameter in parameters:
        pair = tuple(list(map(str.strip, (parameter.split('=')))))
        if len(pair) == 1:
            args.append(pair)
        elif len(pair) == 2:
            kwargs[pair[0]] = pair[1]
        else:
            raise ValueError('Unable to parse augmentation value assignment')
    return augmentation_cls(*args, **kwargs)
