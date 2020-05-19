import os
import io
import wave
import math
import tempfile
import collections
import numpy as np

from .helpers import LimitingPool
from collections import namedtuple

AudioFormat = namedtuple('AudioFormat', 'rate channels width')

DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_WIDTH = 2
DEFAULT_FORMAT = AudioFormat(DEFAULT_RATE, DEFAULT_CHANNELS, DEFAULT_WIDTH)

AUDIO_TYPE_NP = 'application/vnd.mozilla.np'
AUDIO_TYPE_PCM = 'application/vnd.mozilla.pcm'
AUDIO_TYPE_WAV = 'audio/wav'
AUDIO_TYPE_OPUS = 'application/vnd.mozilla.opus'
SERIALIZABLE_AUDIO_TYPES = [AUDIO_TYPE_WAV, AUDIO_TYPE_OPUS]
LOADABLE_AUDIO_EXTENSIONS = {'.wav': AUDIO_TYPE_WAV}

OPUS_PCM_LEN_SIZE = 4
OPUS_RATE_SIZE = 4
OPUS_CHANNELS_SIZE = 1
OPUS_WIDTH_SIZE = 1
OPUS_CHUNK_LEN_SIZE = 2


class Sample:
    """
    Represents in-memory audio data of a certain (convertible) representation.

    Attributes
    ----------
    audio_type : str
        See `__init__`.
    audio_format : util.audio.AudioFormat
        See `__init__`.
    audio : binary
        Audio data represented as indicated by `audio_type`
    duration : float
        Audio duration of the sample in seconds
    """
    def __init__(self, audio_type, raw_data, audio_format=None, sample_id=None):
        """
        Parameters
        ----------
        audio_type : str
            Audio data representation type
            Supported types:
                - util.audio.AUDIO_TYPE_OPUS: Memory file representation (BytesIO) of Opus encoded audio
                    wrapped by a custom container format (used in SDBs)
                - util.audio.AUDIO_TYPE_WAV: Memory file representation (BytesIO) of a Wave file
                - util.audio.AUDIO_TYPE_PCM: Binary representation (bytearray) of PCM encoded audio data (Wave file without header)
                - util.audio.AUDIO_TYPE_NP: NumPy representation of audio data (np.float32) - typically used for GPU feeding
        raw_data : binary
            Audio data in the form of the provided representation type (see audio_type).
            For types util.audio.AUDIO_TYPE_OPUS or util.audio.AUDIO_TYPE_WAV data can also be passed as a bytearray.
        audio_format : util.audio.AudioFormat
            Required in case of audio_type = util.audio.AUDIO_TYPE_PCM or util.audio.AUDIO_TYPE_NP,
            as this information cannot be derived from raw audio data.
        sample_id : str
            Tracking ID - should indicate sample's origin as precisely as possible
        """
        self.audio_type = audio_type
        self.audio_format = audio_format
        self.sample_id = sample_id
        if audio_type in SERIALIZABLE_AUDIO_TYPES:
            self.audio = raw_data if isinstance(raw_data, io.BytesIO) else io.BytesIO(raw_data)
            self.duration = read_duration(audio_type, self.audio)
        else:
            self.audio = raw_data
            if self.audio_format is None:
                raise ValueError('For audio type "{}" parameter "audio_format" is mandatory'.format(self.audio_type))
            if audio_type == AUDIO_TYPE_PCM:
                self.duration = get_pcm_duration(len(self.audio), self.audio_format)
            elif audio_type == AUDIO_TYPE_NP:
                self.duration = get_np_duration(len(self.audio), self.audio_format)
            else:
                raise ValueError('Unsupported audio type: {}'.format(self.audio_type))

    def change_audio_type(self, new_audio_type, bitrate=None):
        """
        In-place conversion of audio data into a different representation.

        Parameters
        ----------
        new_audio_type : str
            New audio-type - see `__init__`.
        bitrate : int
            Bitrate to use in case of converting to a lossy audio-type.
        """
        if self.audio_type == new_audio_type:
            return
        if new_audio_type == AUDIO_TYPE_PCM and self.audio_type in SERIALIZABLE_AUDIO_TYPES:
            self.audio_format, audio = read_audio(self.audio_type, self.audio)
            self.audio.close()
            self.audio = audio
        elif new_audio_type == AUDIO_TYPE_PCM and self.audio_type == AUDIO_TYPE_NP:
            self.audio = np_to_pcm(self.audio, self.audio_format)
        elif new_audio_type == AUDIO_TYPE_NP:
            self.change_audio_type(AUDIO_TYPE_PCM)
            self.audio = pcm_to_np(self.audio, self.audio_format)
        elif new_audio_type in SERIALIZABLE_AUDIO_TYPES:
            self.change_audio_type(AUDIO_TYPE_PCM)
            audio_bytes = io.BytesIO()
            write_audio(new_audio_type, audio_bytes, self.audio, audio_format=self.audio_format, bitrate=bitrate)
            audio_bytes.seek(0)
            self.audio = audio_bytes
        else:
            raise RuntimeError('Changing audio representation type from "{}" to "{}" not supported'
                               .format(self.audio_type, new_audio_type))
        self.audio_type = new_audio_type


def _change_audio_type(sample_and_audio_type):
    sample, audio_type, bitrate = sample_and_audio_type
    sample.change_audio_type(audio_type, bitrate=bitrate)
    return sample


def change_audio_types(samples, audio_type=AUDIO_TYPE_PCM, bitrate=None, processes=None, process_ahead=None):
    with LimitingPool(processes=processes, process_ahead=process_ahead) as pool:
        yield from pool.imap(_change_audio_type, map(lambda s: (s, audio_type, bitrate), samples))


def get_audio_type_from_extension(ext):
    if ext in LOADABLE_AUDIO_EXTENSIONS:
        return LOADABLE_AUDIO_EXTENSIONS[ext]
    return None


def read_audio_format_from_wav_file(wav_file):
    return AudioFormat(wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth())


def get_num_samples(pcm_buffer_size, audio_format=DEFAULT_FORMAT):
    return pcm_buffer_size // (audio_format.channels * audio_format.width)


def get_pcm_duration(pcm_buffer_size, audio_format=DEFAULT_FORMAT):
    """Calculates duration in seconds of a binary PCM buffer (typically read from a WAV file)"""
    return get_num_samples(pcm_buffer_size, audio_format) / audio_format.rate


def get_np_duration(np_len, audio_format=DEFAULT_FORMAT):
    """Calculates duration in seconds of NumPy audio data"""
    return np_len / audio_format.rate


def convert_audio(src_audio_path, dst_audio_path, file_type=None, audio_format=DEFAULT_FORMAT):
    import sox
    transformer = sox.Transformer()
    transformer.set_output_format(file_type=file_type,
                                  rate=audio_format.rate,
                                  channels=audio_format.channels,
                                  bits=audio_format.width * 8)
    transformer.build(src_audio_path, dst_audio_path)


class AudioFile:
    def __init__(self, audio_path, as_path=False, audio_format=DEFAULT_FORMAT):
        self.audio_path = audio_path
        self.audio_format = audio_format
        self.as_path = as_path
        self.open_file = None
        self.tmp_file_path = None

    def __enter__(self):
        if self.audio_path.endswith('.wav'):
            self.open_file = wave.open(self.audio_path, 'r')
            if read_audio_format_from_wav_file(self.open_file) == self.audio_format:
                if self.as_path:
                    self.open_file.close()
                    return self.audio_path
                return self.open_file
            self.open_file.close()
        _, self.tmp_file_path = tempfile.mkstemp(suffix='.wav')
        convert_audio(self.audio_path, self.tmp_file_path, file_type='wav', audio_format=self.audio_format)
        if self.as_path:
            return self.tmp_file_path
        self.open_file = wave.open(self.tmp_file_path, 'r')
        return self.open_file

    def __exit__(self, *args):
        if not self.as_path:
            self.open_file.close()
        if self.tmp_file_path is not None:
            os.remove(self.tmp_file_path)


def read_frames(wav_file, frame_duration_ms=30, yield_remainder=False):
    audio_format = read_audio_format_from_wav_file(wav_file)
    frame_size = int(audio_format.rate * (frame_duration_ms / 1000.0))
    while True:
        try:
            data = wav_file.readframes(frame_size)
            if not yield_remainder and get_pcm_duration(len(data), audio_format) * 1000 < frame_duration_ms:
                break
            yield data
        except EOFError:
            break


def read_frames_from_file(audio_path, audio_format=DEFAULT_FORMAT, frame_duration_ms=30, yield_remainder=False):
    with AudioFile(audio_path, audio_format=audio_format) as wav_file:
        for frame in read_frames(wav_file, frame_duration_ms=frame_duration_ms, yield_remainder=yield_remainder):
            yield frame


def vad_split(audio_frames,
              audio_format=DEFAULT_FORMAT,
              num_padding_frames=10,
              threshold=0.5,
              aggressiveness=3):
    from webrtcvad import Vad  # pylint: disable=import-outside-toplevel
    if audio_format.channels != 1:
        raise ValueError('VAD-splitting requires mono samples')
    if audio_format.width != 2:
        raise ValueError('VAD-splitting requires 16 bit samples')
    if audio_format.rate not in [8000, 16000, 32000, 48000]:
        raise ValueError('VAD-splitting only supported for sample rates 8000, 16000, 32000, or 48000')
    if aggressiveness not in [0, 1, 2, 3]:
        raise ValueError('VAD-splitting aggressiveness mode has to be one of 0, 1, 2, or 3')
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    vad = Vad(int(aggressiveness))
    voiced_frames = []
    frame_duration_ms = 0
    frame_index = 0
    for frame_index, frame in enumerate(audio_frames):
        frame_duration_ms = get_pcm_duration(len(frame), audio_format) * 1000
        if int(frame_duration_ms) not in [10, 20, 30]:
            raise ValueError('VAD-splitting only supported for frame durations 10, 20, or 30 ms')
        is_speech = vad.is_speech(frame, audio_format.rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > threshold * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > threshold * ring_buffer.maxlen:
                triggered = False
                yield b''.join(voiced_frames), \
                      frame_duration_ms * max(0, frame_index - len(voiced_frames)), \
                      frame_duration_ms * frame_index
                ring_buffer.clear()
                voiced_frames = []
    if len(voiced_frames) > 0:
        yield b''.join(voiced_frames), \
              frame_duration_ms * (frame_index - len(voiced_frames)), \
              frame_duration_ms * (frame_index + 1)


def pack_number(n, num_bytes):
    return n.to_bytes(num_bytes, 'big', signed=False)


def unpack_number(data):
    return int.from_bytes(data, 'big', signed=False)


def get_opus_frame_size(rate):
    return 60 * rate // 1000


def write_opus(opus_file, audio_data, audio_format=DEFAULT_FORMAT, bitrate=None):
    frame_size = get_opus_frame_size(audio_format.rate)
    import opuslib  # pylint: disable=import-outside-toplevel
    encoder = opuslib.Encoder(audio_format.rate, audio_format.channels, 'audio')
    if bitrate is not None:
        encoder.bitrate = bitrate
    chunk_size = frame_size * audio_format.channels * audio_format.width
    opus_file.write(pack_number(len(audio_data), OPUS_PCM_LEN_SIZE))
    opus_file.write(pack_number(audio_format.rate, OPUS_RATE_SIZE))
    opus_file.write(pack_number(audio_format.channels, OPUS_CHANNELS_SIZE))
    opus_file.write(pack_number(audio_format.width, OPUS_WIDTH_SIZE))
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        # Preventing non-deterministic encoding results from uninitialized remainder of the encoder buffer
        if len(chunk) < chunk_size:
            chunk = chunk + b'\0' * (chunk_size - len(chunk))
        encoded = encoder.encode(chunk, frame_size)
        opus_file.write(pack_number(len(encoded), OPUS_CHUNK_LEN_SIZE))
        opus_file.write(encoded)


def read_opus_header(opus_file):
    opus_file.seek(0)
    pcm_buffer_size = unpack_number(opus_file.read(OPUS_PCM_LEN_SIZE))
    rate = unpack_number(opus_file.read(OPUS_RATE_SIZE))
    channels = unpack_number(opus_file.read(OPUS_CHANNELS_SIZE))
    width = unpack_number(opus_file.read(OPUS_WIDTH_SIZE))
    return pcm_buffer_size, AudioFormat(rate, channels, width)


def read_opus(opus_file):
    pcm_buffer_size, audio_format = read_opus_header(opus_file)
    frame_size = get_opus_frame_size(audio_format.rate)
    import opuslib  # pylint: disable=import-outside-toplevel
    decoder = opuslib.Decoder(audio_format.rate, audio_format.channels)
    audio_data = bytearray()
    while len(audio_data) < pcm_buffer_size:
        chunk_len = unpack_number(opus_file.read(OPUS_CHUNK_LEN_SIZE))
        chunk = opus_file.read(chunk_len)
        decoded = decoder.decode(chunk, frame_size)
        audio_data.extend(decoded)
    audio_data = audio_data[:pcm_buffer_size]
    return audio_format, bytes(audio_data)


def write_wav(wav_file, pcm_data, audio_format=DEFAULT_FORMAT):
    with wave.open(wav_file, 'wb') as wav_file_writer:
        wav_file_writer.setframerate(audio_format.rate)
        wav_file_writer.setnchannels(audio_format.channels)
        wav_file_writer.setsampwidth(audio_format.width)
        wav_file_writer.writeframes(pcm_data)


def read_wav(wav_file):
    wav_file.seek(0)
    with wave.open(wav_file, 'rb') as wav_file_reader:
        audio_format = read_audio_format_from_wav_file(wav_file_reader)
        pcm_data = wav_file_reader.readframes(wav_file_reader.getnframes())
        return audio_format, pcm_data


def read_audio(audio_type, audio_file):
    if audio_type == AUDIO_TYPE_WAV:
        return read_wav(audio_file)
    if audio_type == AUDIO_TYPE_OPUS:
        return read_opus(audio_file)
    raise ValueError('Unsupported audio type: {}'.format(audio_type))


def write_audio(audio_type, audio_file, pcm_data, audio_format=DEFAULT_FORMAT, bitrate=None):
    if audio_type == AUDIO_TYPE_WAV:
        return write_wav(audio_file, pcm_data, audio_format=audio_format)
    if audio_type == AUDIO_TYPE_OPUS:
        return write_opus(audio_file, pcm_data, audio_format=audio_format, bitrate=bitrate)
    raise ValueError('Unsupported audio type: {}'.format(audio_type))


def read_wav_duration(wav_file):
    wav_file.seek(0)
    with wave.open(wav_file, 'rb') as wav_file_reader:
        return wav_file_reader.getnframes() / wav_file_reader.getframerate()


def read_opus_duration(opus_file):
    pcm_buffer_size, audio_format = read_opus_header(opus_file)
    return get_pcm_duration(pcm_buffer_size, audio_format)


def read_duration(audio_type, audio_file):
    if audio_type == AUDIO_TYPE_WAV:
        return read_wav_duration(audio_file)
    if audio_type == AUDIO_TYPE_OPUS:
        return read_opus_duration(audio_file)
    raise ValueError('Unsupported audio type: {}'.format(audio_type))


def get_dtype(audio_format):
    if audio_format.width not in [1, 2, 4]:
        raise ValueError('Unsupported sample width: {}'.format(audio_format.width))
    return [None, np.int8, np.int16, None, np.int32][audio_format.width]


def pcm_to_np(pcm_data, audio_format=DEFAULT_FORMAT):
    assert audio_format.channels == 1  # only mono supported for now
    dtype = get_dtype(audio_format)
    samples = np.frombuffer(pcm_data, dtype=dtype)
    samples = samples.astype(np.float32) / np.iinfo(dtype).max
    return np.expand_dims(samples, axis=1)


def np_to_pcm(np_data, audio_format=DEFAULT_FORMAT):
    assert audio_format.channels == 1  # only mono supported for now
    dtype = get_dtype(audio_format)
    np_data = np_data.squeeze()
    np_data = np_data * np.iinfo(dtype).max
    np_data = np_data.astype(dtype)
    return np_data.tobytes()


def rms_to_dbfs(rms):
    return 20.0 * math.log10(max(1e-16, rms)) + 3.0103


def max_dbfs(sample_data):
    # Peak dBFS based on the maximum energy sample. Will prevent overdrive if used for normalization.
    return rms_to_dbfs(max(abs(np.min(sample_data)), abs(np.max(sample_data))))


def mean_dbfs(sample_data):
    return rms_to_dbfs(math.sqrt(np.mean(np.square(sample_data, dtype=np.float64))))


def gain_db_to_ratio(gain_db):
    return math.pow(10.0, gain_db / 20.0)


def normalize_audio(sample_data, dbfs=3.0103):
    return np.maximum(np.minimum(sample_data * gain_db_to_ratio(dbfs - max_dbfs(sample_data)), 1.0), -1.0)
