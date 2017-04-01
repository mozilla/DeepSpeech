import unicodedata

class STMSegment(object):
    r"""
    Representation of an individual segment in an STM file.
    """
    def __init__(self, stm_line):
        tokens = stm_line.split()
        self._filename    = tokens[0]
        self._channel     = tokens[1]
        self._speaker_id  = tokens[2]
        self._start_time  = float(tokens[3])
        self._stop_time   = float(tokens[4])
        self._labels      = tokens[5]
        self._transcript  = ""
        for token in tokens[6:]:
          self._transcript += token + " "
        self._transcript = unicodedata.normalize("NFKD", self._transcript.strip())  \
                                      .encode("ascii", "ignore")                    \
                                      .decode("ascii", "ignore")

    @property
    def filename(self):
        return self._filename

    @property
    def channel(self):
        return self._channel

    @property
    def speaker_id(self):
        return self._speaker_id

    @property
    def start_time(self):
        return self._start_time

    @property
    def stop_time(self):
        return self._stop_time

    @property
    def labels(self):
        return self._labels

    @property
    def transcript(self):
        return self._transcript

def parse_stm_file(stm_file):
    r"""
    Parses an STM file at ``stm_file`` into a list of :class:`STMSegment`.
    """
    stm_segments = []
    with open(stm_file) as stm_lines:
        for stm_line in stm_lines:
            stmSegment = STMSegment(stm_line)
            if not "ignore_time_segment_in_scoring" == stmSegment.transcript:
                stm_segments.append(stmSegment)
    return stm_segments
