# https://stackoverflow.com/questions/7833807/get-wav-file-length-or-duration

import wave
import contextlib


def get_duration(fp):
    with contextlib.closing(wave.open(fp, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration
