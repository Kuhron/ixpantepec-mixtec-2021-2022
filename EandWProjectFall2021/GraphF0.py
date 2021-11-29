import numpy as np
import matplotlib.pyplot as plt
import re
import string



class F0Statistic:
    def __init__(self, fmin, fmax, fmean, fstd):
        self.fmin = fmin
        self.fmax = fmax
        self.fmean = fmean
        self.fstd = fstd


class Interval:
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end
        self.f0_statistics = []
        self.whistled = "w" in self.label
        self.nasalized = "n" in self.label
        self.glottalized = "'" in self.label
        self.tonemes = [("w" if self.whistled else "") + toneme for toneme in get_tonemes_from_str(self.label)]

    def add_f0_statistic(self, f0_statistic):
        assert type(f0_statistic) is F0Statistic, type(f0_statistic)
        self.f0_statistics.append(f0_statistic)


def get_tonemes_from_str(s):
    # return a list of toneme labels (L M or H)
    tone_diacritics = {
        "\u0301": "H",
        "\u0304": "M",
        "\u0300": "L",
        "\u0302": "HL",
    }
    alpha = string.ascii_uppercase + string.ascii_lowercase + "'"
    s = [x for x in s if x not in alpha]  # try to isolate the tones
    tonemes = []
    for x in s:
        if x in tone_diacritics:
            tonemes.append(tone_diacritics[x])
        else:
            raise ValueError(f"unrecognized tone character: {x.encode('unicode_escape')}")
            # should raise for unknown symbol
    return tonemes


def get_interval_from_line(s):
    INTERVAL_LINE_RX = "^interval #(?P<number>\d+)\tlabel: (?P<label>[^\t]*)\tstart: (?P<start>[\d.]+)\tend: (?P<end>[\d.]+)$"

    match = re.match(INTERVAL_LINE_RX, s)
    if match is None:
        raise Exception(f"match is None for line: {repr(s)}")
    label = match["label"]
    start = float(match["start"])
    end = float(match["end"])
    interval = Interval(label, start, end)
    return interval


def get_f0_statistic_from_line(s):
    F0_LINE_RX = "^step #(?P<number>\d+)\/(?P<n_steps>\d+)\tfMin: (?P<fmin>[\d.]+|--undefined--)\tfMax: (?P<fmax>[\d.]+|--undefined--)\tfMean: (?P<fmean>[\d.]+|--undefined--)\tstdev: (?P<fstd>[\d.]+|--undefined--)$"

    match = re.match(F0_LINE_RX, s)
    if match is None:
        raise Exception(f"match is None for line: {repr(s)}")
    fmin = float(match["fmin"]) if match["fmin"] != "--undefined--" else None
    fmax = float(match["fmax"]) if match["fmax"] != "--undefined--" else None
    fmean = float(match["fmean"]) if match["fmean"] != "--undefined--" else None
    fstd = float(match["fstd"]) if match["fstd"] != "--undefined--" else None
    f0_statistic = F0Statistic(fmin, fmax, fmean, fstd)
    return f0_statistic


def get_intervals_from_file(fp):
    with open(fp, encoding="utf-16") as f:
        lines = f.readlines()
    intervals = []
    current_interval = None
    for l in lines:
        l = l.strip()
        if l.startswith("interval"):
            new_interval = get_interval_from_line(l)
            # bank this interval
            if current_interval is not None:
                intervals.append(current_interval)
            current_interval = new_interval
        elif l.startswith("step"):
            f0_statistic = get_f0_statistic_from_line(l)
            # add it to existing interval
            assert current_interval is not None
            current_interval.add_f0_statistic(f0_statistic)
        else:
            print(f"ignoring line: {l}")
    return intervals


if __name__ == "__main__":
    f0_fp = "PitchStats/MKS-20211004-Space_Time_Pronouns_1-M-F0-10ms.txt"
    intervals = get_intervals_from_file(f0_fp)

    intervals_by_toneme = {}
    for interval in intervals:
        tonemes = interval.tonemes
        # for now, assign all f0 statistics in the interval to all tonemes in the interval (i.e. don't try saying the first N steps were the first toneme and the next ones were the next toneme, unsure where to draw the boundary)
        for toneme in tonemes:
            if toneme not in intervals_by_toneme:
                intervals_by_toneme[toneme] = []
            intervals_by_toneme[toneme].append(interval)

    for toneme, intervals in intervals_by_toneme.items():
        fmeans = []
        for interval in intervals:
            for f0 in interval.f0_statistics:
                fmean = f0.fmean
                if fmean is not None:
                    fmeans.append(fmean)
        plt.gcf().clear()
        plt.hist(fmeans, bins=100)
        plt.title(toneme)
        plt.show()
