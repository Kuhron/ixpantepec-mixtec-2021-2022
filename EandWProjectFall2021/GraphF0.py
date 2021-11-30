import numpy as np
import matplotlib.pyplot as plt
import re
import string
import math



class F0Statistic:
    def __init__(self, fmin, fmax, fmean, fstd, start, end, parent_interval):
        assert fmin is None or fmin > 0, fmin
        assert fmax is None or fmax > 0, fmax
        assert fmean is None or fmean > 0, fmean
        assert fstd is None or fstd > 0, fstd
        self.fmin = fmin
        self.fmax = fmax
        self.fmean = fmean
        self.fstd = fstd
        self.start = start
        self.midpoint = (start + end) / 2
        self.end = end
        assert type(parent_interval) is Interval
        self.parent_interval = parent_interval


class Interval:
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.midpoint = (start + end) / 2
        self.end = end
        self.f0_statistics = []
        self.whistled = "w" in self.label
        self.nasalized = "n" in self.label
        self.glottalized = "'" in self.label
        self.toneme_level = "".join(get_tonemes_from_str(self.label))
        self.toneme = ("w" if self.whistled else "") + self.toneme_level

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
    lst = [x for x in s if x not in alpha]  # try to isolate the tones
    tonemes = []
    for x in lst:
        if x in tone_diacritics:
            tonemes.append(tone_diacritics[x])
        else:
            raise ValueError(f"unrecognized tone character: {x.encode('unicode_escape')}")
            # should raise for unknown symbol
    if len(tonemes) == 0:
        raise ValueError(f"no tone symbols found in string: {s.encode('unicode_escape')}")
    return tonemes


def get_interval_from_line(s):
    INTERVAL_LINE_RX = "^interval #(?P<number>\d+?)\tlabel: (?P<label>[^\t]*?)\tstart: (?P<start>[\d.]+?)\tend: (?P<end>[\d.]+?)$"

    match = re.match(INTERVAL_LINE_RX, s)
    if match is None:
        raise Exception(f"match is None for line: {repr(s)}")
    label = match["label"]
    start = float(match["start"])
    end = float(match["end"])
    interval = Interval(label, start, end)
    return interval


def get_f0_statistic_from_line(s, timestep_ms, parent_interval):
    F0_LINE_RX = "^step #(?P<step_number>\d+?)\/(?P<n_steps>\d+?)\tfMin: (?P<fmin>[\d.]+?|--undefined--)\tfMax: (?P<fmax>[\d.]+?|--undefined--)\tfMean: (?P<fmean>[\d.]+?|--undefined--)\tstdev: (?P<fstd>[\d.]+?|--undefined--)$"

    match = re.match(F0_LINE_RX, s)
    if match is None:
        raise Exception(f"match is None for line: {repr(s)}")
    fmin = float(match["fmin"]) if match["fmin"] != "--undefined--" else None
    fmax = float(match["fmax"]) if match["fmax"] != "--undefined--" else None
    fmean = float(match["fmean"]) if match["fmean"] != "--undefined--" else None
    fstd = float(match["fstd"]) if match["fstd"] != "--undefined--" else None
    step_number = int(match["step_number"])
    start = parent_interval.start + step_number * timestep_ms/1000
    end = start + timestep_ms/1000

    # sometimes the stdev is undefined despite there being different fmin and fmax (n > 1) so idk how that happens
    # freqs = [fmin, fmax, fmean, fstd]
    # frequencies_are_none = [x is None for x in freqs]
    # assert all(frequencies_are_none) or not any(frequencies_are_none), f"inconsistent Noneness: {freqs} in f0 starting at {start} seconds"
    # if all(frequencies_are_none):
    #     return None
    # else:
    #     f0_statistic = F0Statistic(fmin, fmax, fmean, fstd, start, end, parent_interval)
    #     return f0_statistic

    f0_statistic = F0Statistic(fmin, fmax, fmean, fstd, start, end, parent_interval)
    return f0_statistic


def get_intervals_from_file(fp):
    f0_timestep_ms = int(re.search("F0-(?P<time>\d+)ms.txt$", fp)["time"])
    with open(fp, encoding="utf-16") as f:
        lines = f.readlines()
    intervals = []
    current_interval = None
    for l in lines:
        l = l.strip()
        if l.startswith("interval"):
            try:
                new_interval = get_interval_from_line(l)
            except:
                print(f"\nError encountered in line: {l}\n")
                raise
            # bank this interval
            if current_interval is not None:
                intervals.append(current_interval)
            current_interval = new_interval
        elif l.startswith("step"):
            try:
                f0_statistic = get_f0_statistic_from_line(l, f0_timestep_ms, current_interval)
            except:
                print(f"\nError encountered in line: {l}\n")
                raise
            # add it to existing interval
            assert current_interval is not None
            if f0_statistic is not None:
                current_interval.add_f0_statistic(f0_statistic)
        else:
            print(f"ignoring line: {l}")
    return intervals


def get_intervals_from_files(fps):
    res = []
    for fp in fps:
        res += get_intervals_from_file(fp)
    return res


def get_intervals_by_toneme_dict(intervals):
    intervals_by_toneme = {}
    for interval in intervals:
        tonemes = list(interval.toneme)
        # for now, assign all f0 statistics in the interval to all tonemes in the interval (i.e. don't try saying the first N steps were the first toneme and the next ones were the next toneme, unsure where to draw the boundary)
        for toneme in tonemes:
            if toneme not in intervals_by_toneme:
                intervals_by_toneme[toneme] = []
            intervals_by_toneme[toneme].append(interval)
    return intervals_by_toneme


def plot_hist_fmeans(intervals_by_toneme):
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


def plot_fmeans_color_coded(intervals):
    fmeans = [f0.fmean for interval in intervals for f0 in interval.f0_statistics]
    fmins = [f0.fmin for interval in intervals for f0 in interval.f0_statistics]
    fmaxs = [f0.fmax for interval in intervals for f0 in interval.f0_statistics]
    color_dict = {  # https://matplotlib.org/stable/gallery/color/named_colors.html
        "H": "gold", "M": "red", "L": "blue", 
        "HH": "gold", "MM": "red", "LL": "blue", 
        "HL": "green", "HM": "darkorange", "ML": "purple",
        "LH": "lime", "MH": "orange", "LM": "darkorchid",
        "MLM": "purple",
    }
    times = [f0.midpoint for interval in intervals for f0 in interval.f0_statistics]
    toneme_levels = [interval.toneme_level for interval in intervals for f0 in interval.f0_statistics]
    color_series = [color_dict[toneme] for toneme in toneme_levels]
    for series in [fmeans, fmins, fmaxs]:
        plt.scatter(times, series, c=color_series)
    transform_axis_to_logarithmic_with_notes(plt.gca())
    # plt.legend(color_dict)
    plt.show()


def get_midi_note_from_hz(hz):
    # A4 = 440 Hz, C4 = MIDI 60, so A4 = 440 Hz = MIDI 69
    assert hz > 0, hz
    ratio = hz / 440
    log2_ratio = math.log2(ratio)
    semitones_diff = 12 * log2_ratio
    return 69 + semitones_diff


def get_hz_from_midi_note(n):
    # A4 = 440 Hz, C4 = MIDI 60, so A4 = 440 Hz = MIDI 69
    semitones_diff = n - 69
    log2_ratio = semitones_diff / 12
    ratio = 2 ** log2_ratio
    return 440 * ratio


def get_note_symbol_from_midi_note(mn):
    # A4 = 440 Hz, C4 = MIDI 60, so A4 = 440 Hz = MIDI 69
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = mn // 12 - 1
    index = mn % 12
    name = names[index]
    return f"{name}{octave}"


def transform_axis_to_logarithmic_with_notes(ax):
    ax.set_yscale("log")
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ymin = max(1e-9, ymin)  # linear y axis will sometimes create negative values in the padding area
    midi_note_min = math.floor(get_midi_note_from_hz(ymin))
    midi_note_max = math.ceil(get_midi_note_from_hz(ymax))
    yticks_midi_notes = list(range(midi_note_min, midi_note_max + 1))
    yticks_hz = [get_hz_from_midi_note(mn) for mn in yticks_midi_notes]
    yticks_note_symbols = [get_note_symbol_from_midi_note(mn) for mn in yticks_midi_notes]
    yticks_str = []
    for midi_note, hz, symbol in zip(yticks_midi_notes, yticks_hz, yticks_note_symbols):
        s = f"{symbol} : {round(hz)} Hz"
        # s = f"{symbol} : #{midi_note} : {int(hz)} Hz"
        yticks_str.append(s)
    ax.set_yticks(yticks_hz)
    ax.set_yticklabels(yticks_str)
    ax.hlines(yticks_hz, linestyles="dashed", xmin=xmin, xmax=xmax, alpha=0.4)


if __name__ == "__main__":
    f0_fps = [
        "PitchStats/MKS-20211004-Space_Time_Pronouns_1-M-F0-10ms.txt",
        "PitchStats/MKS-20211020-Plants_and_Animals_4-M-F0-10ms.txt",
    ]
    intervals = get_intervals_from_files(f0_fps)

    # intervals_by_toneme = get_intervals_by_toneme_dict(intervals)
    # plot_hist_fmeans(intervals_by_toneme)
    # plot_fmeans_color_coded(intervals)

    intervals_high_nasal = [x for x in intervals if x.nasalized and x.toneme == "H"]
    fmeans = [f0.fmean for interval in intervals_high_nasal for f0 in interval.f0_statistics if f0.fmean is not None]
    print(fmeans)
    # plt.hist(fmeans, bins=100)
    # plt.show()

    intervals_high_preceded_by_low = [interval for previous_interval, interval in zip(intervals[:-1], intervals[1:]) if previous_interval.toneme == "L" and interval.toneme == "H"]
    fmeans = [f0.fmean for interval in intervals_high_preceded_by_low for f0 in interval.f0_statistics if f0.fmean is not None]
    print(fmeans)
    weights = np.ones_like(fmeans)/float(len(fmeans))
    plt.hist(fmeans, bins=100, color="r", label="H / L_", alpha=0.5, weights=weights)

    intervals_high_not_preceded_by_low = [interval for previous_interval, interval in zip(intervals[:-1], intervals[1:]) if previous_interval.toneme != "L" and interval.toneme == "H"]
    fmeans = [f0.fmean for interval in intervals_high_not_preceded_by_low for f0 in interval.f0_statistics if f0.fmean is not None]
    print(fmeans)
    weights = np.ones_like(fmeans)/float(len(fmeans))
    plt.hist(fmeans, bins=100, color="b", label="H / [MH]_", alpha=0.5, weights=weights)
    plt.legend()
    plt.show()


