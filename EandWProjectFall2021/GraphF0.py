import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
import statistics
import string
import sys



class F0Statistic:
    def __init__(self, raw_line, source_fp, fmin_hz, fmax_hz, fmean_hz, fstd_hz, start, end, parent_interval):
        self.raw_line = raw_line
        self.source_fp = source_fp
        assert np.isnan(fmin_hz) or fmin_hz > 0, fmin_hz
        assert np.isnan(fmax_hz) or fmax_hz > 0, fmax_hz
        assert np.isnan(fmean_hz) or fmean_hz > 0, fmean_hz
        assert np.isnan(fstd_hz) or fstd_hz > 0, fstd_hz
        self.fmin_hz = fmin_hz
        self.fmax_hz = fmax_hz
        self.fmean_hz = fmean_hz
        self.fstd_hz = fstd_hz

        # st means semitones
        self.fmin_st = get_midi_note_from_hz(fmin_hz)
        self.fmax_st = get_midi_note_from_hz(fmax_hz)
        self.fmean_st = get_midi_note_from_hz(fmean_hz)
        # can't just log-transform the std because it's not a linear transformation (could be an okay approximation though, for cases where the frequency doesn't move very much)

        self.start = start
        self.midpoint = (start + end) / 2
        self.end = end
        assert type(parent_interval) is Interval
        self.parent_interval = parent_interval


class Interval:
    def __init__(self, raw_line, source_fp, label, start, end):
        self.raw_line = raw_line
        self.source_fp = source_fp
        self.label = label
        self.start = start
        self.midpoint = (start + end) / 2
        self.end = end
        self.f0_statistics = []
        self.whistled = "w" in self.label
        self.nasalized = "n" in self.label
        self.glottalized = "'" in self.label
        self.toneme_level = self.get_toneme_level()
        self.toneme = ("w" if self.whistled else "") + self.toneme_level

    def get_toneme_level(self):
        s = "".join(get_tonemes_from_str(self.label))
        no_nos = {"LL": "L", "MM": "M", "HH": "H"}
        for no_no, rep in no_nos.items():
            while no_no in s:
                s = s.replace(no_no, rep)
        return s

    def add_f0_statistic(self, f0_statistic):
        assert type(f0_statistic) is F0Statistic, type(f0_statistic)
        self.f0_statistics.append(f0_statistic)
        self.sanity_check()

    def sanity_check(self):
        # make sure we don't have whistled labeled as spoken or vice versa
        error_strs = []
        assert type(self.whistled) is bool  # in case the `if self.x` is False not because the attr is false but because it doesn't exist
        f0s = [f0.fmean_hz for f0 in self.f0_statistics]
        f0s = [hz for hz in f0s if np.isfinite(hz)]

        if self.whistled:
            matches = all(hz > 500 for hz in f0s)
            if not matches:
                error_strs.append(f"spoken tone labeled as whistled: F0s in Hz are {f0s}")
        else:
            matches = all(hz < 700 for hz in f0s)
            if not matches:
                error_strs.append(f"whistled tone labeled as spoken: F0s in Hz are {f0s}")

        if len(error_strs) > 0:
            error_str = f"\n\nWarning: sanity check failed in interval from line:\n{self.raw_line}\nin file {self.source_fp}\nErrors:"
            for s in error_strs:
                error_str += "\n" + s
            print(error_str + "\n")


def get_tonemes_from_str(s):
    # return a list of toneme labels (L M or H)
    tone_diacritics = {
        "\u0301": "H",
        "\u0304": "M",
        "\u0300": "L",
        "\u0302": "HL",
        "\u1dc4": "MH",
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
        print(f"Warning: no tone symbols found in string: {s.encode('unicode_escape')}")
        return []
    return tonemes


def get_interval_from_line(s, source_fp):
    INTERVAL_LINE_RX = "^interval #(?P<number>\d+?)\tlabel: (?P<label>.*?)\tstart: (?P<start>[\d.]+?)\tend: (?P<end>[\d.]+?)$"

    match = re.match(INTERVAL_LINE_RX, s)
    if match is None:
        raise Exception(f"match is None for line: {repr(s)}")
    label = match["label"]
    # strip off the tabs from when I accidentally labeled an interval with tab while meaning to play it in Praat
    if label.strip() == "":
        print("ignoring interval which is labeled only with whitespace")
        return None

    start = float(match["start"])
    end = float(match["end"])
    interval = Interval(s, source_fp, label, start, end)
    return interval


def get_f0_statistic_from_line(s, source_fp, timestep_ms, parent_interval):
    F0_LINE_RX = "^step #(?P<step_number>\d+?)\/(?P<n_steps>\d+?)\tfMin: (?P<fmin>[\d.]+?|--undefined--)\tfMax: (?P<fmax>[\d.]+?|--undefined--)\tfMean: (?P<fmean>[\d.]+?|--undefined--)\tstdev: (?P<fstd>[\d.]+?|--undefined--)$"

    match = re.match(F0_LINE_RX, s)
    if match is None:
        raise Exception(f"match is None for line: {repr(s)}")
    fmin = float(match["fmin"]) if match["fmin"] != "--undefined--" else np.nan
    fmax = float(match["fmax"]) if match["fmax"] != "--undefined--" else np.nan
    fmean = float(match["fmean"]) if match["fmean"] != "--undefined--" else np.nan
    fstd = float(match["fstd"]) if match["fstd"] != "--undefined--" else np.nan
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

    f0_statistic = F0Statistic(s, source_fp, fmin, fmax, fmean, fstd, start, end, parent_interval)
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
                new_interval = get_interval_from_line(l, fp)
                if new_interval is None:
                    # it was an erroneous label where I typed tab to play but Praat put that in the label instead
                    continue
            except:
                print(f"\nError encountered in file\n\t{fp}\nin line: {l}\n")
                raise
            # bank this interval
            if current_interval is not None:
                intervals.append(current_interval)
            current_interval = new_interval
        elif l.startswith("step"):
            try:
                f0_statistic = get_f0_statistic_from_line(l, fp, f0_timestep_ms, current_interval)
            except:
                print(f"\nError encountered in file\n\t{fp}\nin line: {l}\n")
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


def plot_hist_fmeans(intervals_by_toneme, semitones=True):
    for toneme, intervals in intervals_by_toneme.items():
        fmeans = []
        for interval in intervals:
            for f0 in interval.f0_statistics:
                fmean = f0.fmean_st if semitones else f0.fmean_hz
                if fmean is not None:
                    fmeans.append(fmean)
        plt.gcf().clear()
        plt.hist(fmeans, bins=100)
        plt.title(toneme)
        plt.show()


def plot_fmeans_color_coded(intervals):
    fmeans_st = [f0.fmean_st for interval in intervals for f0 in interval.f0_statistics]
    fmins_st = [f0.fmin_st for interval in intervals for f0 in interval.f0_statistics]
    fmaxs_st = [f0.fmax_st for interval in intervals for f0 in interval.f0_statistics]
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
    for series in [fmeans_st, fmins_st, fmaxs_st]:
        plt.scatter(times, series, c=color_series)
    transform_axis_to_logarithmic_with_notes(plt.gca())
    # plt.legend(color_dict)
    plt.show()


def get_midi_note_from_hz(hz):
    # A4 = 440 Hz, C4 = MIDI 60, so A4 = 440 Hz = MIDI 69
    if np.isnan(hz): return np.nan
    assert hz > 0, hz
    ratio = hz / 440
    log2_ratio = math.log2(ratio)
    semitones_diff = 12 * log2_ratio
    return 69 + semitones_diff


def get_hz_from_midi_note(n):
    # A4 = 440 Hz, C4 = MIDI 60, so A4 = 440 Hz = MIDI 69
    if np.isnan(n): return np.nan
    semitones_diff = n - 69
    log2_ratio = semitones_diff / 12
    ratio = 2 ** log2_ratio
    return 440 * ratio


def get_note_symbol_from_midi_note(mn):
    # A4 = 440 Hz, C4 = MIDI 60, so A4 = 440 Hz = MIDI 69
    if np.isnan(mn): return np.nan
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


def get_intervals_by_condition(intervals, 
        tonemes=None, preceding_tonemes=None, following_tonemes=None,
        nasalized=None, whistled=None, glottalized=None,
    ):
    # for each condition, filter what remains of the intervals
    # keep a bool array
    mask = [True for x in intervals]
    n = len(intervals)

    if tonemes is not None:
        for i in range(n):
            if not mask[i]:
                # if it's already being excluded due to another condition, move on
                continue
            matches = intervals[i].toneme in tonemes
            mask[i] = mask[i] and matches

    if preceding_tonemes is not None:
        for i in range(n):
            if not mask[i]: continue
            matches = (i > 0) and intervals[i-1].toneme in preceding_tonemes
            mask[i] = mask[i] and matches

    if following_tonemes is not None:
        for i in range(n):
            if not mask[i]: continue
            matches = (i < n-1) and intervals[i+1].toneme in following_tonemes
            mask[i] = mask[i] and matches

    if nasalized is not None:
        for i in range(n):
            if not mask[i]: continue
            matches = intervals[i].nasalized == nasalized
            mask[i] = mask[i] and matches

    if whistled is not None:
        for i in range(n):
            if not mask[i]: continue
            matches = intervals[i].whistled == whistled
            mask[i] = mask[i] and matches

    if glottalized is not None:
        for i in range(n):
            if not mask[i]: continue
            matches = intervals[i].glottalized == glottalized
            mask[i] = mask[i] and matches

    return [x for i, x in enumerate(intervals) if mask[i]]


def get_fmeans_from_intervals(intervals, semitones=True):
    return np.array([(f0.fmean_st if semitones else f0.fmean_hz) for interval in intervals for f0 in interval.f0_statistics])


def get_fstds_from_intervals(intervals, semitones=True):
    if semitones:
        raise ValueError("can't get std in semitones because it's a nonlinear transformation")
    return np.array([f0.fstd_hz for interval in intervals for f0 in interval.f0_statistics])


def hist_multiple_fmean_sets(intervals_list, labels, colors, bins=100, alpha=0.5, semitones=True, mean_and_std_lines=False, separate_axes=False):
    fmeans_lists = [get_fmeans_from_intervals(intervals, semitones=semitones) for intervals in intervals_list]
    hist_multiple_sets(fmeans_lists, labels, colors, bins, alpha, semitones, mean_and_std_lines)


def hist_multiple_sets(values_lists, labels, colors=None, bins=100, alpha=0.5, semitones=True, mean_and_std_lines=False, separate_axes=False):
    if separate_axes:
        n_plots = len(labels)
        fig = plt.figure()
        ax1 = plt.subplot(n_plots, 1, 1)
        axes = [ax1] + [plt.subplot(n_plots, 1, i+1, sharex=ax1) for i in range(1, n_plots)]
    if colors is None:
        colors = [None for x in labels]
    for i, (values, label, color) in enumerate(zip(values_lists, labels, colors)):
        assert type(values) is list, type(values)
        values = np.array(values)
        defined_values = values[np.isfinite(values)]
        mean_overall = np.mean(defined_values)
        std_overall = np.std(defined_values)
        n_samples = len(defined_values)
        print(f"set of label {label} has mean {mean_overall} and std {std_overall} (n={n_samples})")
        weights = np.ones_like(values)/float(len(values))

        if separate_axes:
            ax = axes[i]
        else:
            ax = plt.gca()
        hist = ax.hist(values, bins=bins, color=color, label=label, alpha=alpha, weights=weights)
        if mean_and_std_lines:
            add_mean_and_std_lines(ax, values, hist)
        if separate_axes:
            ax.legend()  # one for each subplot
    plt.legend()
    plt.show()


def add_mean_and_std_lines(ax, xs, hist):
    xs = [x for x in xs if np.isfinite(x)]
    mean = np.mean(xs)
    std = np.std(xs)
    ymin = 0
    heights, bins, patches = hist
    ymax = max(heights)
    alpha_bottom = 0.2
    alpha_top = 0.8
    bottom = (1 - alpha_bottom) * ymin + (alpha_bottom) * ymax
    top = (1 - alpha_top) * ymin + (alpha_top) * ymax
    ax.plot([mean, mean], [bottom, top], c="k")
    ymid = (ymin + ymax)/2
    xleft = mean - std
    xright = mean + std
    ax.plot([xleft, xright], [ymid, ymid], c="k")


def plot_f0_trajectory(ts, fmeans_st, fmins_st, fmaxs_st, ranges_st, midpoints_between_ts, df_dts, second_order_midpoints, d2f_dt2s):
    plt.subplot(3,1,1)
    plt.plot(ts, fmeans_st, label="mean")
    plt.plot(ts, fmins_st, label="min")
    plt.plot(ts, fmaxs_st, label="max")
    xlim = plt.gca().get_xlim()
    plt.ylabel("F0 (semitones)")
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(ts, ranges_st, label="range")
    plt.gca().set_xlim(xlim)
    plt.ylabel("semitones")
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(midpoints_between_ts, df_dts, label="df/dt")
    plt.plot(second_order_midpoints, d2f_dt2s, label="d2f/dt2")
    plt.axhline(0, c="k")
    plt.gca().set_xlim(xlim)
    plt.legend()
    plt.ylabel("semitones / s^n")
    plt.xlabel("time (s)")

    plt.show()


def interpolate(ts, ys, interpolation_ts):
    # assert len(set(ts)) == len(ts), f"duplicate t values: {sorted(t for t in set(ts) if ts.count(t) > 1)}"
    tups = sorted(zip(ts, ys))
    ts = [tup[0] for tup in tups]
    ys = [tup[1] for tup in tups]
    interpolated_ys = []
    for interp_t in interpolation_ts:
        if interp_t in ts:
            # we already have a value here, just get it
            interp_y = ys[ts.index(interp_t)]  # inefficient if need to optimize something
        else:
            below_t = None
            above_t = None
            below_y = None
            above_y = None
            for t, y in tups:
                if t < interp_t:
                    if below_t is None or t > below_t:
                        below_t = t
                        below_y = y
                elif t > interp_t:
                    if above_t is None or t < above_t:
                        above_t = t
                        above_y = y
                else:
                    raise RuntimeError("shouldn't happen")
            alpha = (interp_t - below_t) / (above_t - below_t)
            dy = above_y - below_y
            interp_y = below_y + alpha * dy
            # print(f"t {below_t:.4f} -> {interp_t:.4f} -> {above_t:.4f} => y {below_y:.4f} -> {interp_y:.4f} -> {above_y:.4f}")
        interpolated_ys.append(interp_y)
    return interpolated_ys


def get_stationary_tone_targets_from_intervals(intervals, slope_tolerance, outlier_bounds):
    # this one should NOT return times because that would be treating times in different intervals as though they are the same
    targets = []
    for interval in intervals:
        times_i, targets_i = get_stationary_tone_targets(interval, slope_tolerance, outlier_bounds, plot=False)
        targets += targets_i
    return targets


def get_stationary_tone_targets(interval, slope_tolerance, outlier_bounds, plot=False):
    # get the min, mean, and max F0 (could also have a condition on the std? but min-max range should be fine as first pass)
    # keep the Nones so the arrays align if somehow one of them has some of these frequencies defined but others undefined (shouldn't be possible I hope, but who knows what Praat will output in weird cases)
    # slope tolerance is max absolute value of first derivative of F0 wrt time (in semitones/second)

    f0s = [f0 for f0 in interval.f0_statistics if not is_outlier(f0.fmean_st, outlier_bounds)]
    fmeans_st = [f0.fmean_st for f0 in f0s]
    fmins_st = [f0.fmin_st for f0 in f0s]
    fmaxs_st = [f0.fmax_st for f0 in f0s]
    ranges_st = [fmax - fmin for fmax, fmin in zip(fmaxs_st, fmins_st)]
    ts = np.array([f0.midpoint for f0 in f0s])  # return times in absolute position in the recording

    dts = np.diff(ts)
    dfmeans = np.diff(fmeans_st)
    midpoints_between_ts = [(t0 + t1)/2 for t0, t1 in zip(ts[:-1], ts[1:])]  # where the df/dt point will be plotted
    df_dts = [df/dt for df, dt in zip(dfmeans, dts)]

    # second derivative, want critical points where the second derivative has a large magnitude (tone comes to this target and then switches direction, good sign that it's a target (or maybe a little bit of an overshoot but don't worry about that right now))
    # but also want level tones where neither derivative has a large value, how to tell difference between this and a slowly but surely changing tone?

    dt2s = ts[2:] - ts[:-2]  # don't do np.diff(dts) or np.diff(ts, 2) because that's the difference between the differences which could be like [0.1, 0.1, 0.1] -> [0, 0]
    d2fmeans = np.diff(dfmeans)
    second_order_midpoints = [(t0 + t1)/2 for t0, t1 in zip(midpoints_between_ts[:-1], midpoints_between_ts[1:])]
    d2f_dt2s = [d2f/dt2 for d2f, dt2 in zip(d2fmeans, dt2s)]

    range_tolerance = 0.1
    # second_derivative_tolerance = 5  # only use this if you want to restrict it to level tones; if want any critical points then don't put conditions on the second derivative

    # times_satisfying_range = [t for t, rng in zip(ts, ranges_st) if abs(rng) <= range_tolerance]
    times_satisfying_slope = [t for t, df_dt in zip(midpoints_between_ts, df_dts) if abs(df_dt) <= slope_tolerance]
    # times_satisfying_second_derivative = [t for t, d2f_dt2 in zip(second_order_midpoints, d2f_dt2s) if abs(d2f_dt2) <= second_derivative_tolerance]

    # linearly interpolate the semitones at the desired times
    # fmeans_at_small_range = interpolate(ts, fmeans_st, times_satisfying_range)
    fmeans_at_small_slope = interpolate(ts, fmeans_st, times_satisfying_slope)
    # print("times with small range:", times_satisfying_range)
    # print("fmeans at these times:", fmeans_at_small_range)
    # print("times with small slope:", times_satisfying_first_derivative)
    # print("fmeans at these times:", fmeans_at_small_slope)

    if plot:
        plot_f0_trajectory(ts, fmeans_st, fmins_st, fmaxs_st, ranges_st, midpoints_between_ts, df_dts, second_order_midpoints, d2f_dt2s)

    return times_satisfying_slope, fmeans_at_small_slope


def is_outlier(value, outlier_bounds):
    return value < outlier_bounds[0] or value > outlier_bounds[1]


def plot_stationary_targets_over_time(intervals, slope_tolerance, outlier_bounds):
    # sort them into files and place those along a timeline with vertical bars showing the file boundaries
    intervals_by_fp = {}
    for x in intervals:
        assert type(x) is Interval
        fp = x.source_fp
        if fp not in intervals_by_fp:
            intervals_by_fp[fp] = []
        intervals_by_fp[fp].append(x)
    durations_by_fp = {}
    for fp in intervals_by_fp:
        end = max(x.end for x in intervals_by_fp[fp])
        assert type(end) is float
        durations_by_fp[fp] = end
    fps = sorted(intervals_by_fp.keys())  # just make some well-defined order so we can stack the times into one big timeline

    cumulative_start_times_by_fp = {fps[0]: 0}
    cumulative_end_times_by_fp = {fps[0]: durations_by_fp[fps[0]]}
    artificial_gap_between_files = 10  # make sure it's long enough to definitely count as a "pause"
    for i in range(1, len(fps)):
        previous_end = cumulative_end_times_by_fp[fps[i-1]]
        current_length = durations_by_fp[fps[i]]
        current_start = previous_end + artificial_gap_between_files
        current_end = current_start + current_length
        cumulative_start_times_by_fp[fps[i]] = current_start
        cumulative_end_times_by_fp[fps[i]] = current_end

    times_to_plot = []
    targets_to_plot = []
    times_seen = set()
    for fp in fps:
        t0 = cumulative_start_times_by_fp[fp]
        intervals_in_fp = intervals_by_fp[fp]
        for interval in intervals_in_fp:
            times_i, targets_i = get_stationary_tone_targets(interval, slope_tolerance, outlier_bounds)
            times_i = [t0 + t for t in times_i]  # align it to where this file starts
            set_times_i = set(times_i)
            assert set_times_i & times_seen == set(), "duplicate times"
            times_seen |= set_times_i
            times_to_plot += times_i
            targets_to_plot += targets_i
    file_time_boundaries = set(cumulative_start_times_by_fp.values()) | set(cumulative_end_times_by_fp.values())
    for t in file_time_boundaries:
        plt.axvline(t, c="k")
    plt.scatter(times_to_plot, targets_to_plot)
    plt.show()



if __name__ == "__main__":
    emma = False
    if emma:
        print("running as Emma")
        os.chdir('/Users/emmacmazzuchi/Desktop/MKS_tone_stats')
        f0_fps = [
            "MKS-20211020-Plants_and_Animals_4-M-F0-25ms.txt",
            "MKS-20211004-Space_Time_Pronouns_1-M-F0-25ms.txt",
            "MKS-20211018-music_and_verbs-M_mono-F0-50ms.txt",
        ]
    else:
        print("running as Wesley")
        # wesley
        timesteps_to_use = [10]  # ran Praat script with 10ms, 25ms, and 50ms
        fnames_to_use = [
            "MKS-20211004-Space_Time_Pronouns_1-M",
            "MKS-20211018-music_and_verbs-M",
            "MKS-20211020-Plants_and_Animals_4-M",
            "MKS-20211006-Memories_of_Tortilla_Making-M",
            "MKS-20211011-Space_Time_Pronouns_2-M",
        ]
        f0_fps = [f"PitchStats/{fname}-F0-{ms}ms.txt" for fname in fnames_to_use for ms in timesteps_to_use]

    intervals = get_intervals_from_files(f0_fps)

    # intervals_by_toneme = get_intervals_by_toneme_dict(intervals)
    # plot_hist_fmeans(intervals_by_toneme)
    # plot_fmeans_color_coded(intervals)

    # --- mean and std by slope tolerance and label --- #
    # for slope_tolerance in [0.1, 0.316, 1, 3.16, 10]:
    #     print(f"slope tolerance is {slope_tolerance}")
    #     stationary_targets_by_toneme = {}
    #     for interval in intervals:
    #         if interval.toneme not in stationary_targets_by_toneme:
    #             stationary_targets_by_toneme[interval.toneme] = []
    #         times, targets = get_stationary_tone_targets(interval, slope_tolerance)
    #         stationary_targets_by_toneme[interval.toneme] += targets
    #     tonemes_to_plot = ["wL", "wM", "wH"]
    #     targets_lists = [stationary_targets_by_toneme[toneme] for toneme in tonemes_to_plot]
    #     colors = None
    #     hist_multiple_sets(targets_lists, tonemes_to_plot, colors, mean_and_std_lines=True, semitones=True, separate_axes=True)
    # /// mean and std by slope tolerance and label /// #

    # --- distribution of stationary targets overall (regardless of label) ---
    intervals_spoken = [x for x in intervals if not x.whistled]
    intervals_whistled = [x for x in intervals if x.whistled]
    assert intervals_spoken != intervals_whistled, "probably mistake with hasattr returning False when it doesn't exist"
    # for slope_tolerance in [0.1, 0.316, 1, 3.16, 10]:
    #     print(f"slope tolerance is {slope_tolerance}")
    #     targets_spoken = get_stationary_tone_targets_from_intervals(intervals_spoken, slope_tolerance)
    #     hist_multiple_sets([targets_spoken], ["all spoken"], None)
    #     targets_whistled = get_stationary_tone_targets_from_intervals(intervals_whistled, slope_tolerance)
    #     hist_multiple_sets([targets_whistled], ["all whistled"])
    # /// distribution of stationary targets overall (regardless of label) ///

    # --- interval start and end times (to find gaps that can be proxy for intonation resets)
    gaps = []
    for fp in f0_fps:
        intervals_this_file = get_intervals_from_file(fp)
        gaps_this_file = []
        for i in range(len(intervals_this_file)-1):
            previous_end = intervals_this_file[i].end
            next_start = intervals_this_file[i + 1].start
            gap = next_start - previous_end
            assert gap >= 0
            gaps_this_file.append(gap)
        gaps += gaps_this_file
    print(sorted(gaps))
    # hist_multiple_sets([gaps], ["gaps"])
    # /// interval start and end times (to find gaps that can be proxy for intonation resets)

    outlier_bounds_spoken = (44.082, 65.244)
    outlier_bounds_whistled = (82.742, 93.562)

    slope_tolerance = 5
    plot_stationary_targets_over_time(intervals_spoken, slope_tolerance, outlier_bounds_spoken)
    plot_stationary_targets_over_time(intervals_whistled, slope_tolerance, outlier_bounds_whistled)



    # ----

    if emma:
        # more succinctly writing the plotting code for histograms of tones with certain conditions

        all_tonemes = set(x.toneme for x in intervals)
        non_low_tonemes = [t for t in all_tonemes if t != "L"]

        intervals_H = get_intervals_by_condition(intervals, tonemes=["H"], whistled=False)
        intervals_HL = get_intervals_by_condition(intervals, tonemes=["HL"], whistled=False)
        intervals_HM = get_intervals_by_condition(intervals, tonemes=["HM"], whistled=False)
        intervals_H_nasal = get_intervals_by_condition(intervals, nasalized=True, tonemes=["H"], whistled=False)
        intervals_H_not_preceded_by_L = get_intervals_by_condition(intervals, tonemes=["H"], preceding_tonemes=non_low_tonemes, whistled=False)
        intervals_H_preceded_by_L = get_intervals_by_condition(intervals, tonemes=["H"], preceding_tonemes=["L"], whistled=False)
        intervals_L = get_intervals_by_condition(intervals, tonemes=["L"], whistled=False)
        intervals_LH = get_intervals_by_condition(intervals, tonemes=["LH"], whistled=False)
        intervals_LM = get_intervals_by_condition(intervals, tonemes=["LM"], whistled=False)
        intervals_M = get_intervals_by_condition(intervals, tonemes=["M"], whistled=False)
        intervals_MH = get_intervals_by_condition(intervals, tonemes=["MH"], whistled=False)
        intervals_ML = get_intervals_by_condition(intervals, tonemes=["ML"], whistled=False)
        intervals_glottalized = get_intervals_by_condition(intervals, glottalized=True, whistled=False)
        intervals_non_whistled = get_intervals_by_condition(intervals, whistled=False)
        # caveat: the "not preceded by low" should include the first tone (which isn't preceded by anything)
        # - but it won't because the condition function assumes any condition on preceding tones disallows the first interval

        # plot the histograms of each of these interval sets
        intervals_list = [
            # intervals, 
            # intervals_non_whistled, 
            intervals_H_nasal, 
            intervals_H_preceded_by_L, 
            intervals_H_not_preceded_by_L, 
        ]
        labels = [
            # "all", 
            # "non-whistled", 
            "Hn", 
            "H / L_", 
            "H / !L_", 
        ]
        colors = [
            # "blue", 
            # "blue", 
            "orange", 
            "red", 
            "green", 
        ]
        # hist_multiple_fmean_sets(intervals_list, labels, colors, mean_and_std_lines=True, semitones=False)


