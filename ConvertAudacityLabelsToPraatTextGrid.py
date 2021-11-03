import re

from GetWavDuration import get_duration


filename = "MKS-20211004-Space_Time_Pronouns_1-M"

input_wav_fp = f"/media/wesley/easystore/FieldMethodsBackup/{filename}.wav"
input_fp = f"AudacityLabels/{filename}-Labels.txt"
output_fp = f"AudacityLabels/{filename}.TextGrid"
print(f"converting:\n    {input_fp}\ninto:\n    {output_fp}")

duration = get_duration(input_wav_fp)
print(f"input wav:\n    {input_wav_fp}\nof duration:\n    {duration} seconds")

with open(input_fp) as f:
    lines = f.readlines()
lines = [l.strip() for l in lines]  # get rid of trailing newlines

output_tier_name = "word"
regex = r"^(?P<start>[.\d]+)\t(?P<end>[.\d]+)\t(?P<label>[^\t]*)$"

boundaries = {0, duration}
start_end_label_by_start = {}
for line in lines:
    match = re.match(regex, line)
    if match:
        start = match["start"]
        end = match["end"]
        label = match["label"]
        # print(f"{label} starts at {start} and ends at {end}")
        # don't think I need to cast the times to float since I'm just gonna write them as str again, unless Praat wants a certain precision
        boundaries |= {start, end}
        assert start not in start_end_label_by_start, f"duplicate start time {start}"
        start_end_label_by_start[start] = [start, end, label]
    else:
        raise Exception(f"invalid line: {line}")


output_lines = []

textgrid_template_lines = [
    "File type = \"ooTextFile",
    "Object class = \"TextGrid\"",
    "",
    "xmin = 0",
    f"xmax = {duration}",
    "tiers? <exists>",
    "size = 1",
    "item []:",
]
output_lines += textgrid_template_lines

n_intervals = len(boundaries) - 2 + 1  # fencepost, but with bookends
tier_template_lines = [
    "\titem [1]:",  # only indented once, the others are twice
    "\t\tclass = \"IntervalTier\"",
    f"\t\tname = \"{output_tier_name}\"",
    "\t\txmin = 0",
    f"\t\txmax = {duration}",
    f"\t\tintervals: size = {n_intervals}",
]
output_lines += tier_template_lines

sorted_boundaries = sorted(boundaries, key=float)
for i, (t0, t1) in enumerate(zip(sorted_boundaries[:-1], sorted_boundaries[1:])):
    start_end_label = start_end_label_by_start.get(t0)
    if start_end_label is None:
        # there is no label starting at this time; it's a blank space between labels
        start = t0
        end = t1
        label = ""
    else:
        start, end, label = start_end_label
        assert end == t1, f"mismatch between end of {end} and t1 of {t1}"

    interval_num = i + 1
    interval_lines = [
        f"\t\tintervals [{interval_num}]:",
        f"\t\t\txmin = {start}",
        f"\t\t\txmax = {end}",
        f"\t\t\ttext = \"{label}\"",
    ]
    output_lines += interval_lines

with open(output_fp, "w") as f:
    for l in output_lines:
        f.write(l + "\n")
print(f"written to {output_fp}")
