import re
import sys
import os

from GetWavDuration import get_duration
from WriteTextGrid import write_text_grid


if len(sys.argv) <= 1:
    print("usage: python ...py {filename}")
    print("filename should exclude AudioLabels/ and -Labels.txt. e.g. MKS-20211020-Things-M")
    sys.exit()

filename = sys.argv[1]

input_wav_fp = f"/media/wesley/easystore/FieldMethodsBackup/{filename}.wav"
input_wav_fp2 = f"/mnt/e/FieldMethodsBackup/{filename}.wav"
input_fp = f"AudioLabels/{filename}-Labels.txt"
output_fp = f"AudioLabels/{filename}.TextGrid"
print(f"converting:\n    {input_fp}\ninto:\n    {output_fp}")

if not os.path.exists(input_wav_fp):
    input_wav_fp = input_wav_fp2

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


write_text_grid(output_fp, duration, boundaries, output_tier_name, start_end_label_by_start)
