import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Vowel_length.csv")  # since I can't be bothered to parse it myself lol
durations = df["duration"]

# throw out weird outliers like 800 seconds
durations = durations[durations < 1]

plt.hist(durations, bins=100)
print(sorted(durations))
plt.savefig("fig.png")
