import random
import math


NOTES = "CKDHEFXGJARB"
assert len(NOTES) == 12
# A4 is 440 Hz

def get_hz_from_note(s):
    s = s.upper()
    note, octave = s
    # substitutions from older system
    if note == "L":
        note = "J"
    elif note == "M":
        note = "R"
    note_index = NOTES.index(note)
    octave = int(octave)
    octave_diff = octave - 4
    index_diff = note_index - NOTES.index("A")
    return 440 * 2 ** (octave_diff + index_diff/12)


def get_random_note():
    note = random.choice(NOTES)
    octave = random.randint(2, 6)  # inclusive
    return f"{note}{octave}"


def quiz_random_note():
    note = get_random_note()
    hz = get_hz_from_note(note)
    if random.random() < 0.5:
        # quiz by note name
        input(f"The note is {note}. What is the pitch in Hz?")
    else:
        # quiz by pitch
        input(f"The pitch is {round(hz)} Hz. What is the note?")
    print(f"The note {note} has a pitch of {round(hz)}.")
    print()


def show_random_interval_ratio():
    semitones = random.randint(1, 11)
    factor_down = 2 ** (-semitones/12)
    factor_up = 2 ** (semitones/12)
    print(f"{semitones} semitones down is a factor of {factor_down:.2f}, up is {factor_up:.2f}")


def show_all_interval_factors():
    for semitones in range(1, 12):
        factor_down = 2 ** (-semitones/12)
        factor_up = 2 ** (semitones/12)
        nd = round(100 * factor_down)
        nu = round(100 * factor_up)
        print(f"{semitones} semitones\t{nd}\t{nu}")
    print()


def show_all_notes_hz():
    for octave in range(2, 7):
        for note in NOTES:
            note_str = f"{note}{octave}"
            hz = get_hz_from_note(note_str)
            hz = round(hz)
            print(f"{note_str}\t{hz}")
    print()


while True:
    show_all_notes_hz()
    show_all_interval_factors()
    print("----")
    show_random_interval_ratio()
    quiz_random_note()
    input("press enter to continue")
