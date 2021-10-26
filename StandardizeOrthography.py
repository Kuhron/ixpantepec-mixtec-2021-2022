

class Word:
    def __init__(self, form, glosses):
        self.form = form
        self.glosses = glosses

    @staticmethod
    def from_line_in_lexicon_file(s):
        form, glosses = s.split(" = ")
        form = form.strip()
        glosses = glosses.split(";")
        glosses = [g.strip() for g in glosses]
        return Word(form, glosses)

    def tuple(self):
        return (self.form, self.glosses)


def get_wordforms_raw():
    lexicon_fp = "lexicon_IxpantepecMixtec.txt"
    with open(lexicon_fp) as f:
        lines = f.readlines()
    forms = []
    for line in lines:
        forms.append(Word.from_line_in_lexicon_file(line))
    return forms


def get_all_characters_in_wordlist(words):
    res = set()
    for w in words:
        res |= set(w.form)
    return res


def standardize_forms(words):
    # diacritics to two char
    # contour tone diacritics to two vowels with level diacritics
    # get rid of aspiration
    # standardize 5 vowels
    # esh/c ezh/z beta/v
    changes = [
        ["?", "\u0294"],  # glottal stop
        ["\xf1", "\u0272"],  # palatal nasal
        ["\xe0", "aL"], ["\xe1", "aH"], ["\xe2", "aHaL"], ["\u0101", "aM"], ["\u01ce", "aLaH"],  # tones on a
        ["\xe8", "eL"], ["\xe9", "eH"], ["\xea", "eHeL"], ["\u0113", "eM"],  # tones on e
        ["\xec", "iL"], ["\xed", "iH"], ["\xee", "iHiL"], ["\u012b", "iM"], ["\u01d0", "iLiH"],  # tones on i
        ["\xf2", "oL"], ["\xf3", "oH"], ["\xf4", "oHoL"], ["\u014d", "oM"],  # tones on o
        ["\xf9", "uL"], ["\xfa", "uH"], ["\xfb", "uHuL"], ["\u016b", "uM"], ["\u01d4", "uLuH"],  # tones on u
        ["\u1e3f", "mH"],  # tone on m
        ["\u0129", "iN"],  # nasalization
        ["\u0254", "o"], ["\u1d10", "o"], ["\u025b", "e"], ["\u026a", "i"],  # 5 vowels
        ["\u02b0", ""], ["\u035c", ""], ["\u0361", ""],  # get rid of aspiration, tie bars
        ["\u01b7", "\u0291"], ["\u0292", "\u0291"], ["\u0283", "\u0255"],  # esh and ezh
        ["\u02a4", "d\u0291"], ["\u02a7", "t\u0255"],  # coronal affricates
        ["\u027e", "r"],  # fishhook
        ["\u02b7", "w"],  # labialization diacritic > w
        ["\u03b2", "v"],  # beta
        ["\u0300", "L"], ["\u0301", "H"], ["\u0304", "M"], ["\u1dc4", "MH"], ["\u1dc5", "LM"],  # tone diacritics
    ]

    res = []
    for word in words:
        for old, new in changes:
            form = word.form.replace(old, new)
            word.form = form
        res.append(word)
    return res

"""
æ b'\\xe6'  # could be e or a
ç b'\\xe7'  # some could be h, some could be curly c
ð b'\\xf0'
ŋ b'\\u014b'
ɟ b'\\u025f'  # could be dj, dz, or gj
ɣ b'\\u0263'
ɲ b'\\u0272'
ʔ b'\\u0294'
ʲ b'\\u02b2'
"""

if __name__ == "__main__":
    words = get_wordforms_raw()
    words = standardize_forms(words)
    tups = [w.tuple() for w in words]
    print(sorted(tups))
    chars = get_all_characters_in_wordlist(words)
    print("all characters:")
    for c in sorted(chars):
        print(c, c.encode("unicode_escape"))
