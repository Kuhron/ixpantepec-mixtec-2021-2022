# FLEx merge conflicts show mojibake in the "Conflict Details" box, but correct characters in the "Conflict Report"
# this script will make it easy to convert and know what the conflict actually was in the case of diacritics / weird chars

# https://stackoverflow.com/questions/24140497/unbaking-mojibake

import chardet
import codecs


def get_str():
    print("paste string here:\n")
    lines = []
    while True:
        line = input()
        if line != "":
            lines.append(line)
        else:
            break
    falsely_decoded_str = "\n".join(lines)
    return falsely_decoded_str


def decode_unsupervised():
    # falsely_decoded_str = "Ä×èÈÄÄî¦è¤ô_üiâAâjâüâpâXüj_10òb"
    falsely_decoded_str = get_str()
    
    try:
        encoded_str = falsely_decoded_str.encode("cp850")
    except UnicodeEncodeError:
        print("could not encode falsely decoded string")
        encoded_str = None
    
    if encoded_str:
        detected_encoding = chardet.detect(encoded_str)["encoding"]
    
        try:
            correct_str = encoded_str.decode(detected_encoding)
        except UnicodeEncodeError:
            print("could not decode encoded_str as %s" % detected_encoding)
    
        # with codecs.open("output.txt", "w", "utf-8-sig") as out:
        #     out.write(correct_str)
        print(correct_str)


def decode_manual():
    s = get_str()
    TILDE = "\u0303"
    ACUTE = "\u0301"
    MACRON = "\u0304"
    GRAVE = "\u0300"
    PALATAL_NASAL = "\u0272"
    PALATALIZATION = "\u02b2"

    # would be nice if I understood the pattern behind these replacements, but I don't at the moment
    # appears to be "Wrong windows-1252 Mojibake", according to https://codepoints.net/
    # process described in detail at https://www.datafix.com.au/BASHing/2021-05-19.html

    d = {
        "Ã±": "n" + TILDE,
        "Å«": "u" + MACRON,
        "Ã¬": "i" + GRAVE,
        "É²": PALATAL_NASAL,
        "Ä": "a" + MACRON,
        "Ê²": PALATALIZATION,
    }

    for moji, rep in d.items():
        s = s.replace(moji, rep)
    print(s)


if __name__ == "__main__":
    decode_manual()
