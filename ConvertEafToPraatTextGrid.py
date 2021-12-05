import re
import xml.etree.ElementTree as ET


filename = "MKS-20211006-Memories_of_Tortilla_Making-M"
input_fp = "/home/kuhron/ixpantepec-mixtec-2021-2022/SayMore/Sessions/Memories of Tortilla Making/MKS-20211006-Memories_of_Tortilla_Making-M_Amplified.wav.annotations.eaf"
output_fp = f"AudioLabels/{filename}.TextGrid"

tier_name = "sentence"

tree = ET.parse(input_fp)
root = tree.getroot()
assert root.tag == "ANNOTATION_DOCUMENT", root.tag

time_order_el, = root.findall("TIME_ORDER")
time_slot_els = time_order_el.findall("TIME_SLOT")
time_slot_dict = {}
for el in time_slot_els:
    ts_id = el.attrib["TIME_SLOT_ID"]
    t_ms = int(el.attrib["TIME_VALUE"])
    t_s = t_ms / 1000
    assert ts_id not in time_slot_dict
    time_slot_dict[ts_id] = t_s
    last_time_s = t_s

duration = last_time_s  # SayMore puts a time value at the very end of the sound file

tier_els = root.findall("TIER")
typerefs = [el.attrib["LINGUISTIC_TYPE_REF"] for el in tier_els]
assert len(tier_els) == 2, f"need 2 tiers, got {typerefs}"
transcription_tier_el, = [el for el in tier_els if el.attrib["LINGUISTIC_TYPE_REF"] == "Transcription"]
translation_tier_el, = [el for el in tier_els if el.attrib["LINGUISTIC_TYPE_REF"] == "Translation"]

transcription_els = transcription_tier_el.findall("ANNOTATION")
translation_els = translation_tier_el.findall("ANNOTATION")
annotation_dict = {}
assert len(transcription_els) == len(translation_els)
for tsc_a_el, tsl_a_el in zip(transcription_els, translation_els):
    aa_el, = tsc_a_el.findall("ALIGNABLE_ANNOTATION")
    tsc_a_id = aa_el.attrib["ANNOTATION_ID"]
    ts1 = aa_el.attrib["TIME_SLOT_REF1"]
    ts2 = aa_el.attrib["TIME_SLOT_REF2"]
    av_el, = aa_el.findall("ANNOTATION_VALUE")
    tsc_text = av_el.text
    if tsc_text == "%ignore%":
        tsc_text = ""

    ra_el, = tsl_a_el.findall("REF_ANNOTATION")
    tsl_a_id = ra_el.attrib["ANNOTATION_ID"]
    tsl_ref_id = ra_el.attrib["ANNOTATION_REF"]
    assert tsc_a_id == tsl_ref_id, (tsc_a_id, tsl_ref_id)
    av_el, = ra_el.findall("ANNOTATION_VALUE")
    tsl_text = av_el.text

    key = (ts1, ts2)
    text = f"{tsc_text} = {tsl_text}"
    assert key not in annotation_dict
    annotation_dict[key] = text

print(annotation_dict)
