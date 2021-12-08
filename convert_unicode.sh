for f in AudioLabels/*.TextGrid; do
    fname="$(basename -- $f)"
    iconv -f utf-16 -t utf-8 $f > "AudioLabels/utf-8/"$fname;
done

for f in EandWProjectFall2021/PitchStats/*.txt; do
    fname="$(basename -- $f)"
    iconv -f utf-16 -t utf-8 $f > "EandWProjectFall2021/PitchStats/utf-8/"$fname;
done

