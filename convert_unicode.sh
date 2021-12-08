for f in AudioLabels/*.TextGrid; do
    iconv -f utf-16 -t utf-8 $f > "AudioLabels/utf-8/"$f;
done

for f in EandWProjectFall2021; do
    iconv -f utf-16 -t utf-8 $f > "EandWProjectFall2021/utf-8/"$f;
done

