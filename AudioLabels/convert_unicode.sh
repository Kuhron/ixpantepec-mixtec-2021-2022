for f in *.TextGrid; do
    iconv -f utf-16 -t utf-8 $f > "utf-8/"$f;
done
