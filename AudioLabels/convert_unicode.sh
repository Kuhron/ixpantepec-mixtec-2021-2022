for f in *.Textgrid; do
    iconv -f utf-16 -t utf-8 $f > "utf-8/"$f
done
