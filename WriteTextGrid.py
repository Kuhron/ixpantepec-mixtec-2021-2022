def write_text_grid(output_fp, duration, boundaries, output_tier_name, start_end_label_by_start):
    output_lines = []

    textgrid_template_lines = [
        "File type = \"ooTextFile",
        "Object class = \"TextGrid\"",
        "",
        "xmin = 0",
        f"xmax = {duration}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
    ]
    output_lines += textgrid_template_lines

    n_intervals = len(boundaries) - 2 + 1  # fencepost, but with bookends
    tier_template_lines = [
        "\titem [1]:",  # only indented once, the others are twice
        "\t\tclass = \"IntervalTier\"",
        f"\t\tname = \"{output_tier_name}\"",
        "\t\txmin = 0",
        f"\t\txmax = {duration}",
        f"\t\tintervals: size = {n_intervals}",
    ]
    output_lines += tier_template_lines

    sorted_boundaries = sorted(boundaries, key=float)
    for i, (t0, t1) in enumerate(zip(sorted_boundaries[:-1], sorted_boundaries[1:])):
        start_end_label = start_end_label_by_start.get(t0)
        if start_end_label is None:
            # there is no label starting at this time; it's a blank space between labels
            start = t0
            end = t1
            label = ""
        else:
            start, end, label = start_end_label
            assert end == t1, f"mismatch between end of {end} and t1 of {t1}"

        interval_num = i + 1
        interval_lines = [
            f"\t\tintervals [{interval_num}]:",
            f"\t\t\txmin = {start}",
            f"\t\t\txmax = {end}",
            f"\t\t\ttext = \"{label}\"",
        ]
        output_lines += interval_lines

    with open(output_fp, "w") as f:
        for l in output_lines:
            f.write(l + "\n")
            print(f"wrote: {l}")
    print(f"\nwritten to {output_fp}")

