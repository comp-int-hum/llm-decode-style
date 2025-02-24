import argparse
import re
import json
import csv

#--text_input ${SOURCES[0]} --out ${TARGETS[0]}
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--text_input", help="Plaintext input")
    parser.add_argument("--min_chars", type=int, default=10)
    parser.add_argument("--char_name", default="")
    parser.add_argument("--out", help="jsonl out")
    args, rest = parser.parse_known_args()

    print(args.text_input)
    print(args.char_name)

    if args.text_input.endswith(".txt"):
        with open(args.text_input, "rt") as s_in:
            text = s_in.read()
            text_split = [s.replace("\n"," ") for s in re.split("\n\n",text) if not any([c.isdigit() for c in s]) and len(s)>=args.min_chars and (s[0]!="[" and s[-1]!="]") and "CHAPTER" not in s]
    elif args.text_input.endswith(".csv"):
        text_split = []
        text_buffer = []
        with open(args.text_input, "rt", newline="") as s_in:
            s_reader = csv.reader(s_in, delimiter="\t")
            for row in s_reader:
                if args.char_name in row[1] and len(row[0]) >= args.min_chars and "[" not in row[0]:
                    text_buffer.append(row[0].replace("\t", " ").strip())
                    if text_buffer[-1][-1] in [".","!","?"] or (text_buffer[-1][-1] in ["'",'"'] and text_buffer[-1][-2] in [".","!","?"]):
                        text_split.append(" ".join(text_buffer))
                        text_buffer = []

    with open(args.out, "wt") as t_out:
        for c in text_split:
            t_out.write(json.dumps({"text":c})+"\n") 
