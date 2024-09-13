import argparse
import re
import json

#--text_input ${SOURCES[0]} --out ${TARGETS[0]}
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--text_input", help="Plaintext input")
    parser.add_argument("--min_chars", type=int, default=20)
    parser.add_argument("--out", help="jsonl out")
    args, rest = parser.parse_known_args()

    with open(args.text_input, "rt") as s_in:
        text = s_in.read()
        text_split = [s.replace("\n"," ") for s in re.split("\n\n",text) if not any([c.isdigit() for c in s]) and len(s)>=args.min_chars and (s[0]!="[" and s[-1]!="]") and "CHAPTER" not in s]

    with open(args.out, "wt") as t_out:
        for c in text_split:
            t_out.write(json.dumps({"text":c})+"\n") 
