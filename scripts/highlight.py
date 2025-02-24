
import json
import math
import argparse

import string


def wrap_latex_color(text, color):
    return "\\textcolor{" + color + "}{" + text + "}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="jsonl to evaluate")
    parser.add_argument("--output", help="Report out")
    parser.add_argument("--colors", nargs="+", help="list of highlighting colors")


    args = parser.parse_args()
    with open(args.output, "wt") as h_out, open(args.input, "rt") as uhi:
        for line in uhi:
            jline = json.loads(line)
            os = ""
            for nu, tok in zip(jline["ngram_weight_used"], jline["tokens"]):
                if nu != -1:
                    os += wrap_latex_color(''.join(char for char in tok if ord(char) < 128), args.colors[nu]) + " "
                else:
                    os += ''.join(char for char in tok if ord(char) < 128) + " "
            h_out.write("\\textbf{"+"".join([str(f) for f in jline["scaling_factor"]]) + "} &" + os +"\\\\"+ "\n")
    
