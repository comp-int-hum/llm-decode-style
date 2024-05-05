import argparse
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile",  help="JSONl in")
    args, rest = parser.parse_known_args()

    with open(args.infile, "rt") as j_in:
        for line in j_in:
            j_line = json.loads(line)
            print("ngram scalings: " + " ".join([str(i) for i in j_line["scaling_factor"]]))
            print("gold continuation: " + j_line["gold"])
            print(j_line["text"])
            print("")
