import argparse
import json
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="JSONL text in")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    parser.add_argument("--train_size", type=float, default=0.9, help="Proportion of chunks to use for training")
    parser.add_argument("--random_state", type=int, default=1)
    args, rest = parser.parse_known_args()

    with open(args.input, "rt") as j_in:
        samples = [json.loads(line) for line in j_in]

    train, test = train_test_split(samples, train_size=args.train_size, random_state=args.random_state, shuffle=True)
    print("Train paragraphs: ", len(train))
    print("Test paragraphs: ", len(test))

    with open(args.outputs[0], "wt") as train_o:
        for s in train:
            train_o.write(json.dumps(s)+"\n")

    with open(args.outputs[1], "wt") as test_o:
        for s in test:
            test_o.write(json.dumps(s)+"\n")
