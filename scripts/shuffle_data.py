import argparse
import json
from sklearn.model_selection import train_test_split
import logging

log_format = "%(asctime)s::%(filename)s::%(message)s"
logging.basicConfig(level='INFO', format=log_format)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="JSONL text in")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    parser.add_argument("--train_size", type=float, default=0.9, help="Proportion of chunks to use for training")
    parser.add_argument("--random_state", type=int, default=1)
    args, rest = parser.parse_known_args()

    with open(args.input, "rt") as j_in:
        samples = [json.loads(line) for line in j_in]
        s_indices = iter([i for i in range(0, len(samples))])
        paired_samples = [[x, y] for x,y in zip(s_indices, s_indices)]
            
    train, test = train_test_split(paired_samples, train_size = args.train_size, random_state=args.random_state, shuffle=True)
    print(test)

    
    skip = []
    with open(args.outputs[1], "wt") as test_o:
        for ts in test:
            skip += ts
            test_o.write(json.dumps({"text":samples[ts[0]]["text"], "gold":samples[ts[1]]["text"]})+"\n")
    with open(args.outputs[0], "wt") as train_o:
        for i,trs in enumerate(samples):
            if i not in skip:
                train_o.write(json.dumps(trs)+"\n")
            else:
                print(i)
    #input()
    #train, test = train_test_split(paired_samples, train_size=args.train_size, random_state=args.random_state, shuffle=True)
    #logging.info(len(train))
    #logging.info(len(test))
    #print("Train paragraphs: ", len(train))
    #print("Test paragraphs: ", len(test))

    #with open(args.outputs[0], "wt") as train_o:
    #    for s in train:
    #        train_o.write(json.dumps({"text": s[0]["text"], "gold":s[1]["text"]})+"\n")

    #with open(args.outputs[1], "wt") as test_o:
    #    for s in test:
    #        for t in s:
    #            test_o.write(json.dumps(t)+"\n")
