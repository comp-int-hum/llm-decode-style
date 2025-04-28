import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from adjustText import adjust_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_in", help="perplexity jsonl")
    parser.add_argument("--r_in", help="perplexity reflection jsonl")
    parser.add_argument("--target", help="target value")
    parser.add_argument("--output", help="Chart out")

    args = parser.parse_args()
    rows = []

    p_df = pd.read_json(args.p_in, lines=True)
    r_df = pd.read_json(args.r_in, lines=True)

    with open(args.target, "rt") as t_in:
        target = json.loads(t_in)["p"]

    new_index = ["".join([str(c) for c in i]) for i in r_df["scalings"]]

    def m_m_norm(series):
        return (series - np.min(series)) / (np.max(series) - np.min(series))

    c_df = pd.DataFrame({"scalings": new_index, "tPPL()": p_df["p"], "rPPL()": r_df["p"]})

    #abs of this and target
    c_df["tPPL()"] = abs(c_df["tPPL()"]-target)
    #tppl_norm = m_m_norm(c_df["tPPL()"].to_numpy())
    #rppl_norm = m_m_norm(c_df["rPPL()"].to_numpy())

    texts = []
    
    plt.plot(c_df["rPPL()"], c_df["tPPL()"], "o")
    for x, y, lbl in zip(c_df["rPPL()"], c_df["tPPL()"], c_df["scalings"]):
        texts.append(plt.text(x+2, y, lbl, ha="left", va="center", fontsize=6))

    plt.xlabel("rPPL()")
    plt.ylabel("target text PPL() - gPPL()")
    ax = plt.subplot(111)
    ax.spines[['right', 'top']].set_visible(False)
    
    plt.savefig(args.output, dpi=300)


    

