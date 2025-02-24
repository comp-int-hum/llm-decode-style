import argparse
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_in", help="perplexity jsonl")
    parser.add_argument("--r_in", help="perplexity reflection jsonl")
    parser.add_argument("--target", type=float, help="target value")
    parser.add_argument("--output", help="Chart out")

    args = parser.parse_args()
    rows = []

    p_df = pd.read_json(args.p_in, lines=True)
    r_df = pd.read_json(args.r_in, lines=True)

    new_index = ["".join([str(c) for c in i]) for i in r_df["scalings"]]

    def m_m_norm(series):
        return (series - np.min(series)) / (np.max(series) - np.min(series))

    c_df = pd.DataFrame({"scalings": new_index, "tPPL()": p_df["p"], "rPPL()": r_df["p"]})

    #abs of this and target
    print(c_df["tPPL()"])
    c_df["tPPL()"] = abs(c_df["tPPL()"]-args.target)
    print(c_df["tPPL()"])
    #tppl_norm = m_m_norm(c_df["tPPL()"].to_numpy())
    #rppl_norm = m_m_norm(c_df["rPPL()"].to_numpy())

    texts = []
    
    #plt.plot(rppl_norm, tppl_norm,"o")
    plt.plot(c_df["rPPL()"], c_df["tPPL()"], "o")
    for x, y, lbl in zip(c_df["rPPL()"], c_df["tPPL()"], c_df["scalings"]):
    #for x, y, lbl in zip(rppl_norm, tppl_norm, c_df["scalings"]):
        texts.append(plt.text(x+2, y, lbl, ha="left", va="center", fontsize=6))

    plt.xlabel("rPPL()")
    plt.ylabel("target text PPL() - tPPL()")
    ax = plt.subplot(111)
    ax.spines[['right', 'top']].set_visible(False)
    
    plt.savefig(args.output, dpi=300)


    
    #r_df.to_latex(args.output, index=False, float_format="%.2f")

