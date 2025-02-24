import argparse
import pandas as pd
import json



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p_in", nargs="+", help="perplexity jsonls to evaluate")
    parser.add_argument("--a_in", nargs="+", help="automatic measure jsonls to evaluate")
    parser.add_argument("--output", help="Latex out")
    parser.add_argument("--drop", default=[], nargs="+", help="Col to drop")

    args = parser.parse_args()
    rows = []

    for perps, res in zip(args.p_in, args.a_in):
        print(perps, res)
        with open(res, "rt") as r_in, open(perps, "rt") as p_in:
            for p_line, r_line in zip(p_in, r_in):
                rows.append(json.loads(p_line) | json.loads(r_line))

    r_df = pd.DataFrame(rows)
    r_df.rename(columns={'p': 'tPPL()'}, inplace=True)
    r_df.rename(columns={'D_avg': 'DAvg'}, inplace=True)
    if args.drop:
        args.drop.append("scaled_by_counts")
        for td in args.drop:
            if td in r_df.columns:
                r_df = r_df.drop(td, axis=1)
    print(r_df)
    r_df.to_latex(args.output, index=False, float_format="%.2f")

