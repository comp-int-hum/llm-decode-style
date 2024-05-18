import argparse
import pandas as pd
import json



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", help="jsonls to evaluate")
    parser.add_argument("--output", help="Latex out")
    parser.add_argument("--drop", default=[], nargs="+", help="Col to drop")

    args = parser.parse_args()
    print(" ".join(args.drop))
    rows = []
    for res in args.inputs:
        with open(res, "rt") as r_in:
            for line in r_in:
                rows.append(json.loads(line))

    r_df = pd.DataFrame(rows)
    if args.drop:
        r_df = r_df.drop(" ".join(args.drop), axis=1)
    r_df.to_latex(args.output, index=False, float_format="%.2f")

