import argparse
import pandas as pd
import json
from io import StringIO


def make_df(fname):
    with open(fname) as f_in:
        text = ""
        for line in f_in:
            if "&" in line:
                text += line.replace("\\", "") + "\n"
    data = StringIO(text)
    return pd.read_csv(data, sep="&")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_in", nargs="+", help="tables to stitch")
    parser.add_argument("--names", nargs="+", help="text names used to replace targeted column")
    parser.add_argument("--output", help="Latex out")
    parser.add_argument("--col", help="Col to stitch on")

    args = parser.parse_args()

    dfs = []
    for table,name in zip(args.t_in, args.names):
        tdf = make_df(table)
        tdf.columns = [col.strip() for col in tdf.columns]
        #tdf.set_index("scalings", inplace=True)
        tdf = tdf.drop(columns=set(tdf.columns) - {args.col})
        tdf = tdf.rename(columns={args.col: name})
        print(tdf)
        dfs.append(tdf)
    f_df = pd.concat(dfs, axis="columns")
    print(f_df)
    f_df.to_latex(args.output, float_format="%.2f")

