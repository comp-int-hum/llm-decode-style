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
    parser.add_argument("--t_in", help="full ppl table to stitch")
    parser.add_argument("--json_folder", help="json rppl folder results to stitch in")
    parser.add_argument("--output", help="Latex out")
    parser.add_argument("--drop", default="Huck", help="columns to drop")

    args = parser.parse_args()

    base_df = make_df(args.t_in)
    base_df.columns = [col.strip() for col in base_df.columns]
    base_df = base_df.drop(args.drop,axis=1)
    base_df.set_index("scalings",inplace=True)

    jsons = {}
    for cname in base_df.columns:
        with open(args.json_folder+cname+".jsonl","rt") as jin:
            jl = []
            for line in jin:
                jline = json.loads(line)
                jl.append(jline["p"])
            jsons[cname] = jl
    s_df = pd.DataFrame.from_records(jsons)
    s_df.set_index(base_df.index,inplace=True)

    keys = base_df.columns

    ndfs = []

    for c1, c2 in zip(base_df.columns, s_df.columns):
        ndfs.append(pd.DataFrame({"PPL()": base_df[c1].values, "rPPL()": s_df[c2].values}))


    
    res_df = pd.concat(ndfs, axis=1,keys=keys)
    res_df.set_index(base_df.index, inplace=True)
    print(res_df)
    res_df.to_latex(args.output,
        index=True,
        escape=False,
        sparsify=True,
        multirow=True,
        multicolumn=True,
        multicolumn_format='c',
        position='p',
        bold_rows=True,
        float_format="%.2f")
    
    
