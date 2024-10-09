
import torch
import json
import math
import argparse
import spacy

import string

import numpy as np
import scipy
from collections import Counter
from collections import defaultdict
from transformers import AutoTokenizer

def zipf_coef(tokens):
    word_counter = Counter(tokens)
    count_counter = Counter(word_counter.values())

    count_counter = sorted(
        count_counter.items(),
        key = lambda pair: pair[1],
        reverse=True)
    
    word_counts, freq_of_counts = np.asarray(count_counter).T

    def nll_zipf(s, word_counts, freq_of_counts):
        log_probs = -s * np.log(word_counts)
        document_length = sum(freq_of_counts)
        all_possible_counts = np.arange(1, document_length + 1)
        log_probs -= scipy.special.logsumexp(-s * np.log(all_possible_counts))
        return -np.sum(freq_of_counts * log_probs)

    loss_fn = lambda s: nll_zipf(s, word_counts, freq_of_counts)
    s_best = scipy.optimize.minimize_scalar(loss_fn, [0.1, 3.0])
    return s_best.x

def find_ngrams(t,n):
    return zip(*[t[i:] for i in range(n)])


def ngram_diversity(tokens, top_n=4):

    #def find_ngrams(t,n):
    #    return zip(*[t[i:] for i in range(n)])

    unique_len = []
    gram_len = []

    for n in range(1,top_n+1):
        grams = [g for g in find_ngrams(tokens, n)]
        gram_len.append(len(grams))
        unique_len.append(len(set(grams)))

    
    return float(sum(unique_len))/sum(gram_len)


def retokenize(orig_tok):
	found_prefix = None
	new_toks = []
	for t in orig_tok:
		if t[0] == "SUFFIX" and t[1] in ["’","'","‘"]:
			new_toks[-1] = new_toks[-1]+t[1]
		elif t[0] == "PREFIX" and t[1] in ["’","'","‘"]:
			found_prefix = t[1]
		else:
			if found_prefix:
				new_toks.append(found_prefix+t[1])
				found_prefix = None
			else:
				new_toks.append(t[1])
	return new_toks

def c_tokenize(segment, mode, tk):
    if mode == "spacy":
        r_t = []
        for sent in tk(segment).sents:
            r_t += retokenize(tk.tokenizer.explain(sent.text))
    else:
        r_t = tk.tokenize(segment)

    return r_t
        

class eval_gen:
    def __init__(self, tname):
        self.tfile = open(tname, "rt")
    def __iter__(self):
        self.tfile.seek(0)
        for line in self.tfile:
            yield line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="text for original texts, open for open generation, instruct for prompted")
    parser.add_argument("--tokenizer", nargs="+", default=["spacy"])
    parser.add_argument("--input", help="jsonl to evaluate")
    parser.add_argument("--output", help="Report out")


    args = parser.parse_args()
    with open(args.output, "wt") as eval_out:
        eval_in = eval_gen(args.input)
        for tname in args.tokenizer:
            if tname == "spacy":
                tk = spacy.load("en_core_web_sm", exclude=["ner"])
                tk.tokenizer.rules = {key: value for key, value in tk.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}
            else:
                tk = AutoTokenizer.from_pretrained(tname)

            if args.mode == "text":
                tokens = []
                d = []
                reps = []
                for line in eval_in:
                    j_line = json.loads(line)
                    segment = c_tokenize(j_line["text"], tname, tk)
                    tokens += segment
                    d.append(ngram_diversity(segment))
                    #reps.append(rep(segment))
                eval_out.write(json.dumps({"name": tname, "N": len(tokens), "Zipf": zipf_coef(tokens), "D_avg": sum(d)/len(d), "D_full": ngram_diversity(tokens)})+"\n")

            elif args.mode == "instruct":
                res_by_scaling = defaultdict(list)
                for line in eval_in:
                    j_line = json.loads(line)
                    generated = c_tokenize(j_line["text"].split("[/INST]")[1], tname, tk)
                    print(generated)
                    print(len(generated))
                    print(len(j_line["selected_was_weighted"]))
                    print(len(j_line["ngram_weight_used"]))
                    input()
                    
    
"""            
            with open(args.output, "wt") as j_out:
                j_out.write(json.dumps({"N": len(tokens), "Zipf":zipf_coef(tokens), "D": sum(d)/len(d), "Perplexity": sum(avg_perp)/len(avg_perp), "Per Token Perp":sum(per_token_perp)/len(per_token_perp) })+"\n")
                
        elif args.mode == "instruct":
            res_by_scaling = defaultdict(list)
            for line in eval_in:
                res = {"per_token_perp":[]}
                j_line = json.loads(line)
                res["text"] = j_line["text"]
                res["perp"] = j_line["perp"]
                res["per_token_perp"] += j_line["per_token_perp"]
                generated = c_tokenize(j_line["text"], args.tokenizer, tk)
                generated_orig = tk(j_line["text"], add_special_tokens=False).input_ids
                scaling = "".join([str(i) for i in j_line["scaling_factor"]])
                res["generated"] = generated
                if len(generated_orig) > len(j_line["ngram_weight_used"]):
                    generated_orig = generated_orig[1:]
                res["ngram_tokens"] = [tk.decode(e) for i,e in zip(j_line["selected_was_weighted"], generated_orig) if i > 0]
                res["unscaled_tokens"] = [tk.decode(e) for i,e in zip(j_line["selected_was_weighted"], generated_orig) if i == 0]
                res_by_scaling[scaling].append(res)
            with open(args.output, "wt") as j_out:
                for s, res in res_by_scaling.items():
                    tokens = []
                    ngram_tokens = []
                    per_token_perp = []
                    unscaled_tokens = []
                    for r in res:
                        tokens += r["generated"]
                        ngram_tokens += r["ngram_tokens"]
                        unscaled_tokens += r["unscaled_tokens"]
                        per_token_perp += r["per_token_perp"]
                    ds = [ngram_diversity(r["generated"]) for r in res]
                    perps = [r["perp"] for r in res]
                    ngram_tokens = Counter(ngram_tokens).most_common(40)
                    unscaled_tokens = Counter(unscaled_tokens).most_common(40)
                    j_out.write(json.dumps({"Scaling": s, "Zipf": zipf_coef(tokens), "D": sum(ds)/len(ds), "Perplexity": sum(perps)/len(perps), "Per Token Perp": sum(per_token_perp)/len(per_token_perp), "Ngram tokens": ngram_tokens, "Unscaled tokens":unscaled_tokens})+"\n")

            

        elif args.mode == "open":
            for line in eval_in:
                j_line = json.loads(line)
                if len(j_line["selected_was_weighted"]) > 1:
                    print(j_line["scaling_factor"])
                    print("Prefix: " + j_line["prefix"])
                    print("Cont: " + j_line["text"][len(j_line["prefix"][4:]):])
                    print("Gold: " + j_line["gold"])
                    input()
"""
