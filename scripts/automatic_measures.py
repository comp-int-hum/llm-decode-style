
import torch
import json
import math
import argparse
import spacy

import string

import numpy as np
import scipy
from collections import Counter

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


def ngram_diversity(tokens, top_n=4):
    def find_ngrams(t, n):
        return zip(*[t[i:] for i in range(n)])

    unique_len = []
    gram_len = []

    for n in range(1,top_n+1):
        print(n)
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
    
nlp = spacy.load("en_core_web_sm", exclude=["ner"])
nlp.tokenizer.rules = {key: value for key, value in nlp.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="text for original texts, gen for generated")
    parser.add_argument("--input", help="jsonl to evaluate")
    parser.add_argument("--output", help="Report out")


    args = parser.parse_args()
    
    
    with open(args.input, "rt") as eval_in, open(args.output, "wt") as eval_out:
        tokens = []
        if args.mode == "text":
            for line in eval_in:
                j_line = json.loads(line)
                for sent in nlp(j_line["text"].lower()).sents:
                    r_t = retokenize(nlp.tokenizer.explain(sent.text))
                    tokens += [r for r in r_t if any([c for c in r if c not in string.punctuation])]

            print(ngram_diversity(tokens))
            print(zipf_coef(tokens))
