
from transformers import  AutoTokenizer

from torch.nn import Softmax
from torch.nn.functional import log_softmax
from torch.distributions.categorical import Categorical

import torch
import json
import re
from collections import defaultdict, Counter
import math
import argparse
import logging
import dill as pickle


from nltk.util import pad_sequence, everygrams
from nltk.lm.preprocessing import flatten
from nltk.lm import MLE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", help="Generative model name for tokenizer")
    parser.add_argument("--text", help="Text to train ngram model on, jsonl")
    parser.add_argument("--output")
    parser.add_argument("--max_n", type=int)

    args = parser.parse_args()

    log_format = "%(asctime)s::%(filename)s::%(message)s"
    logging.basicConfig(level='INFO', format=log_format)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    text_split = []
    with open(args.text, "rt") as s_in:
        for line in s_in:
            j_line = json.loads(line)
            text_split.append(list(pad_sequence([str(i_id) for i_id in tokenizer(j_line["text"], add_special_tokens=False).input_ids],
                                                pad_left=True,
                                                left_pad_symbol = str(tokenizer.bos_token_id),
                                                pad_right=True,
                                                right_pad_symbol = str(tokenizer.eos_token_id),
                                                n=args.max_n)))

        eg = list(everygrams(seq, max_len=3) for seq in text_split)
        vocab = list(flatten(s for s in text_split))
        ngm = MLE(args.max_n)
        ngm.fit(eg, vocab)

    with open(args.output, "wb") as m_out:
        pickle.dump(ngm, m_out)
