from transformers import GPT2LMHeadModel, AutoTokenizer, set_seed, MistralForCausalLM
from torch.nn import Softmax, CrossEntropyLoss
import torch
import json
import math
import argparse
import torch.nn.functional as F
import torch
import spacy
import re
from collections import defaultdict
import math
from tqdm import tqdm
from itertools import islice
from datasets import load_dataset
from run_prompt_nltk import AdaptiveNGramWarper
import pickle


def do_perp(encodings, model, stride, max_length, agw,t,ng):
    #max_length = model.config.n_positions
    lfunc = CrossEntropyLoss()
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            logits = model(input_ids).logits
            scaling_factors = [1,1,1,0]

            for grams, scale in zip([i for i in reversed(range(0,len(scaling_factors)))],scaling_factors):
                if scale != 0:
                    if grams != 0:
                        str_list = [str(i) for i in input_ids[0][0:-1][-1*grams:].tolist()]
                        counts = ng.counts[str_list].items()
                    else:
                        counts = ng.counts.unigrams.items()
                
            shift_logits = agw(input_ids[0][0:-1].unsqueeze(dim=0), logits[0][0:-1])
            shift_labels = target_ids[..., 1:].contiguous()

            loss = lfunc(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            
        nlls.append(loss)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai-community/gpt2-large", help="Generative model name with Transformers generate interface")
    parser.add_argument("--input", help="input text to test perplexity on")
    parser.add_argument("--ngram", help="ngram model to scale and evaluate perplexity with")
    parser.add_argument("--scalings", help="JSON file of scalings to calculate")
    parser.add_argument("--output", help="Perplexity output")
    parser.add_argument("--chunk_len", type=int, default=32)
    parser.add_argument("--stride",type=int, default=1)
    parser.add_argument("--device",default="cpu"),
    parser.add_argument("--random_state",type=int, default=20)



    args = parser.parse_args()
    set_seed(args.random_state)

    model = GPT2LMHeadModel.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with open(args.ngram, "rb") as ng_in:
        ngram = pickle.load(ng_in)

    with open(args.input, "rt") as eval_in, open(args.output,"wt") as eval_out:
        full_text = []
        for para in eval_in:
            p_text = json.loads(para)["text"]
            full_text.append(p_text)
               
        encodings = tokenizer("\n\n".join(full_text), return_tensors="pt").to(args.device)
        with open(args.scalings, "rt") as s_i:
            scalings = json.loads(s_i.read())

        for scaling in scalings:
            print(scaling)
            bg = AdaptiveNGramWarper(ngram, scaling, model.config.vocab_size, device=args.device)
            perp = do_perp(encodings, model, 1, 32, bg, tokenizer,ngram)
            eval_out.write(json.dumps({"scalings":scaling,"name":args.model, "p": perp})+"\n")

                    


