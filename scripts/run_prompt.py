
from transformers import MistralForCausalLM, AutoTokenizer
from transformers import GenerationConfig, set_seed
from torch.nn import Softmax
from torch.nn.functional import log_softmax
from torch.distributions.categorical import Categorical
from transformers import LogitsWarper, LogitsProcessorList
import torch
import json
import re
from collections import defaultdict, Counter
import math
import argparse
import logging


def ngram(text, grams):
    return [text[i:i+grams] for i in range(len(text)-(grams-1))]


class Ngram:
    def __init__(self, seqs, grams):
        self.model = []
        
        for text in seqs:
            self.model += ngram(text+[2], grams)

        self.grams = grams-1

    def __call__(self, seq):
        if self.grams == 0:
            s_counts = defaultdict(int)
            for s in self.model:
                s_counts[s[0]] += 1
            return s_counts
        
        else:
            s_counts = defaultdict(int)
            for g in self.model:
                if " ".join([str(s) for s in seq[self.grams*-1:].tolist()]) == " ".join([str(s) for s in g[:self.grams]]):
                    s_counts[g[self.grams]] += 1
            return s_counts

            
class AdaptiveNGramWarper(LogitsWarper):
    def __init__(self, sequences, ng=4):
        self.models = []
        self.ng = ng
        self.scaling_factors = []
        self.weights = None
        self.was_scaled = [0]
        self.weight_info_used = []
        #self.cond_ents = []
        
        for ngs in range(1,self.ng+1):
            self.models.append(Ngram(sequences,ngs))
            
        
    def __call__(self, input_ids, scores):
        if self.weights is not None:
            self.was_scaled.append(self.weights[input_ids[0][-1]].item())

        mn=0
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        #self.cond_ents.append(-(normalized * p).nansum(-1, keepdim=True).item())
        
        for model, scale in zip(reversed(self.models), self.scaling_factors):
            #print(input_ids[0])
            counts = model(input_ids[0])
            #print(counts)
            if sum(counts) > 0:
                self.weight_info_used.append(mn)
                tf = sum(counts.values())
                normalized_counts = {t: freq/tf for t,freq in counts.items()}
                weights = [0]*32000
                for i, v in normalized_counts.items():
                    weights[i] = abs(scale/math.log(v)) if v != 1 else abs(scale/math.log(.999))

                self.weights = torch.tensor(weights)
                #print(self.weights)
                #input()
                #print(torch.add(scores, weights))
            
                return torch.add(scores,self.weights)
            mn+=1
        return scores
        
class BoWWarper(LogitsWarper):
    def __init__(self, bow):
        self.bow=bow

    def __call__(self, input_ids, scores):
        #print(input_ids.shape)
        return self.bow*scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Generative model name with Transformers generate interface")
    parser.add_argument("--text_input", help="Text to train ngram model on, jsonl")
    parser.add_argument("--prompts", help="Jsonl prompts file")
    parser.add_argument("--n_prompt_sets", type=int, help="Number of prompt sets to generate over")
    #parser.add_argument("--prompts", help="Text prompts file")
    #parser.add_argument("--story_prompts",help="JSON story prompts file")
    parser.add_argument("--scalings", help="JSON scalings file")
    parser.add_argument("--out", help="JSONl out file")
    parser.add_argument("--do_sample", type=int)
    #parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=1)

    args = parser.parse_args()
    log_format = "%(asctime)s::%(filename)s::%(message)s"
    logging.basicConfig(level='INFO', format=log_format)
    
    do_sample = bool(args.do_sample)

    set_seed(args.random_state)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MistralForCausalLM.from_pretrained(args.model)

    text_split = []
    with open(args.text_input, "rt") as s_in:
        for line in s_in:
            j_line = json.loads(line)
            text_split.append(j_line["text"])
        tokenized_text = tokenizer(text_split).input_ids
    
    with open(args.prompts, "rt") as p_i:
        #prompts = p_i.readlines()
        prompts = json.loads(p_i.read())

    #with open(args.story_prompts, "rt") as s_pi:
    #    story_prompts = json.loads(s_pi.read())

    with open(args.scalings, "rt") as s_i:
        scalings = json.loads(s_i.read())

    #combine author specific and story prompts
    #combined_prompts = []
    #for a_p in prompts:
    #    for s_p in story_prompts["prompts"]:
    #        print(a_p)
    #        print(s_p)
    #        combined_prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": a_p + s_p + ":"}], return_tensors="pt"))
            

    bg = AdaptiveNGramWarper(tokenized_text)

    with open(args.out, "wt") as j_out:
        #for model_inputs in combined_prompts:
        for b_p in prompts["prompts"][0:args.n_prompt_sets]:
            logging.info(b_p)
            model_inputs = tokenizer.apply_chat_template([{"role":"user", "content":b_p}], return_tensors="pt")
            for scaling in scalings:
                bg.was_scaled = [0]
                bg.weights = None
                bg.weight_info_used = []
                #bg.cond_ents = []
                bg.scaling_factors = scaling
                logging.info(bg.scaling_factors)
                out  = model.generate(model_inputs, max_new_tokens=256, output_logits=True, return_dict_in_generate=True, logits_processor=[bg], do_sample=args.do_sample, pad_token_id=tokenizer.eos_token_id)
                decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                logging.info(decoded)
                j_out.write(json.dumps({"text": decoded, "scaling_factor": bg.scaling_factors, "selected_was_weighted": bg.was_scaled, "ngram_weight_used": bg.weight_info_used})+"\n")


