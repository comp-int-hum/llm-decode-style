
from transformers import MistralForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from torch.nn import Softmax
from torch.distributions.categorical import Categorical
from transformers import LogitsWarper, LogitsProcessorList
import torch
import json
import re
from collections import defaultdict, Counter
import math
import argparse


#def sample_logits(logits, weights):
#    logits = logits * weights
#    sf = Softmax(dim=1)
#    pd = sf(logits)
#    print(pd.shape)
#    p = Categorical(pd)
#    return p.sample()



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
        
        for ngs in range(1,self.ng+1):
            self.models.append(Ngram(sequences,ngs))
            
        
    def __call__(self, input_ids, scores):
        if self.weights is not None:
            #print(self.weights)
            #print(input_ids[0][-1])
            self.was_scaled.append(self.weights[input_ids[0][-1]].item())
        for model, scale in zip(reversed(self.models), self.scaling_factors):
            counts = model(input_ids[0])
            if sum(counts) > 0:
                tf = sum(counts.values())
                normalized_counts = {t: freq/tf for t,freq in counts.items()}
                weights = [0]*32000
                for i, v in normalized_counts.items():
                    weights[i] = abs(scale/math.log(v)) if v != 1 else abs(scale/math.log(.999))

                self.weights = torch.tensor(weights)
                #print(weights)
                #print(scores)
                #print(torch.add(scores, weights))
            
                return torch.add(scores,self.weights)
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
    parser.add_argument("--text_input", help="Text to train ngram model on")
    parser.add_argument("--prompts", help="JSON prompts file")
    parser.add_argument("--scalings", help="JSON scalings file")
    parser.add_argument("--out", help="JSONl out file")
    parser.add_argument("--do_sample", type=int)
    parser.add_argument("--top_k", type=int, default=0)

    args = parser.parse_args()

    do_sample = bool(args.do_sample)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MistralForCausalLM.from_pretrained(args.model)
    
    with open(args.text_input, "rt") as s_in:
        text = s_in.read()
        text_split = [s for s in re.split("\n\n",text) if not any([c.isdigit() for c in s]) and len(s)>1]
        tokenized_text = tokenizer(text_split).input_ids
    
    with open(args.prompts, "rt") as p_i:
        prompts = json.loads(p_i.read())

    with open(args.scalings, "rt") as s_i:
        scalings = json.loads(s_i.read())

    model_inputs = tokenizer.apply_chat_template(prompts, return_tensors="pt")
    bg = AdaptiveNGramWarper(tokenized_text)

    with open(args.out, "wt") as j_out:
        for scaling in scalings:
            bg.was_scaled = [0]
            bg.weights = None
            bg.scaling_factors = scaling
            if args.top_k != 0:
                out = model.generate(model_inputs, max_new_tokens=500, output_logits=True, return_dict_in_generate=True, logits_processor=[bg], do_sample=args.do_sample, pad_token_id=tokenizer.eos_token_id, top_k=args.top_k)
            else:
                out  = model.generate(model_inputs, max_new_tokens=500, output_logits=True, return_dict_in_generate=True, logits_processor=[bg], do_sample=args.do_sample, pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(bg.scaling_factors)
            print(decoded)
            j_out.write(json.dumps({"text": decoded, "scaling_factor": bg.scaling_factors, "selected_was_weighted": bg.was_scaled})+"\n")

#prompt = [{"role":"user", "content": "You are a modern poet. Write a sonnet about love in the computer age:"}]

#prompt = [{"role": "user", "content": "You are an folksy storyteller. Tell me a five sentence story about a fox:"}]

#model_inputs = tokenizer.apply_chat_template(prompts, return_tensors="pt")

#out = model.generate(model_inputs, max_new_tokens=500, output_logits=True, return_dict_in_generate=True, logits_processor=[bg], do_sample=args.do_sample, pad_token_id=tokenizer.eos_token_id, top_k=args.top_k)#, top_k=50)
#out2 = model.generate(model_inputs, max_new_tokens=500, output_logits=True, return_dict_in_generate=True, do_sample=True, pad_token_id=tokenizer.eos_token_id, top_k=50) #top_k=50)


#gen = []
#for l in out.logits:
#    gen.append(sample_logits(l).item())
#print(gen)

#print(out.logits)

#print("Mod")
#print(tokenizer.batch_decode(out.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

#print("Orig")
#print(tokenizer.batch_decode(out2.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
#print(tokenizer.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False))
#out = model.forward(**model_inputs,  return_dict=True)
#print(out.logits[0][-1].shape)

