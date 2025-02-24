
from transformers import MistralForCausalLM, AutoTokenizer, AutoModel, LlamaForCausalLM
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
import dill as pickle
from nltk import lm
            
class AdaptiveNGramWarper(LogitsWarper):
    def __init__(self,  ngram, scaling_factors, vocab_size, ng=4, device="cpu"):
        self.models = ngram
        self.ng = [i for i in reversed(range(0,len(scaling_factors)))]
        self.scaling_factors = scaling_factors
        self.was_scaled = []
        self.weight_info_used = []
        self.prev_weights = None
        self.vocab_size = vocab_size
        self.device = device
        
            
        
    def __call__(self, input_ids, scores):
        if self.prev_weights is not None:
            self.was_scaled.append(self.prev_weights[input_ids[0][-1]].item())
        else:
            self.was_scaled.append(None)
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        for grams, scale in zip(self.ng, self.scaling_factors):
            if scale != 0:
                if grams != 0:
                    str_list = [str(i) for i in input_ids[0][-1*grams:].tolist()]
                    counts = self.models.counts[str_list].items()
                else:
                    counts = self.models.counts.unigrams.items()
                weights = [0]*self.vocab_size
                tf = sum(c for t, c in counts)
                if tf > 0:
                    self.weight_info_used.append(grams)
                    for i, c in counts:
                        #print(i,c, tf)
                        weights[int(i)] = abs(scale/math.log(c/tf)) if c != tf else abs(scale/math.log(.999))
                    weights = torch.tensor(weights).to(self.device)
                    scores = torch.add(weights, scores)
                    self.prev_weights = weights
                    return scores
        self.weight_info_used.append(-1)        
        self.prev_weights = None
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
    parser.add_argument("--ngram", help="NLTK Ngram model, pkl")
    parser.add_argument("--prompts", help="Jsonl prompts file")
    #parser.add_argument("--story_prompts",help="JSON story prompts file")
    parser.add_argument("--n_prompt_sets",type=int, help="Number of storyprompt prompt sets to generate over")
    parser.add_argument("--scalings", help="JSON scalings file")
    parser.add_argument("--out", help="JSONl out file")
    parser.add_argument("--do_sample", type=int)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--random_state", type=int, default=1)

    args = parser.parse_args()

    log_format = "%(asctime)s::%(filename)s::%(message)s"
    logging.basicConfig(level='INFO', format=log_format)
    
    do_sample = bool(args.do_sample)
    if "Llama" in args.model:
        print("Llama")

    set_seed(args.random_state)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = LlamaForCausalLM.from_pretrained(args.model) if "Llama" in args.model else MistralForCausalLM.from_pretrained(args.model)
    vocab_size = 128256 if "Llama" in args.model else 32000
    
    with open(args.ngram, "rb") as ng_in:
        ngram = pickle.load(ng_in)

    with open(args.prompts, "rt") as p_i:
        prompts = json.loads(p_i.read())
    

    with open(args.scalings, "rt") as s_i:
        scalings = json.loads(s_i.read())
    

    #combine base and story prompts along with scalings
    #combined_prompts = []
        

    #combine author specific and story prompts
    #for a_p in prompts:
    #    for s_p in story_prompts["prompts"][0:args.n_prompt_sets]:
    #        combined_prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": a_p + s_p + ":"}], return_tensors="pt"))
            

    #bg = AdaptiveNGramWarper(ngram)

    with open(args.out, "wt") as j_out:
        for b_p in prompts["prompts"][0:args.n_prompt_sets]:
            logging.info(b_p)
            model_inputs = tokenizer.apply_chat_template([{"role": "user", "content": b_p}], tools=None, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            for scaling in scalings:
                logging.info(scaling)
                bg = AdaptiveNGramWarper(ngram, scaling, vocab_size)
                logging.info(bg.ng)
                out  = model.generate(model_inputs, max_new_tokens=256, output_logits=True, return_dict_in_generate=True, logits_processor=[bg], do_sample=args.do_sample, pad_token_id=tokenizer.eos_token_id, temperature=args.temp)
                #decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                #logging.info(decoded)
                generated = tokenizer.batch_decode(out.sequences[:, model_inputs.shape[1]:], clean_up_tokenization_spaces=False, skip_special_tokens=True)[0]
                tokens = tokenizer.convert_ids_to_tokens(out.sequences[:, model_inputs.shape[1]:].tolist()[0])
                prompt = tokenizer.batch_decode(model_inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                logging.info(generated)
                logging.info("Token len: {}".format(len(tokens)))
                logging.info("SWW len: {}".format(len(bg.was_scaled)))
                logging.info("NGWU len: {}".format(len(bg.weight_info_used)))
                j_out.write(json.dumps({"text": generated, "tokens": tokens, "prompt": prompt, "scaling_factor": bg.scaling_factors, "selected_was_weighted": bg.was_scaled, "ngram_weight_used": bg.weight_info_used})+"\n")


