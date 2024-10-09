
from transformers import GPT2LMHeadModel, AutoTokenizer, set_seed, MistralForCausalLM
from torch.nn import Softmax
import torch
import json
import math
import argparse
import torch.nn.functional as F
import torch
import spacy

from automatic_measures import eval_gen


def do_perp(encodings, model, stride, max_length):
    #max_length = model.config.n_positions
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break  
    ppl = torch.exp(torch.stack(nlls).mean())

    return ppl.item(), torch.stack(nlls).mean().item()**(1/seq_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=["openai-community/gpt2-large"], nargs="+", help="Generative model name with Transformers generate interface")
    parser.add_argument("--input", help="jsonl to evaluate perplexity on")
    parser.add_argument("--output", help="Per-word perplexity output")
    parser.add_argument("--mode", help="text, instruct, open, prompts")
    parser.add_argument("--chunk_len", type=int, default=512)
    parser.add_argument("--random_state",type=int, default=29)


    args = parser.parse_args()
    set_seed(args.random_state)

    with open(args.output, "wt") as eval_out:
        eval_in = eval_gen(args.input)
        for mname in args.model:
            print(mname)
            tokenizer = AutoTokenizer.from_pretrained(mname)
            if mname == "openai-community/gpt2-large":
                model = GPT2LMHeadModel.from_pretrained(mname)
                ml = model.config.n_positions
            elif mname == "mistralai/Mistral-7B-v0.1":
                model = MistralForCausalLM.from_pretrained(mname)
                ml = 4096


            full_text = []
            avg_pt_ppl = []


            if args.mode =="text":
                for para in eval_in:
                    p_text = json.loads(para)["text"]
                    full_text.append(p_text)
                    toks = tokenizer(p_text, return_tensors="pt")
                    _ , ppt = do_perp(toks, model, ml, ml)
                    avg_pt_ppl.append(ppt)
                full_text = "\n\n".join(full_text)
                encodings = tokenizer(full_text, return_tensors="pt")
                perp, full_ppt = do_perp(encodings, model, args.chunk_len, ml)
                print(perp)
                print(full_ppt)
                print(sum(avg_pt_ppl)/len(avg_pt_ppl))
                eval_out.write(json.dumps({"name":mname, "p": perp, "full_ppt": full_ppt, "avg_ppt": sum(avg_pt_ppl)/len(avg_pt_ppl)})+"\n")
                            
    
"""
    zc = []
    nd = []
    tk = spacy.load("en_core_web_sm", exclude=["ner"])
    with open(args.input, "rt") as eval_in, open(args.output, "wt") as eval_out:
        if args.mode == "text":
            #full_text = "\n\n".join([json.loads(para)["text"] for para in eval_in])
            para_text = [json.loads(para)["text"] for para in eval_in]
            for p in para_text:
                #toks = [t.text for t in tk(p)]
                toks = tokenizer(para_text).input_ids[0]
                zc.append(zipf_coef(toks))
                nd.append(ngram_diversity(toks))
            #encodings = tokenizer(full_text, return_tensors="pt")
            print(sum(nd)/len(nd))
            print(sum(zc)/len(zc))
            #print(do_perp(encodings, model, args.chunk_len))
            input()

            #elif args.mode == "instruct":
            #    j_line["prompt"], j_line["text"] = j_line["text"].split("[/INST]")
            #    j_line["perp"], j_line["per_token_perp"] = do_perp(j_line["text"], tokenizer, model)
            #    eval_out.write(json.dumps(j_line) + "\n")
"""
        
                
