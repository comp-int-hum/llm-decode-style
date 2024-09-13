
from transformers import GPT2LMHeadModel, AutoTokenizer, set_seed
from torch.nn import Softmax
import torch
import json
import math
import argparse
import torch.nn.functional as F
import torch
import spacy

from automatic_measures import zipf_coef, ngram_diversity, rep


#def do_perp(segment, tokenizer, model):
#    text = tokenizer(segment, return_tensors="pt", truncation=True)  
#    labels = text.input_ids.clone()

#    with torch.no_grad():
#        model_out = model(**text, return_dict=True, labels=labels)
#        shift_logits = model_out.logits[..., :-1, :].contiguous()
#        shift_labels = labels[..., 1:].contiguous()
#        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
        #per_token_perp = torch.exp(loss)
            
#        ppl = torch.exp(model_out.loss)
#        print(ppl)
#        return ppl
#        print(ppl)
#    return ppl.item(), per_token_perp.tolist()


def do_perp(encoded, model, stride):
    max_length = model.config.n_positions
    seq_len = encodings.input_ids.size(1)
    print(seq_len)
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

    return ppl, torch.stack(nlls).mean().item()**(1/seq_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai-community/gpt2", help="Generative model name with Transformers generate interface")
    parser.add_argument("--input", help="jsonl to evaluate perplexity on")
    parser.add_argument("--output", help="Per-word perplexity output")
    parser.add_argument("--mode", help="text, instruct, open, prompts")
    parser.add_argument("--chunk_len", type=int, default=256)
    parser.add_argument("--random_state",type=int, default=29)


    args = parser.parse_args()
    set_seed(args.random_state)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")


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
        
                
