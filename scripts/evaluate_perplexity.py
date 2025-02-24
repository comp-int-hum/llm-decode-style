
from transformers import GPT2LMHeadModel, AutoTokenizer, set_seed, MistralForCausalLM
from torch.nn import Softmax
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
#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def batch_perp(model, encodings, batch_size, stride, max_len, device, t):
    text_len = encodings.input_ids.size(1)
    lls = []

    for i in tqdm(range(0, text_len, batch_size * stride)):
        begin_locs, end_locs, trg_lens = [], [], []
        for j in range(batch_size):
            j = i + j * stride
            if j >= text_len:
                break
            begin_loc = max(j + stride - max_len, 0)
            end_loc = min(j + stride, text_len)
            trg_len = end_loc - j 

            begin_locs.append(begin_loc)
            end_locs.append(end_loc)
            trg_lens.append(trg_len)

        input_ids = [encodings.input_ids[:, b:e] for b, e in zip(begin_locs, end_locs)]
        target_end_locs = [sen.size(-1) for sen in input_ids]
        input_ids = [
            F.pad(sen, (0, max_len - sen.size(-1)), "constant", 0) for sen in input_ids
        ]
        input_ids = torch.stack(input_ids, dim=1).squeeze(0).to(device)

        target_ids = torch.ones_like(input_ids) * -100
        for i, (b, e) in enumerate(zip(trg_lens, target_end_locs)):
            labels = input_ids[i, -b:e].clone()
            target_ids[i, -b:e] = labels

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs["loss"] * sum(trg_lens)

        lls.append(log_likelihood)

    ppl = torch.exp(sum(torch.stack(lls) / end_locs[-1]))
    print(ppl.item())
    return ppl.item()

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
            print(neg_log_likelihood.item())
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=["openai-community/gpt2-large"], nargs="+", help="Generative model name with Transformers generate interface")
    parser.add_argument("--input", help="jsonl to evaluate perplexity on")
    parser.add_argument("--output", help="Perplexity output")
    parser.add_argument("--mode", help="text, targets, control, prompts")
    parser.add_argument("--chunk_len", type=int, default=32)
    parser.add_argument("--stride",type=int, default=1)
    parser.add_argument("--device",default="cpu"),
    parser.add_argument("--random_state",type=int, default=20)



    args = parser.parse_args()
    set_seed(args.random_state)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with open(args.output, "wt") as eval_out, open(args.input, "rt") as eval_in:
        for mname in args.model:
            tokenizer = AutoTokenizer.from_pretrained(mname)
            if mname == "openai-community/gpt2-large":
                model = GPT2LMHeadModel.from_pretrained(mname).to(args.device)
                ml = model.config.n_positions


            if args.mode == "verify":
                test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
                perp = batch_perp(model, encodings, 512, 1, 32, args.device, tokenizer)
                print(perp)
                input()

            if args.mode == "text":
                full_text = []
                for para in eval_in:
                    p_text = json.loads(para)["text"]
                    full_text.append(p_text)
                #encodings = tokenizer("\n\n".join(full_text), return_tensors="pt", return_overflowing_tokens=True, stride=31, max_length=32).to(args.device)
                encodings = tokenizer("\n\n".join(full_text), return_tensors="pt").to(args.device)
                #perp = do_perp(encodings, model, args.stride, args.chunk_len)
                #perp = alt_perp(encodings, model)
                perp = batch_perp(model, encodings, 256, 1, 32, args.device, tokenizer)
                eval_out.write(json.dumps({"name":mname, "p": perp})+"\n")
                    
            elif args.mode == "prompt":
                gen_text_by_scaling = defaultdict(list)
                for gen in eval_in:
                    j_gen = json.loads(gen)
                    gen_text_by_scaling["".join([str(i) for i in j_gen["scaling_factor"]])].append(j_gen["text"])
                for scaling, texts in gen_text_by_scaling.items():
                    print(scaling)
                    encodings = tokenizer("\n\n".join(texts), return_tensors="pt").to(args.device)
                    #perp = do_perp(encodings, model, args.stride, args.chunk_len)
                    perp = batch_perp(model, encodings, 256, 1, 32, args.device, tokenizer)
                    eval_out.write(json.dumps({"scalings":scaling,"name":mname, "p": perp})+"\n")

            elif args.mode == "control":
                def nread(f, n):
                    lines = []
                    for line in f:
                        lines.append(line)
                        if len(lines) == n:
                            yield lines
                            lines = []
                    if lines:
                        yield lines

                p1 = []
                p2 = []
                p3 = []
                for gen in nread(eval_in, 3):
                    p1.append(json.loads(gen[0])["text"])
                    p2.append(json.loads(gen[1])["text"])
                    p3.append(json.loads(gen[2])["text"])
                prompts = [p1,p2,p3]
                for full_text in prompts:
                    encodings = tokenizer("\n\n".join(full_text), return_tensors="pt").to(args.device)
                    #perp, _ = do_perp(encodings, model, args.chunk_len, ml)
                    #perp = do_perp(encodings, model, args.stride, args.chunk_len)
                    perp = batch_perp(model, encodings, 512, 1, 32, args.device, tokenizer)
                    eval_out.write(json.dumps({"name":mname, "p": perp})+"\n")
                    

            elif args.mode =="targets":
                full_targets = []
                for target in eval_in:
                    full_targets.append(json.loads(target)["text"].replace("<newline>", ""))
                encodings = tokenizer("\n\n".join(full_targets), return_tensors="pt").to(args.device)
                #perp, _ = do_perp(encodings, model, args.chunk_len, ml)
                #perp = do_perp(encodings, model, args.stride, args.chunk_len)
                perp = batch_perp(model, encodings, 256, 1, 32, args.device, tokenizer)
                eval_out.write(json.dumps({"name":mname, "p": perp})+"\n")

                """
                perps = []
                for target in eval_in:
                    enc = tokenizer(json.loads(target)["text"].replace("<newline>", ""),return_tensors="pt")
                    perps.append(batch_perp(model, enc, 512, 1, 32, args.device, tokenizer))
                    print(sum(perps)/len(perps))
                """

