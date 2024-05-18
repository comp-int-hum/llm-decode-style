
from transformers import MistralForCausalLM, AutoTokenizer
from torch.nn import Softmax
import torch
import json
import math
import argparse
import torch.nn.functional as F
import torch
#--model ${EVAL_MODEL} --input ${SOURCES[0]} --output ${TARGETS[0]}")

def do_perp(segment, tokenizer, model):
    text = tokenizer(segment, return_tensors="pt")  
    labels = text.input_ids.clone()

    with torch.no_grad():
        model_out = model(**text, return_dict=True, labels=labels)
        shift_logits = model_out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
        per_token_perp = torch.exp(loss)
            
        ppl = torch.exp(model_out.loss)
        print(ppl)
    return ppl.item(), per_token_perp.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Generative model name with Transformers generate interface")
    parser.add_argument("--input", help="jsonl to evaluate perplexity on")
    parser.add_argument("--output", help="Per-word perplexity output")
    parser.add_argument("--mode", help="text, instruct, open, prompts")
    parser.add_argument("--chunk_len", default=256)


    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MistralForCausalLM.from_pretrained(args.model)

    
    with open(args.input, "rt") as eval_in, open(args.output, "wt") as eval_out:
        for line in eval_in:
            j_line = json.loads(line)

            if args.mode == "text":
                j_line["perp"], j_line["per_token_perp"] = do_perp(j_line["text"], tokenizer, model)
                eval_out.write(json.dumps(j_line) + "\n")
            elif args.mode == "instruct":
                j_line["prompt"], j_line["text"] = j_line["text"].split("[/INST]")
                j_line["perp"], j_line["per_token_perp"] = do_perp(j_line["text"], tokenizer, model)
                eval_out.write(json.dumps(j_line) + "\n")
        
