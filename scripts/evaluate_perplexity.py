
from transformers import MistralForCausalLM, AutoTokenizer
from torch.nn import Softmax
import torch
import json
import math
import argparse
import torch.nn.functional as F

#--model ${EVAL_MODEL} --input ${SOURCES[0]} --output ${TARGETS[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Generative model name with Transformers generate interface")
    parser.add_argument("--input", help="jsonl to evaluate perplexity on")
    parser.add_argument("--output", help="Per-word perplexity output")


    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MistralForCausalLM.from_pretrained(args.model)

    
    with open(args.input, "rt") as eval_in, open(args.output, "wt") as eval_out:
        for line in eval_in:
            j_line = json.loads(line)
            text = tokenizer([j_line["text"]], return_tensors="pt")
            labels = text.input_ids.clone()
            
            #prepared = tokenizer.prepare_for_model([built_out], add_special_tokens=False, return_tensors="pt")
            model_out = model(**text, return_dict=True, labels=labels)
            #sm = F.log_softmax(model_out.logits, dim=-1)
            #print(sm.shape)
            #print(text.input_ids.shape)

            shift_logits = model_out.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
            per_token_perp = torch.exp(loss)
            print(torch.mean(torch.exp(loss)))
            print(per_token_perp)
            
            
            #per_sm = torch.take(sm[0],text.input_ids[0])
            #print(per_sm)
            #print(torch.exp(-per_sm))
            #print(torch.mean(torch.exp(-per_sm)))
            #input()
            ppl = torch.exp(model_out.loss)
            print(ppl)
            j_line["perp"] = ppl.item()
            j_line["per_token_perp"] = per_token_perp.tolist()
            eval_out.write(json.dumps(j_line) + "\n")
