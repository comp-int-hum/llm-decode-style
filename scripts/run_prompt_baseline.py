
from transformers import MistralForCausalLM, AutoTokenizer
from transformers import GenerationConfig, set_seed
import torch
import json
import re
from collections import defaultdict, Counter
import math
import argparse
import logging
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Generative model name with Transformers generate interface")
    parser.add_argument("--prompts", help="Jsonl prompts file")
    parser.add_argument("--n_prompt_sets",type=int, help="Number of storyprompt prompt sets to generate over")
    parser.add_argument("--out", help="JSONl out file")
    parser.add_argument("--do_sample", type=int)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=1)

    args = parser.parse_args()

    log_format = "%(asctime)s::%(filename)s::%(message)s"
    logging.basicConfig(level='INFO', format=log_format)
    
    do_sample = bool(args.do_sample)

    set_seed(args.random_state)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MistralForCausalLM.from_pretrained(args.model)


    with open(args.out, "wt") as j_out, open(args.prompts, "rt") as p_i:
        for prompt in p_i.readlines()[0:args.n_prompt_sets]:
            j_prompt = json.loads(prompt)
            for p in j_prompt["prompts"]:
                logging.info(p)
                model_inputs = tokenizer.apply_chat_template([{"role": "user", "content": p}], return_tensors="pt")

                if args.top_k != 0:
                    out = model.generate(model_inputs, max_new_tokens=256, return_dict_in_generate=True, do_sample=args.do_sample, pad_token_id=tokenizer.eos_token_id, top_k=args.top_k)
                else:
                    out = model.generate(model_inputs, max_new_tokens=256, return_dict_in_generate=True, do_sample=args.do_sample, pad_token_id=tokenizer.eos_token_id)
                decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                logging.info(decoded)
                j_out.write(json.dumps({"text": decoded})+"\n")


