import json
import argparse
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Combined base and story prompts")
    parser.add_argument("--author_stems", help="Author specific control prompt modifications")
    parser.add_argument("--output", help="jsonl combined prompts")

    args = parser.parse_args()

    log_format = "%(asctime)s::%(filename)s::%(message)s"
    logging.basicConfig(level='INFO', format=log_format)
    
    
    with open(args.author_stems, "rt") as p_i:
        prompts = p_i.readlines()

    with open(args.input, "rt") as s_pi:
        sprompts = json.loads(s_pi.read())

    with open(args.output, "wt") as s_po:
        for s_p in sprompts["prompts"]:
            generated_author_prompts = [a_p.replace("\n", " ") + s_p for a_p in prompts]
            s_po.write(json.dumps({"prompts": generated_author_prompts})+"\n")
            
        #story_prompts = json.loads(s_pi.read())
    

    #combine base and story prompts along with scalings
    #combined_prompts = []
        

    #combine author specific and story prompts
    #for a_p in prompts:
    #    for s_p in story_prompts["prompts"][0:args.n_prompt_sets]:
    #        combined_prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": a_p + s_p + ":"}], return_tensors="pt"))
            
