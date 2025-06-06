import argparse
import re
import json
import tarfile
import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_input", help="Writingprompts tgz in")
    parser.add_argument("--set_name", default="test")
    parser.add_argument("--base_prompt", type=str, help="Base prompt")
    #parser.add_argument("--n", type=int, default=1, help="Number of original prompts to draw from")
    parser.add_argument("--prompt_out", help="json out")
    parser.add_argument("--target_out", help="json out of target stories")
    
    args, rest = parser.parse_known_args()

    prompt_member = "writingPrompts/"+args.set_name+".wp_source"
    eval_member = "writingPrompts/"+args.set_name+".wp_target"

    print(args.prompts_input)

    with tarfile.open(args.prompts_input, "r") as t_in:
        p_m = t_in.extractfile(prompt_member).readlines()
        e_m = t_in.extractfile(eval_member).readlines()

    
    #print(re.sub(r"\s(?=\.|,|\:|;|\!|\?)","",re.sub(r"\[\s\S*\s\]","",p_m[0].decode("utf-8"))))
        
        
    with open(args.prompt_out, "wt") as t_out:
        t_out.write(json.dumps({"prompts": [args.base_prompt + re.sub(r"\s(?=\.|,|\:|;|\!|\?)","",re.sub(r"\[\s\S*\s\]", "", p.decode("utf-8"))).rstrip() for p in p_m]})) #\[\s\S*\s\]
        
    with open(args.target_out, "wt") as targ_out:
        for t_o in e_m:
            targ_out.write(json.dumps({"text": re.sub(r"\s(?=\.|,|\:|;|\!|\?)","",re.sub(r"\[\s\S*\s\]","",t_o.decode("utf-8")))})+"\n")
        #targ_out.write(json.dumps({"targets": [re.sub(r"\[\s\S*\s\]","",t.decode("utf-8")) for t in e_m[:args.n]]}))

