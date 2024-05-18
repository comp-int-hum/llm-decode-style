import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
import steamroller

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

# Variables control various aspects of the experiment.  Note that you have to declare
# any variables you want to use here, with reasonable default values, but when you want
# to change/override the default values, do so in the "custom.py" file (see it for an
# example, changing the number of folds).
vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("TEXTS", "", ["remus", "dooley"]),
    ("DATA_DIR","","data/"),
    #("TEXTS", "", ["data/remus.txt", "data/dooley.txt"]),
    ("GEN_MODELS","", ["mistralai/Mistral-7B-Instruct-v0.2"]),
    ("STORYPROMPTS_LOC","","data/writingPrompts.tar.gz"),
    ("N_STORY_PROMPTS", "", 2),
    ("EVAL_MODEL","", ["mistralai/Mistral-7B-v0.1"]),
    ("NG_SCALINGS", "", ["ng_scalings.json"]),
    ("GEN_MODEL_PARAMS", "", [{"do_sample": 1, "top_k":0, "temperature": 0.0}, {"do_sample": 1, "top_k": 40, "temperature":0.0}, {"do_sample": 1, "top_k":0, "temperature":0.5}]),
    ("RANDOM_STATE", "", 20),
    ("TRAIN_SIZE", "", 0.95),
    ("USE_GRID","",1),
    ("GRID_TYPE","", "slurm"),
    #("GRID_QUEUE","", "a100"),
    #("GRID_ACCOUNT","", "tlippi1_gpu"),
    ("GRID_GPU_COUNT","", 0),
    ("GRID_MEMORY", "", "64G"),
    ("GRID_LABEL", "", "NGStyle"),
    ("GRID_TIME", "", "48:00:00"),

    #("FOLDS", "", 1),
)

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[steamroller.generate],
    
    BUILDERS={
        "CreateData": Builder(
            action="python scripts/create_json_text.py --text_input ${SOURCES[0]} --out ${TARGETS[0]}"
        ),
	"LoadPrompts": Builder(action="python scripts/generate_prompts.py --prompts_input ${STORYPROMPTS_LOC} --n ${N_STORY_PROMPTS} --prompt_out ${TARGETS[0]} --target_out ${TARGETS[1]}"),
        "RunPrompt": Builder(action="python scripts/run_prompt.py --model ${MODEL} --text_input ${SOURCES[0]} --prompts ${SOURCES[1]} --scalings ${SOURCES[2]} --out ${TARGETS[0]} --do_sample ${DO_SAMPLE} --top_k ${TOP_K} --random_state ${RANDOM_STATE} --story_prompts ${SOURCES[3]}"),
        "SplitForOpenEnded": Builder(action="python scripts/shuffle_data.py --input ${SOURCES[0]} --outputs ${TARGETS} --train_size ${TRAIN_SIZE} --random_state ${RANDOM_STATE}"),
        "RunOpenEnded": Builder(action="python scripts/run_open_ended.py --model ${MODEL} --train ${SOURCES[0]} --test ${SOURCES[1]} --scalings ${SOURCES[2]} --out ${TARGETS[0]} --do_sample ${DO_SAMPLE} --top_k ${TOP_K} --temperature ${TEMP} --random_state ${RANDOM_STATE}"),
        "EvalPerplexity": Builder(action="python scripts/evaluate_perplexity.py --model ${EVAL_MODEL} --input ${SOURCES[0]} --output ${TARGETS[0]} --mode ${MODE}"),
	"AutomaticEval": Builder(action="python scripts/automatic_measures.py --input ${SOURCES[0]} --output ${TARGETS[0]} --mode ${MODE}"),
        "ProduceTable": Builder(action="python scripts/produce_table.py --inputs ${SOURCES} --output ${TARGETS[0]} --drop ${DROP}")
    }
)

res = []
text_ins = []
text_res = []


story_prompts, prompt_targets = env.LoadPrompts(["work/prompts.jsonl", "work/targets.jsonl"], [])
target_perps = env.EvalPerplexity(["work/target_perplexities.jsonl"], [prompt_targets], mode="text")
text_res.append(env.AutomaticEval(["work/target_automatic_eval.jsonl"], [target_perps], mode="text"))
for text in env["TEXTS"]:
    text_ins.append(env.CreateData(["work/${TEXT}/text.jsonl"], [env["DATA_DIR"]+text+".txt"], TEXT=text))
    tp = env.EvalPerplexity(["work/${TEXT}/perplexities.jsonl"], [text_ins[-1]], TEXT=text, mode="text")
    text_res.append( env.AutomaticEval(["work/${TEXT}/automatic_eval.jsonl"], [tp], TEXT=text, mode="text"))

text_table = env.ProduceTable(["work/results/text_table.tex"], text_res)

for model_type in env["GEN_MODELS"]:
    for text_name, text in zip(env["TEXTS"], text_ins):
        for scaling in env["NG_SCALINGS"]:
            for pn, params in enumerate(env["GEN_MODEL_PARAMS"]):
                gen = env.RunPrompt(["work/${MODEL}/${TEXT_NAME}/${PARAM_NUM}/results.jsonl"],
                                [text,env["DATA_DIR"]+text_name+"_prompts.txt",env["DATA_DIR"]+scaling, story_prompts],
                                MODEL = model_type,
                                TEXT = text,
                                DO_SAMPLE = params["do_sample"],
                                TOP_K = params["top_k"],
                                PARAM_NUM = pn,
                                TEXT_NAME = text_name       
                        )
                gen_perps = env.EvalPerplexity(["work/${MODEL}/${TEXT_NAME}/${PARAM_NUM}/perplexities.jsonl"], [gen],
                                MODEL=model_type,
                                TEXT=text,
                                PARAM_NUM = pn,
                                TEXT_NAME = text_name,
                                MODE = "instruct"
                        )
                res.append(env.AutomaticEval(["work/${MODEL}/${TEXT_NAME}/${PARAM_NUM}/automatic_eval.jsonl"], [gen_perps],
                               MODEL=model_type,
                               TEXT=text,
                               PARAM_NUM = pn,
                               TEXT_NAME = text_name,
                               MODE = "instruct"
                        )
                )
                table=env.ProduceTable(["work/results/${MODEL}/${TEXT_NAME}_${PARAM_NUM}_table.tex"], res[-1], MODEL=model_type, TEXT_NAME=text_name, PARAM_NUM=pn, DROP="Ngram tokens")

"""
for open_ended in env["EVAL_MODEL"]:
    for text,text_name in zip(text_ins, env["TEXTS"]):
        #test train split
        train, test = env.SplitForOpenEnded(["work/${MODEL}/${TEXT_NAME}/train.jsonl", "work/${MODEL}/${TEXT_NAME}/test.jsonl"], [text], MODEL=open_ended, TEXT=text, TEXT_NAME=text_name)
        for scaling in env["NG_SCALINGS"]:
            for pn, params in enumerate(env["GEN_MODEL_PARAMS"]):
                open_res.append(env.RunOpenEnded(["work/${MODEL}/${TEXT_NAME}/${PARAM_NUM}/results.jsonl"],
                [train,test, env["DATA_DIR"]+scaling],
                MODEL = open_ended,
                TEXT = text,
                DO_SAMPLE = params["do_sample"],
                TOP_K = params["top_k"],
                TEMP = params["temperature"],
                PARAM_NUM = pn,
                TEXT_NAME = text_name,
                #GRID_GPU_COUNT=1,
                #GRID_ACCOUNT="tlippin1_gpu",
                #GRID_QUEUE="a100"
               )
        )
"""


#gen_perps.append(env.EvalPerplexity(["work/${MODEL}/${TEXT}/${PARAM_NUM}/perplexities.jsonl"], [res[-1]], TEXT=text, MODEL=model_type, PARAM_NUM=pn))
