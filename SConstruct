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
    ("TEXTS", "", ["data/remus.txt"]),
    ("GEN_MODELS","", ["mistralai/Mistral-7B-Instruct-v0.2"]),
    ("PROMPTS", "", ["data/story_prompts.json"]),
    ("NG_SCALINGS", "", ["data/ng_scalings.json"]),
    ("GEN_MODEL_PARAMS", "", [{"do_sample": 0, "top_k":0}, {"do_sample": 1, "top_k":0}, {"do_sample": 1, "top_k": 50}]),
    ("USE_GRID","",0),
    ("GRID_TYPE","", "slurm"),
    ("GRID_GPU_COUNT","", 0),
    ("GRID_MEMORY", "", "64G"),
    #("GRID_TIME"
    #("FOLDS", "", 1),
)

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[steamroller.generate],
    
    BUILDERS={
        "RunPrompt" : Builder(
            action="python scripts/run_prompt.py --model ${MODEL} --text_input ${SOURCES[0]} --prompts ${SOURCES[1]} --scalings ${SOURCES[2]} --out ${TARGETS[0]}, --do_sample ${DO_SAMPLE} --top_k ${TOP_K}"
        )
    }
)

res = []
for model_type in env["GEN_MODELS"]:
    for text, prompt in zip(env["TEXTS"], env["PROMPTS"]):
        for scaling in env["NG_SCALINGS"]:
            for pn, params in enumerate(env["GEN_MODEL_PARAMS"]):
                print(params)
                res.append(env.RunPrompt(["work/${MODEL}/${TEXT}/${PARAM_NUM}/results.jsonl"],
                                [text,prompt,scaling],
                                MODEL = model_type,
                                TEXT = text,
                                DO_SAMPLE = params["do_sample"],
                                TOP_K = params["top_k"],
                                PARAM_NUM = pn
                                
                        )
                )