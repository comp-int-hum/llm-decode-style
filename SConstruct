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

# Variables control various aspects of the experiment.	Note that you have to declare
# any variables you want to use here, with reasonable default values, but when you want
# to change/override the default values, do so in the "custom.py" file (see it for an
# example, changing the number of folds).
vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("TEXTS","", ["remus","dooley","Biglow","Todd","Remus","Julius"]),
    #("TEXTS", "", ["remus", "dooley"]),
    #("CSV_TEXT_CHARS", "", ["Biglow","Todd","Remus", "Julius"]),
    ("TEXT_LOCS", "", ["data/remus.txt","data/dooley.txt","Gut_hand_attr/128.csv", "Gut_hand_attr/85.csv", "Gut_hand_attr/213.csv", "Gut_hand_attr/129.csv"]),
    ("DATA_DIR","","data/"),
    ("MAX_N","", 4),
    ("GEN_MODELS","", ["mistralai/Mistral-7B-Instruct-v0.2"]),
    ("STORYPROMPTS_LOC","","data/writingPrompts.tar.gz"),
    ("BASE_PROMPT","","'Write a few sentences based on the following story prompt: '"),
    ("N_PROMPT_SETS", "", 1),
    ("EVAL_MODEL","", ["openai-community/gpt2-large"]), #, "mistralai/Mistral-7B-v0.1"]),
    ("EVAL_TOKENIZERS","", ["openai-community/gpt2-large", "mistralai/Mistral-7B-v0.1", "spacy"]),
    ("NG_SCALINGS", "", ["reduced_scalings.json"]),
    ("GEN_MODEL_PARAMS", "", [{"do_sample": 1, "top_k":0, "temperature": 0.0}, {"do_sample": 1, "top_k": 40, "temperature":0.0}, {"do_sample": 1, "top_k":0, "temperature":0.5}]),
    ("RANDOM_STATE", "", 20),
    ("TRAIN_SIZE", "", 0.95),
    ("USE_GRID","",0),
    ("GRID_TYPE","", "slurm"),
    #("GRID_QUEUE","", "a100"),
    #("GRID_ACCOUNT","", "tlippi1_gpu"),
    ("GRID_GPU_COUNT","", 0),
    ("GRID_MEMORY", "", "64G"),
    ("GRID_LABEL", "", "NGStyle"),
    ("GRID_TIME", "", "48:00:00"),

    #("FOLDS", "", 1),
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[steamroller.generate],
    
    BUILDERS={
        "CreateData": Builder(
            action="python scripts/create_json_text.py --text_input ${SOURCES[0]} --out ${TARGETS[0]} --char_name ${CHAR_NAME}"
        ),
        "TrainNgram": Builder(action="python scripts/train_ngram.py --text ${SOURCES[0]} --tokenizer ${MODEL} --output ${TARGETS[0]} --max_n ${MAX_N}"),
        "LoadPrompts": Builder(action="python scripts/generate_prompts.py --prompts_input ${STORYPROMPTS_LOC} --prompt_out ${TARGETS[0]} --target_out ${TARGETS[1]} --base_prompt ${BASE_PROMPT}"),
        "GenerateControlPrompts": Builder(action="python scripts/control_prompts.py --input ${SOURCES[0]} --author_stems ${SOURCES[1]} --output ${TARGETS[0]}"),
        "RunPrompt": Builder(action="python scripts/run_prompt.py --model ${MODEL} --text_input ${SOURCES[0]} --prompts ${SOURCES[1]} --scalings ${SOURCES[2]} --out ${TARGETS[0]} --do_sample ${DO_SAMPLE} --random_state ${RANDOM_STATE} --n_prompt_sets ${N_PROMPT_SETS}"),
        "RunPromptControl": Builder(action="python scripts/run_prompt_baseline.py --model ${MODEL} --prompts ${SOURCES[0]} --out ${TARGETS[0]} --do_sample ${DO_SAMPLE} --n_prompt_sets ${N_PROMPT_SETS} --random_state ${RANDOM_STATE}"),
        "RunPromptNLTK": Builder(action="python scripts/run_prompt_nltk.py --model ${MODEL} --ngram ${SOURCES[0]} --prompts ${SOURCES[1]} --scalings ${SOURCES[2]} --out ${TARGETS[0]} --do_sample ${DO_SAMPLE} --n_prompt_sets ${N_PROMPT_SETS} --random_state ${RANDOM_STATE}"), 
        "SplitForOpenEnded": Builder(action="python scripts/shuffle_data.py --input ${SOURCES[0]} --outputs ${TARGETS} --train_size ${TRAIN_SIZE} --random_state ${RANDOM_STATE}"),
        "RunOpenEnded": Builder(action="python scripts/run_open_ended.py --model ${MODEL} --train ${SOURCES[0]} --test ${SOURCES[1]} --scalings ${SOURCES[2]} --out ${TARGETS[0]} --do_sample ${DO_SAMPLE} --top_k ${TOP_K} --temperature ${TEMP} --random_state ${RANDOM_STATE}"),
        "EvalPerplexity": Builder(action="python scripts/evaluate_perplexity.py --model ${EVAL_MODEL} --input ${SOURCES[0]} --output ${TARGETS[0]} --mode ${MODE}"),
        "AutomaticEval": Builder(action="python scripts/automatic_measures.py --input ${SOURCES[0]} --output ${TARGETS[0]} --mode ${MODE} --tokenizer ${EVAL_TOKENIZERS}"),
        "ProduceTable": Builder(action="python scripts/produce_table.py --inputs ${SOURCES} --output ${TARGETS[0]} --drop ${DROP}")
    }
)

res = []
text_ins = []
text_res = []
ngram_models = []
text_perp = []
control_prompts = []

story_prompts, prompt_targets = env.LoadPrompts(["work/prompts.jsonl", "work/targets.jsonl"], [])
#target_perps = env.EvalPerplexity(["work/target_perplexities.jsonl"], [prompt_targets], MODE="text")
#text_res.append(env.AutomaticEval(["work/target_automatic_eval.jsonl"], [target_perps], MODE="text"))

for text, textloc in zip(env["TEXTS"],env["TEXT_LOCS"]):
    text_ins.append(env.CreateData(["work/${TEXT}/text.jsonl"], [textloc], TEXT=text, CHAR_NAME=text))
    control_prompts.append(env.GenerateControlPrompts(["work/${TEXT}/control_prompts.jsonl"], [story_prompts, env["DATA_DIR"]+text+"_prompts.txt"], TEXT=text))
    #text_perp.append( env.EvalPerplexity(["work/${TEXT}/perplexities.jsonl"], [text_ins[-1]], TEXT=text, MODE="text"))
    #text_res.append( env.AutomaticEval(["work/${TEXT}/automatic_eval.jsonl"], [text_ins[-1]], TEXT=text, MODE="text"))
    ngram_models.append(env.TrainNgram(["work/${TEXT}/ngram.pkl"],[text_ins[-1]], MODEL=env["GEN_MODELS"][0], TEXT=text))

for model_type in env["GEN_MODELS"]:
    for text_name,text_in, ngm, control_prompt in zip(env["TEXTS"], text_ins, ngram_models, control_prompts):
        print(text_name)
        print(control_prompt)
        con = env.RunPromptControl(["work/${MODEL}/${TEXT_NAME}/control.jsonl"], [control_prompt],
            MODEL = model_type,
            DO_SAMPLE = 1,
            TEXT_NAME = text_name)
        for scaling in env["NG_SCALINGS"]:
            #gen = env.RunPrompt(["work/${MODEL}/${TEXT_NAME}/results.jsonl"],
            #    [text_in,story_prompts,env["DATA_DIR"]+scaling],
            #    MODEL = model_type,
            #    DO_SAMPLE = 1,
            #    TEXT_NAME = text_name)
            gen = env.RunPromptNLTK(["work/${MODEL}/${TEXT_NAME}/results.jsonl"],
                [ngm, story_prompts, env["DATA_DIR"]+scaling],
                MODEL = model_type,
                NGRAM = ngm,
                DO_SAMPLE = 1,
                TEXT_NAME = text_name)
            #res.append(env.AutomaticEval(["work/${MODEL}/${TEXT}/gen_automatic_eval.jsonl"], [gen],
                #MODE = "instruct",
                #TEXT = text_name))

"""
text_table = env.ProduceTable(["work/results/text_table.tex"], text_res)

for model_type in env["GEN_MODELS"]:
    for text_name, text in zip(env["TEXTS"], text_ins):
        for scaling in env["NG_SCALINGS"]:
            for pn, params in enumerate(env["GEN_MODEL_PARAMS"]):
                gen = env.RunPromptNLTK(["work/${MODEL}/${TEXT_NAME}/${PARAM_NUM}/results.jsonl"],
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
