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
    ("TEXTS","", ["remus","dooley","Biglow","Todd","Remus","Julius","Huck"]),
    #("CSV_TEXT_CHARS", "", ["Biglow","Todd","Remus", "Julius"]),
    ("TEXT_LOCS", "", ["data/remus.txt","data/dooley.txt","Gut_hand_attr/128.csv", "Gut_hand_attr/85.csv", "Gut_hand_attr/213.csv", "Gut_hand_attr/129.csv", "Gut_hand_attr/166.csv"]),
    ("DATA_DIR","","data/"),
    ("MAX_N","", 4),
    ("GEN_MODELS","", ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-3.2-3B-Instruct"]),
    ("STORYPROMPTS_LOC","","data/writingPrompts.tar.gz"),
    ("BASE_PROMPT","","'Write a few sentences based on the following story prompt: '"),
    ("N_PROMPT_SETS", "", 50),
    ("EVAL_MODEL","", ["openai-community/gpt2-large"]),
    ("NG_SCALINGS", "", ["ng_scalings.json"]),
    ("CONTROL_PARAMS", "", [{"top_k":0, "temperature": 1.0, "sample": 1}, {"top_k":0, "temperature": 1.0, "sample": 0}]), #{"top_k":0, "temperature":0.7}]),
    ("PARAM_NAMES", "", ["sample","greedy"]),
    ("RANDOM_STATE", "", 20),
    ("LT_COLORS", "", ["green", "blue", "orange", "pink", "violet"]),
    ("CHUNK_LEN", "", 32),
    ("TRAIN_SIZE", "", 0.95),
    ("USE_GRID","", 0),
    ("GRID_TYPE","", "slurm"),
    #("GRID_QUEUE","", "cpu"),
    #("GRID_ACCOUNT","", "tlippi1_gpu"),
    ("GRID_GPU_COUNT","", 0),
    ("GRID_MEMORY", "", "64G"),
    ("GRID_LABEL", "", "NGStyle"),
    ("GRID_TIME", "", "48:00:00"),
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
        "RunPromptControl": Builder(action="python scripts/run_prompt_baseline.py --model ${MODEL} --prompts ${SOURCES[0]} --out ${TARGETS[0]} --do_sample ${DO_SAMPLE} --n_prompt_sets ${N_PROMPT_SETS} --random_state ${RANDOM_STATE} --top_k ${TOP_K} --temp ${TEMP}"),
        "RunPromptNLTK": Builder(action="python scripts/run_prompt_nltk.py --model ${MODEL} --ngram ${SOURCES[0]} --prompts ${SOURCES[1]} --scalings ${SOURCES[2]} --out ${TARGETS[0]} --do_sample ${DO_SAMPLE} --n_prompt_sets ${N_PROMPT_SETS} --random_state ${RANDOM_STATE} --temp ${TEMP}"), 
        "SplitForOpenEnded": Builder(action="python scripts/shuffle_data.py --input ${SOURCES[0]} --outputs ${TARGETS} --train_size ${TRAIN_SIZE} --random_state ${RANDOM_STATE}"),
        "RunOpenEnded": Builder(action="python scripts/run_open_ended.py --model ${MODEL} --train ${SOURCES[0]} --test ${SOURCES[1]} --scalings ${SOURCES[2]} --out ${TARGETS[0]} --do_sample ${DO_SAMPLE} --top_k ${TOP_K} --temperature ${TEMP} --random_state ${RANDOM_STATE}"),
        "EvalPerplexity": Builder(action="python scripts/evaluate_perplexity.py --model ${EVAL_MODEL} --input ${SOURCES[0]} --output ${TARGETS} --mode ${MODE} --chunk_len ${CHUNK_LEN} --device cuda"),
        "AutomaticEval": Builder(action="python scripts/automatic_measures.py --input ${SOURCES[0]} --output ${TARGETS[0]} --mode ${MODE} --tokenizer ${TOKENIZER}"),
        "ProduceHighlighted": Builder(action="python scripts/highlight.py --input ${SOURCES[0]} --output ${TARGETS[0]} --colors ${LT_COLORS}"),
        "ProduceTable": Builder(action="python scripts/produce_table.py --p_in ${SOURCES} --a_in ${A_IN} --output ${TARGETS[0]} --drop ${DROP}")
    }
)


text_ins = []
text_res = []
ngram_models = []
text_perp = []
control_prompts = []


story_prompts, prompt_targets = env.LoadPrompts(["work/prompts.jsonl", "work/targets.jsonl"], [])
target_perps = env.EvalPerplexity(["work/target_perplexities.jsonl"], [prompt_targets], MODE="targets")
target_res = env.AutomaticEval(["work/target_automatic_eval.jsonl"], [prompt_targets], MODE="targets")

target_table = env.ProduceTable(["work/results/target_table.tex"], [target_perps], A_IN=[target_res], DROP=["name","p_cl","D_full"])


for text, textloc in zip(env["TEXTS"],env["TEXT_LOCS"]):
    text_ins.append(env.CreateData(["work/${TEXT}/text.jsonl"], [textloc], TEXT=text, CHAR_NAME=text))
    control_prompts.append(env.GenerateControlPrompts(["work/${TEXT}/control_prompts.jsonl"], [story_prompts, env["DATA_DIR"]+text+"_prompts.txt"], TEXT=text))
    text_perp.append( env.EvalPerplexity(["work/${TEXT}/perplexities.jsonl"], [text_ins[-1]], TEXT=text, MODE="text"))
    text_res.append( env.AutomaticEval(["work/${TEXT}/automatic_eval.jsonl"], [text_ins[-1]], TEXT=text, MODE="text", TOKENIZER=env["EVAL_MODEL"]))

text_table = env.ProduceTable(["work/results/text_table.tex"], text_perp, A_IN=text_res, DROP=["name","p_cl","D_full"])

for model_type in env["GEN_MODELS"]:
    for text_name,text_in, control_prompt in zip(env["TEXTS"], text_ins, control_prompts):
        ngm = env.TrainNgram(["work/${MODEL}/${TEXT_NAME}/ngram.pkl"], text_in, MODEL=model_type, TEXT_NAME=text_name)
        for control_param, param_name in zip(env["CONTROL_PARAMS"], env["PARAM_NAMES"]):
            gen_perp = []
            res = []
            controls = []
            control_perps = []
            control_res = []
            controls.append( env.RunPromptControl(["work/${MODEL}/${TEXT_NAME}/${PARAM_NAME}/control.jsonl"], [control_prompt],
                MODEL = model_type,
                DO_SAMPLE = control_param["sample"],
                TEXT_NAME = text_name,
                TOP_K = control_param["top_k"],
                TEMP = control_param["temperature"],
                PARAM_NAME = param_name))
            control_perps.append(env.EvalPerplexity(["work/${MODEL}/${TEXT_NAME}/${PARAM_NAME}/control_perplexity.jsonl"], [controls[-1]], TEXT_NAME=text_name, MODEL=model_type, MODE="control", PARAM_NAME=param_name))
            control_res.append( env.AutomaticEval(["work/${MODEL}/${TEXT_NAME}/${PARAM_NAME}/automatic_eval.jsonl"], [controls[-1]], TEXT_NAME=text_name, MODEL=model_type, PARAM_NAME = param_name, MODE="control", TOKENIZER=env["EVAL_MODEL"]))
            for scaling in env["NG_SCALINGS"]:
                gen = env.RunPromptNLTK(["work/${MODEL}/${TEXT_NAME}/${PARAM_NAME}/results.jsonl"],
                    [ngm, story_prompts, env["DATA_DIR"]+scaling],
                    MODEL = model_type,
                    NGRAM = ngm,
                    DO_SAMPLE = control_param["sample"],
                    TEMP = control_param["temperature"],
                    PARAM_NAME = param_name,
                    TEXT_NAME = text_name)
                highlighted = env.ProduceHighlighted(["work/${MODEL}/${TEXT_NAME}/${PARAM_NAME}/highlighted.tex"], [gen], TEXT_NAME=text_name, MODEL=model_type, PARAM_NAME=param_name)
                gen_perp.append(env.EvalPerplexity(["work/${MODEL}/${TEXT_NAME}/${PARAM_NAME}/perplexities.jsonl"], [gen], TEXT_NAME=text_name, MODEL=model_type, PARAM_NAME=param_name,  MODE="prompt"))
                res.append(env.AutomaticEval(["work/${MODEL}/${TEXT}/${PARAM_NAME}/gen_automatic_eval.jsonl"], [gen],
                    MODEL = model_type,
		    TOKENIZER = env["EVAL_MODEL"],
                    MODE = "prompt",
                    PARAM_NAME = param_name,
                    TEXT = text_name))
            gen_table = env.ProduceTable(["work/results/${MODEL}/${TEXT_NAME}/${PARAM_NAME}/gen_table.tex"], gen_perp, A_IN=res, DROP=["name", "p_cl", "D_full"], MODEL=model_type, TEXT_NAME=text_name, PARAM_NAME=param_name)
            text_table = env.ProduceTable(["work/results/${MODEL}/${TEXT_NAME}/${PARAM_NAME}/control_table.tex"], control_perps, A_IN=control_res, DROP=["name","p_cl","D_full"], MODEL=model_type, TEXT_NAME=text_name, PARAM_NAME=param_name)


