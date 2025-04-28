# Transferring Extreme Subword Style Using Ngram Model-Based Logit Scaling

This is the official repository for the paper *Transferring Extreme Subword Style Using Ngram Model-Based Logit Scaling* presented at [NLP4DH 2025 @ NAACL](https://www.nlp4dh.com/nlp4dh-2025). We transfer elements of "nonstandard" English subword style of the type commonly used in literary work by interpolating information derived from author or character-spcific ngram models with the logit-level predictions of a more powerful LM. The paper preprint can be [found on ArXiv](https://arxiv.org/pdf/2503.08550)

## Installation

After cloning the repository, install the dependencies:

```bash
pip install -r requirements.txt
```

The models used for the paper experiments include GPT-2, Llama3, and Mistral V0.2. These are sourced through the HuggingFace hub.

## Running the code

The experiments are organized using the Steamroller extensions of the SCONS build system. The training and evaluation are split into two scons files:

    1. **SConstruct:** Generates the experiment data, trains the ngram scaling models, generates control and experiment texts.
    2. **SConscriptEval:** Produces the perplexity-based evaluation and scaling-selection images discussed in the paper. 

These two should be run in order. This looks like:

```bash
scons -f SConstruct -Q
scons -f SConscriptEval
```
Currently, the table stitching scripts are outside of this pipeline -- they are straightforward jsonl > pandas df > latex converters. 
