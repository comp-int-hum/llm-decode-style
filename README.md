# Transferring Extreme Subword Style Using Ngram Model-Based Logit Scaling

This is the official repository for the paper *Transferring Extreme Subword Style Using Ngram Model-Based Logit Scaling* presented at [NLP4DH 2025 @ NAACL](https://www.nlp4dh.com/nlp4dh-2025). We transfer elements of "nonstandard" English subword style of the type commonly used in literary works by interpolating information derived from author or character-specific ngram models with the logit-level predictions of a more powerful LLM. This approach produces fluent, topical, and author/character style-appropriate text.

In short, it takes a prompt like this:

>Write a few sentences based on the following story prompt:  A person with a high school education gets sent back into the 1600s and tries to explain science and technology to the people.

And produces generations like this (Target style: Charles Chesnutt's character Julius McAdoo):

>In de mawnin' light of the quaint 1600s hamlet, John, a humbleedge Mars' Hill, Maine school graduate, stood astounded ashes amazed crowd gazed up en masse at the metallic apparition he held aloft- a pocketwatch. "This," John began, his voice wa'n't above a whisper, "is a means of keepin' track o' time, a essential component in managin' a more roots in de ground agricultural process." The crowd's quizzical expressions only grew denser, nary a soul comprehendin' the essence of John's mirac' er gadget. Perceivin' their struggle, he continued, "The world around us, it isn't just a collection o' rocks, water, an' earth. There's somethin' called science, a systematic method o' learnin' 'bout the natural workings o' the universe."

Without additional computation or the fragility of prompting. 

 The paper preprint can be [found on ArXiv](https://arxiv.org/pdf/2503.08550)

## Installation

After cloning the repository, install the dependencies:

```bash
pip install -r requirements.txt
```

The models used for the paper experiments include GPT-2, Llama3, and Mistral V0.2. These are sourced through the HuggingFace hub.

## Running the code

The experiments are organized using the Steamroller extensions of the SCONS build system. The training and evaluation are split into two scons files:

    1. SConstruct: Generates the experiment data, trains the ngram scaling models, generates control and experiment texts.
    2. SConscriptEval: Produces the perplexity-based evaluation and scaling-selection images discussed in the paper. 

These two should be run in order. This looks like:

```bash
scons -f SConstruct -Q
scons -f SConscriptEval -Q
```
The addition of the -n switch will perform a scons "dry run"
Currently, the table stitching scripts are outside of this pipeline -- they are straightforward jsonl > pandas df > latex converters. 
