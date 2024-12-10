
# Knowledged Optimized Augmentation for Long-context Access (KOALA)

6.5940 Final Project

Kavya Anbarasu, Gilford Ting, Sarah Wang, Jessica Xu, and Joyce Yuan

[[(need to update link)paper](http://arxiv.org/abs/2309.17453)] [[(change edit permissions when done)poster](https://docs.google.com/presentation/d/1-d03qa8PTlB7mmCnO_tW6KTt-8NnXcJQLrXIPtaBdQE/edit?usp=sharing)][[(need to update link) video](https://youtu.be/hvJsEzP34o8)]

 
## TL;DR

By integrating StreamingLLM with Retrieval-Augmented Generation (RAG), we can dynamically retrieve and use relevant context that would otherwise have been evicted from the cache to allow for infinite-length inputs without sacrificing performance.

  

## Abstract

Engaging with Large Language Models (LLMs) in streaming applications with long interactions, such as multi-round dialogue, is limited by finite attention windows and losing access to past tokens. StreamingLLM introduces an efficient framework that enables LLMs trained with a finite length attention window to generalize to infinite sequence lengths. However, it lacks the ability to retrieve evicted tokens and loses previous context. To overcome this limitation, we propose deploying StreamingLLM with Retrieval-Augmented Generation (RAG) to create Knowledge Optimized Augmentation for Long-context Access (KOALA). RAG employs vectorized storage for retrieving previous or external information. Using the LlamaIndex framework for integrating LLMs with external data, we are able to dynamically reintroduce relevant evicted tokens back into the StreamingLLM cache to simulate "infinite memory". This approach enhances the capabilities of LLMs in real-time tasks, such as long-context conversations, for more robust and context-aware applications. KOALA demonstrates improved results for Needle in Haystack evaluation as well as decreased perplexity compared to StreamingLLM.

## Usage

  

### Environment Setup

  

```bash

conda create -yn streaming python=3.8

conda activate streaming

  

pip install torch torchvision torchaudio

pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

pip install llama-index

  

python setup.py develop

```

### OpenAI Key
An OpenAI Key is needed for LlamaIndex. It can be set with

```bash

os.environ["OPENAI_API_KEY"] =  "{key}"  

```

### Run Demo Chatbot

  

```bash

python examples/koala_demo.py

```


### Run Needle in Haystack Evaluation

  

```bash

python examples/eval_haystack.py

```
  

### Run Perplexity Evaluation

  

```bash

python examples/final_newest_llama_eval_long.py --num_eval_tokens 1000

```

Note: You can preface each Python script with `CUDA_VISIBLE_DEVICES=0` to specify a desired gpu to suit your purposes
  

## Acknowledgements

  Thank you to the 6.5940 staff for all your support and a great semester!

## Citation

  

Our project was based off the following paper for StreamingLLM

  

```bibtex

@article{xiao2023streamingllm,
        title={Efficient Streaming Language Models with Attention Sinks},
        author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
        journal={arXiv},
        year={2023}
        }

```


