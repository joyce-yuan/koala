import csv
import os
import torch
import re
import string

from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from streaming_llm.llama_index_verbose import LlamaIndexKVCache
from streaming_llm.utils import load_jsonl
from examples.run_streaming_llama_index import enable_streaming_llm_llama_index

from datetime import datetime

import logging

current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
log_filename = f"logging/{current_time}_haystack.log"
# Configure the logging system
logging.basicConfig(
    filename=log_filename,         # Log file name
    filemode='a',                  # Append mode
    level=logging.INFO,            # Minimum level of logs to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s %(levelname)s: %(message)s', # Format of each log line
    datefmt='%Y-%m-%d %H:%M:%S'    # Date format
)


# Helper functions for the NON-RAG model (i.e run_streaming_llama.py)
def load_non_rag_model(model_name_or_path="Jiayi-Pan/Tiny-Vicuna-1B", enable_streaming=False, start_size=4, recent_size=1000):
    model, tokenizer = load_model_tokenizer(model_name_or_path)
    kv_cache = None
    if enable_streaming:
        kv_cache = enable_streaming_llm(model, start_size=start_size, recent_size=recent_size)
    return model, tokenizer, kv_cache


def load_model_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    model.eval()
    return model, tokenizer


@torch.no_grad()
def non_rag_greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len=100):
    """
    Generate a response without RAG, returning a final string.
    """
    outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    generated_response = ""

    for _ in range(max_gen_len - 1):
        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if pred_token_idx == tokenizer.eos_token_id:
            break

    generated_response = generated_text.strip()
    return generated_response, past_key_values


@torch.no_grad()
def run_inference_single_prompt_non_rag(model, tokenizer, prompt, kv_cache=None, max_gen_len=100):
    """
    Run inference for a single prompt using the non-RAG setup and return the generated answer.
    """
    full_prompt = "USER: " + prompt + "\n\nASSISTANT: "
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
    if kv_cache is not None:
        seq_len = input_ids.shape[1]
        past_key_values = kv_cache.evict_for_space(None, seq_len + max_gen_len)
    else:
        past_key_values = None

    response, past_key_values = non_rag_greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len)
    return response


# Helper functions for the RAG model (i.e run_streaming_llama_index.py)
def load_rag_model(model_name_or_path="Jiayi-Pan/Tiny-Vicuna-1B", enable_streaming=True, start_size=4, recent_size=1000):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    model.eval()

    kv_cache = None
    if enable_streaming:
        kv_cache = enable_streaming_llm_llama_index(model, start_size, recent_size)

    return model, tokenizer, kv_cache


@torch.no_grad()
def rag_greedy_generate(model, tokenizer, input_ids, past_key_values, kv_cache=None, max_gen_len=100):
    outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]

    for _ in range(max_gen_len - 1):
        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        if pred_token_idx == tokenizer.eos_token_id:
            break

    generated_response = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    if kv_cache is not None:
        kv_cache.store_text(generated_response)
    return generated_response, past_key_values


@torch.no_grad()
def run_inference_single_prompt_rag(model, tokenizer, prompt, kv_cache=None, max_gen_len=100):
    """
    Run inference with RAG system. Retrieves previously evicted tokens if any and add them as context.
    """
    # Retrieve relevant context from kv_cache if available, prepend that context to the prompt and generate.
    # TODO: do above for subsequent prompts. For the first prompt, no retrieval.

    full_prompt = "USER: " + prompt + " \nASSISTANT: "
    if kv_cache is not None:
        past_context_list = kv_cache.retrieve_relevant_context(prompt)
        past_context_string = " ".join(past_context_list)

        if past_context_string.strip():
            input_with_context = past_context_string + " " + full_prompt
        else:
            input_with_context = full_prompt

        input_ids = tokenizer(input_with_context, return_tensors="pt").input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        kv_cache.store_text(prompt)
        past_key_values = kv_cache.evict_for_space(None, seq_len + max_gen_len, past_context=past_context_string)
    else:
        input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
        past_key_values = None

    response, past_key_values = rag_greedy_generate(model, tokenizer, input_ids, past_key_values, kv_cache=kv_cache, max_gen_len=max_gen_len)
    return response


def normalize_text(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match_score(pred, gold_answers):
    """
    Check if any of the gold answers is a substring of the prediction.
    """
    if not gold_answers:
        # If no gold answers, consider it incorrect
        return False
    pred_lower = pred.lower()
    for ans in gold_answers:
        if ans.lower() in pred_lower:
            return True
    return False


def f1_score(pred, gold_answers):
    """
    Compute token-level F1 by choosing the gold answer with the best F1.
    """
    if not gold_answers:
        return 0.0

    pred_tokens = normalize_text(pred).split()

    best_f1 = 0.0
    for ans in gold_answers:
        gold_tokens = normalize_text(ans).split()

        common = set(pred_tokens) & set(gold_tokens)
        num_common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            # If either is empty, define F1 as 0 unless both are empty
            curr_f1 = 1.0 if len(pred_tokens) == len(gold_tokens) else 0.0
        else:
            precision = num_common / len(pred_tokens)
            recall = num_common / len(gold_tokens)
            if precision + recall > 0:
                curr_f1 = 2 * (precision * recall) / (precision + recall)
            else:
                curr_f1 = 0.0

        if curr_f1 > best_f1:
            best_f1 = curr_f1
    return best_f1


def extract_gold_answers(sample):
    """
    Extract all gold short answers from the NQ sample.
    """
    gold_answers = []
    annotations = sample["annotations"]
    for annot in annotations["short_answers"]:
        for text_ans in annot["text"]:
            if text_ans.strip():
                gold_answers.append(text_ans)
    return gold_answers


def main():
    nq = load_dataset("natural_questions", "dev")
    subset = nq["validation"].select(range(50))

    non_rag_model, non_rag_tokenizer, non_rag_kv = load_non_rag_model(enable_streaming=True)
    rag_model, rag_tokenizer, rag_kv = load_rag_model(enable_streaming=True)

    rag_correct_count = 0
    non_rag_correct_count = 0
    rag_f1_sum = 0.0
    non_rag_f1_sum = 0.0
    total = len(subset)

    idx = 0
    for example in tqdm(subset):
        # import pdb
        # pdb.set_trace()
        question_text = example["question"]["text"]
        gold_answers = extract_gold_answers(example)
        print(f"Question Text: {question_text}")
        non_rag_answer = run_inference_single_prompt_non_rag(non_rag_model, non_rag_tokenizer, question_text, kv_cache=non_rag_kv, max_gen_len=100)
        
        rag_answer = run_inference_single_prompt_rag(rag_model, rag_tokenizer, question_text, kv_cache=rag_kv, max_gen_len=100)

        if exact_match_score(non_rag_answer, gold_answers):
            non_rag_correct_count += 1
        if exact_match_score(rag_answer, gold_answers):
            rag_correct_count += 1

        logging.info(f"[{idx}] Question Text: {question_text}")
        logging.info(f"[{idx}] Gold Answers: {gold_answers}")
        logging.info(f"[{idx}] Non-RAG Answer: {non_rag_answer}")
        logging.info(f"[{idx}] RAG Answer: {rag_answer}")
        
        # F1 Score
        non_rag_f1 = f1_score(non_rag_answer, gold_answers)
        rag_f1 = f1_score(rag_answer, gold_answers)
        non_rag_f1_sum += non_rag_f1
        rag_f1_sum += rag_f1

        idx += 1

    print(f"Non-RAG Exact Match Accuracy: {non_rag_correct_count / total:.2f}")
    print(f"RAG Exact Match Accuracy: {rag_correct_count / total:.2f}")
    print(f"Non-RAG F1: {non_rag_f1_sum / total:.2f}")
    print(f"RAG F1: {rag_f1_sum / total:.2f}")


if __name__ == "__main__":
    main()
