import torch
import re
import string
import json
from tqdm import tqdm
from datetime import datetime
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from examples.run_streaming_llama_index import enable_streaming_llm_llama_index
from llama_index.core import VectorStoreIndex

current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
log_filename = f"examples/logging/{current_time}_haystack.log"
logging.basicConfig(
    filename=log_filename,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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

    for _ in range(max_gen_len - 1):
        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        if pred_token_idx == tokenizer.eos_token_id:
            break

    generated_response = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    return generated_response, past_key_values

# @torch.no_grad()
# def run_inference_single_prompt_non_rag(model, tokenizer, prompt, kv_cache=None, max_gen_len=100):
#     full_prompt = "USER: " + prompt + "\n\nASSISTANT: "
#     input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
#     if kv_cache is not None:
#         seq_len = input_ids.shape[1]
#         past_key_values = kv_cache.evict_for_space(None, seq_len + max_gen_len)
#     else:
#         past_key_values = None

#     response, past_key_values = non_rag_greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len)
#     return response

@torch.no_grad()
def run_inference_single_prompt_non_rag(model, tokenizer, prompt, kv_cache=None, max_gen_len=100, context_docs=None):
    """
    Run inference for a single prompt using the non-RAG setup and return the generated answer.
    Now we include all documents directly in the prompt for a fairer comparison.
    """
    if context_docs is not None and len(context_docs) > 0:
        # Combine all 10 documents into a single string
        docs_str_list = []
        for doc_entry in context_docs:
            sentences = doc_entry[1]  # doc_entry = [title, [sentences]]
            docs_str_list.append(" ".join(sentences))
        combined_docs_str = "\n\n".join(docs_str_list)

        full_prompt = f"USER: {prompt}\n\nHere are some documents:\n{combined_docs_str}\n\nASSISTANT: "
    else:
        full_prompt = "USER: " + prompt + "\n\nASSISTANT: "

    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
    if kv_cache is not None:
        seq_len = input_ids.shape[1]
        space_needed = seq_len + max_gen_len
        kv_cache.evict_for_space(None, space_needed)
    response, _ = non_rag_greedy_generate(model, tokenizer, input_ids, None, max_gen_len=max_gen_len)
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
    Run inference with RAG system. Retrieve relevant context from kv_cache and prepend it.
    """
    full_prompt = "USER: " + prompt + " \nASSISTANT: "
    if kv_cache is not None:
        past_context_list = kv_cache.retrieve_relevant_context(prompt)
        past_context_string = " ".join(past_context_list).strip()

        if past_context_string:
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
    return response, past_context_list


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
            precision = num_common / len(pred_tokens) if pred_tokens else 0
            recall = num_common / len(gold_tokens) if gold_tokens else 0
            curr_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        if curr_f1 > best_f1:
            best_f1 = curr_f1
    return best_f1

def extract_gold_answers(record):
    return [record["answer"]]

def insert_context_docs_into_rag(kv_cache, context):
    """
    Insert the 10 documents from hotpot context into the vector store.
    Each element of `context` is: [title, [list_of_sentences]]
    We'll combine the sentences into a single document string.
    We'll clear the vector store each time to ensure only current docs are present.
    """
    # Reset the vector index for each example if needed
    kv_cache.vector_index = VectorStoreIndex([])

    for doc_entry in context:
        title = doc_entry[0]
        sentences = doc_entry[1]
        doc_text = " ".join(sentences)
        kv_cache.store_text(doc_text)


def main():
    filepath = "data/hotpot_dev_distractor_v1.json"
    print(f'Loading HotpotQA dataset from {filepath}')
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    non_rag_model, non_rag_tokenizer, non_rag_kv = load_non_rag_model(enable_streaming=True)
    rag_model, rag_tokenizer, rag_kv = load_rag_model(enable_streaming=True)

    rag_correct_count = 0
    non_rag_correct_count = 0
    rag_f1_sum = 0.0
    non_rag_f1_sum = 0.0

    data = data[:1000]  # comment this out when we want to do full eval
    total = len(data)
    print(f"Number of examples: {total}")

    idx = 0
    for record in tqdm(data, desc="Evaluating"):
        question_text = record["question"]
        gold_answers = extract_gold_answers(record)
        context = record["context"]  # list of docs

        logging.info(f"[{idx}] Question: {question_text}")
        logging.info(f"[{idx}] Gold Answers: {gold_answers}")

        # Non-RAG Inference (put all docs into prompt as context)
        non_rag_answer = run_inference_single_prompt_non_rag(
            non_rag_model, 
            non_rag_tokenizer, 
            question_text, 
            kv_cache=non_rag_kv, 
            max_gen_len=100, 
            context_docs=context
        )        
        logging.info(f"[{idx}] Non-RAG Answer: {non_rag_answer}")

        # RAG Inference (insert this example's context docs into the vector store for retrieval)
        insert_context_docs_into_rag(rag_kv, context)
        rag_answer, past_context = run_inference_single_prompt_rag(rag_model, rag_tokenizer, question_text, kv_cache=rag_kv, max_gen_len=100)
        logging.info(f"[{idx}] Retrieved {len(past_context)} past contexts for RAG: {past_context}")
        logging.info(f"[{idx}] RAG Answer: {rag_answer}")

        if exact_match_score(non_rag_answer, gold_answers):
            non_rag_correct_count += 1
        if exact_match_score(rag_answer, gold_answers):
            rag_correct_count += 1

        non_rag_f1 = f1_score(non_rag_answer, gold_answers)
        rag_f1 = f1_score(rag_answer, gold_answers)
        non_rag_f1_sum += non_rag_f1
        rag_f1_sum += rag_f1

        idx += 1

    logging.info(f"Non-RAG Exact Match Accuracy: {non_rag_correct_count / total:.2f}")
    logging.info(f"RAG Exact Match Accuracy: {rag_correct_count / total:.2f}")
    logging.info(f"Non-RAG F1: {non_rag_f1_sum / total:.2f}")
    logging.info(f"RAG F1: {rag_f1_sum / total:.2f}")


    # Print aggregated results
    print(f"Non-RAG Exact Match Accuracy: {non_rag_correct_count / total:.2f}")
    print(f"RAG Exact Match Accuracy: {rag_correct_count / total:.2f}")
    print(f"Non-RAG F1: {non_rag_f1_sum / total:.2f}")
    print(f"RAG F1: {rag_f1_sum / total:.2f}")

if __name__ == "__main__":
    main()
