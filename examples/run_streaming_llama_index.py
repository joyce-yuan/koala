import warnings

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from streaming_llm.llama_index_newest import LlamaIndexKVCache
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, kv_cache=None):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    generated_response = ""
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        # TODO: what does this section do?
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            generated_response += " ".join(generated_text[pos:now]) + " "
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    generated_response += " ".join(generated_text[pos:])

    # After the loop ends, check if the sequence was truncated
    # if len(generated_ids) >= max_gen_len and generated_ids[-1] != tokenizer.eos_token_id:
        # print("\n[WARNING] Sequence truncated due to length limit.")

    if kv_cache is not None:
        # print("[STORING RESPONSE]: ", generated_response)
        kv_cache.store_text(generated_response)
    
    return past_key_values


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=50):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        # print("[DEBUG]: prompt: ", idx)
        original_prompt = prompt
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        # print("[PROMPT]: " + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        input_with_context = None

        if kv_cache is not None:
            # Get past key values and evict for space
            if idx == 0:
                seq_len = input_ids.shape[1]
                space_needed = seq_len + max_gen_len
                past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)
            else: 
                # Query past context with the current prompt
                past_context = kv_cache.retrieve_relevant_context(prompt)
                past_context_string = " ".join(past_context)

                # Concat the past context with the current prompt
                input_with_context = "PAST CONTEXT: " + past_context_string + "\n " + "CURRENT PROMPT: " + prompt

                input_ids = tokenizer(input_with_context, return_tensors="pt").input_ids
                input_ids = input_ids.to(model.device)

                seq_len = input_ids.shape[1]

                # Get past key values with past context included in cache
                space_needed = seq_len + max_gen_len
                past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

            # print('[STORING PROMPT]: ', prompt + "\n\n")
            kv_cache.store_text(original_prompt) # store prompt

        # if input_with_context is not None:
        #     print("[INPUT WITH CONTEXT]:\n", input_with_context + "\n\n")

        # print("[GREEDY START]")
        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, kv_cache=kv_cache
        )
        # print("[GREEDY DONE]\n\n")


def enable_streaming_llm_llama_index(model, start_size, recent_size):
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = LlamaIndexKVCache(
        start_size=start_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache

def main(args):
    model_name_or_path = args.model_name_or_path
    # model, tokenizer = load(model_name_or_path)
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
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    # test_filepath = os.path.join("data", "secret_answers.jsonl")
    test_filepath = "/mnt/c/users/jessi_/downloads/6.5940/project/streaming-llm-rag/data/mt_bench.jsonl"
    # test_filepath = os.path.join("data", "mt_bench.jsonl")
    
    # print(f"Loading data from {test_filepath} ...")

    # if not os.path.exists(test_filepath):
    #     download_url(
    #         "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
    #         args.data_root,
    #     )
    #     os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if args.enable_streaming:
        kv_cache = enable_streaming_llm_llama_index(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
        #  "--model_name_or_path", type=str, default="lmsys/vicuna-7b-v1.5"
        "--model_name_or_path", type=str, default="Jiayi-Pan/Tiny-Vicuna-1B"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=100)
    args = parser.parse_args()

    main(args)
