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
from streaming_llm.kv_cache import StartRecentKVCache
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, kv_cache=None, kv_cache_type="koala"):
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
    if len(generated_ids) >= max_gen_len and generated_ids[-1] != tokenizer.eos_token_id:
        print("\n[WARNING] Sequence truncated due to length limit.")

    if kv_cache_type=="koala" and generated_response != "":
        kv_cache.store_text(generated_response)
    
    return past_key_values


@torch.no_grad()
def streaming_inference(model, tokenizer, kv_cache=None, max_gen_len=128, kv_cache_type="koala"):
    past_key_values = None
    idx = 0
    while True:
        prompt = input("Enter your prompt (or press Enter to stop and type FILL to fill context window): ").strip()
        if not prompt:
            break

        if prompt.lower() == "fill":
            large_text = " ".join(["random text"] * 250)  # Simulate filling with a large block of text
            print(f"[INFO] Filling context window with random large text.")
            input_ids = tokenizer(large_text, return_tensors="pt").input_ids.to(model.device)
            space_needed = input_ids.shape[1]
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

            # Feed the large text into the model to simulate filling the context window
            outputs = model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values  # Update past_key_values with the filled context
            # kv_cache.store_text(large_text)  # Add to the cache if applicable
            print("[INFO] Context window filled. You can now enter another prompt.")
            continue  # Skip generation and move to the next prompt
        
        prompt = "USER: " + prompt + " \nASSISTANT: "
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)

        if kv_cache_type == "koala":
            # Tokenize the prompt
            input_with_context = None
            # Get past key values and evict for space
            if idx == 0:
                seq_len = input_ids.shape[1]
                kv_cache.store_text(prompt) # store prompt
                space_needed = seq_len + max_gen_len
                past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)
                idx += 1
            else: 
                # For testing, try retrieving relevant past context for the second prompt
                # print("Retrieving relevant context")

                # Query past context with the current prompt
                past_context = kv_cache.retrieve_relevant_context(prompt)
                past_context_string = " ".join(past_context)

                print("Koala remembers: ", past_context_string)

                # Concat the past context with the current prompt
                input_with_context = past_context_string + " " + prompt
                input_ids = tokenizer(input_with_context, return_tensors="pt").input_ids
                input_ids = input_ids.to(model.device)
                
                # Get the length of past context
                seq_len = input_ids.shape[1]
                kv_cache.store_text(input_with_context) # store prompt with past context


                # Get past key values with past context included in cache
                space_needed = seq_len + max_gen_len
                past_key_values = kv_cache.evict_for_space(past_key_values, space_needed, past_context=past_context_string)

            # if input_with_context is not None:
            #     print("prompt: ", prompt, "input_with_context: \n\n", input_with_context, "\n\n")

            print("Koala is thinking...")
            past_key_values = greedy_generate(
                model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, kv_cache=kv_cache, kv_cache_type=kv_cache_type
            )
            print("Koala is done thinking!")

        if kv_cache_type == "start_recent":
            space_needed = input_ids.shape[1] + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)
            print("Streaming starting...")
            past_key_values = greedy_generate(
                model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, kv_cache=kv_cache, kv_cache_type=kv_cache_type
            )
            print("Streaming done!")
        
        if kv_cache_type == "none":
            print("Streaming starting...")
            past_key_values = greedy_generate(
                model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, kv_cache=kv_cache, kv_cache_type=kv_cache_type
            )
            print("Streaming done!")


def enable_streaming_llm_llama_index(model, start_size, recent_size, kv_cache_type="koala"):
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
    
    if kv_cache_type == "koala":
        print("KV cache enabled: Koala")
        kv_cache = LlamaIndexKVCache(
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
    elif kv_cache_type == "start_recent":
        print("KV cache enabled: StartRecent")
        kv_cache = StartRecentKVCache(
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
    else:
        kv_cache = None
        print("KV cache disabled")
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

    # test_filepath = os.path.join("data", "secret_words.jsonl")
    # list_data = load_jsonl(test_filepath)
    # prompts = []
    # for sample in list_data:
        # prompts += sample["turns"]

    if args.enable_streaming:
        kv_cache = enable_streaming_llm_llama_index(
            model, start_size=args.start_size, recent_size=args.recent_size, kv_cache_type=args.cache_type
        )
    else:
        kv_cache = None


    streaming_inference(
        model,
        tokenizer,
        kv_cache,
        kv_cache_type=args.cache_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
        #  "--model_name_or_path", type=str, default="lmsys/vicuna-7b-v1.5"
        "--model_name_or_path", type=str, default="Jiayi-Pan/Tiny-Vicuna-1B"
        # "--model_name_or_path", type=str, default="openlm-research/open_llama_3b_v2"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=512)
    parser.add_argument("--cache_type", type=str, default="koala")
    args = parser.parse_args()

    main(args)
