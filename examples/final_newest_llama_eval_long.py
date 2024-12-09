import os
cwd = os.getcwd()

print(cwd)

import torch
from tqdm import tqdm
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.llama_index_newest import LlamaIndexKVCache
from streaming_llm.utils import load, download_url, load_jsonl

from streaming_llm.utils import parse_args

args = parse_args()

def get_kv_cache_params(model):
    """
    Determine KV cache dimensions based on model type
    """
    model_type = model.config.model_type.lower()
    
    if "llama" in model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return k_seq_dim, v_seq_dim

def evaluate_perplexity(
    model, 
    tokenizer, 
    dataset, 
    max_samples=10, 
    max_eval_tokens=1000,
    start_size=4, # amount of sink tokens
    recent_size=512, # cache size
    enable_kv_cache=True,
    kv_cache_type='start_recent'
):
    print(f"max_eval_tokens: {max_eval_tokens}")
    device = "cuda"
    model.to(device)
    model.eval()

    # Prepare KV Cache
    if enable_kv_cache:
        k_seq_dim, v_seq_dim = get_kv_cache_params(model)
        
        if enable_kv_cache == False:
            kv_cache = None
            print("KV cache disabled")
            kv_cache_type = "no_cache"
        elif kv_cache_type == 'start_recent':
            kv_cache = StartRecentKVCache(
                start_size=start_size,
                recent_size=recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
        elif kv_cache_type == 'llama_index':
            kv_cache = LlamaIndexKVCache(
                start_size=start_size,
                recent_size=recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
        else:
            raise ValueError(f"Unknown KV cache type: {kv_cache_type}")
    else:
        kv_cache = None
        print("KV cache disabled")

    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    num_eval_tokens = 0

    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    log_file = open(f"outputs/log_{kv_cache_type}.txt", "w")

    try:
        for text in dataset["text"][:max_samples]:
            # Ensure text is long enough
            if len(text) < 10:
                print(f"Skipping short text: {text}")
                continue

            # Tokenize the text
            encodings = tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids.to(device)
            input_with_context = None

            # Validate input
            if input_ids.size(1) < 2:
                print(f"Input too short. Skipping. Length: {input_ids.size(1)}")
                continue

            seq_len = input_ids.size(1)
            print(f"Processing sequence of length: {seq_len}")

            past_key_values = None

            # Store the prompt in the KV cache.
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

                kv_cache.store_text(original_prompt) # store prompt

            # Iterate through the sequence
            for idx in tqdm(range(seq_len - 1)):
                # Prepare input
                current_input = input_ids[:, idx:idx+1]
                target = input_ids[:, idx+1:idx+2]

                # Prepare subtext of the prompt
                subtext = (
                    tokenizer.decode(
                        # current_input, #
                        current_input[0], # TODO
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False
                    )
                    .strip()
                )

                seq_len = input_ids.shape[1]

                # Concat the past context with the current text
                input_with_context = past_context_string + " " + subtext

                # Forward pass
                with torch.no_grad():
                    outputs = model(
                        encodings.input_ids[:, idx : idx + 1].to(device), # current input ids
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    
                    logits = outputs.logits.view(-1, model.config.vocab_size)
                    past_key_values = outputs.past_key_values

                    # Compute loss
                    label = target.view(-1)

                    neg_log_likelihood = loss_fn(logits, label)
                    
                    # Store and log results
                    nlls.append(neg_log_likelihood)
                    log_file.write(f"{neg_log_likelihood.item()}\n")
                    log_file.flush()

                num_eval_tokens += 1
                
                # Break if we've evaluated enough tokens
                if num_eval_tokens >= max_eval_tokens:
                    break

            if num_eval_tokens >= max_eval_tokens:
                break

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print(f"Error details: {sys.exc_info()}")
        raise

    finally:
        log_file.close()

    # Compute perplexity
    if len(nlls) > 0:
        ppl = torch.exp(torch.stack(nlls).mean())
        print(f"Perplexity: {ppl.item()}")
        
        # Write perplexity to file
        with open(f"outputs/ppl_{kv_cache_type}.txt", "w") as f:
            f.write(f"{ppl.item()}\n")
        
        return ppl.item()
    else:
        print("No negative log-likelihoods collected. Check your data and model.")
        return None

# Evaluation function for the start recent cache enabled
def yappy_llama_cache_eval():
    # Load model and tokenizer
    model_name = "Jiayi-Pan/Tiny-Vicuna-1B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")

    print("Evaluating with llama index cache")
    # Evaluate perplexity with KV cache
    evaluate_perplexity(
        model, 
        tokenizer, 
        dataset, 
        enable_kv_cache=True,  # Enable KV cache
        kv_cache_type='llama_index',  # Choose between 'start_recent' and 'llama_index'
        start_size=4,
        recent_size=64,
        max_eval_tokens=args.num_eval_tokens,
    )

def main():
    yappy_llama_cache_eval()

if __name__ == "__main__":
    main()