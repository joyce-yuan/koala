import torch
from tqdm import tqdm
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

def evaluate_perplexity(model, tokenizer, dataset, max_samples=10, max_eval_tokens=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    num_eval_tokens = 0

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    log_file = open("output/log.txt", "w")

    try:
        for text in dataset["text"][:max_samples]:
            # Ensure text is long enough
            if len(text) < 10:
                print(f"Skipping short text: {text}")
                continue

            # Tokenize the text
            encodings = tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids.to(device)
            
            # Validate input
            if input_ids.size(1) < 2:
                print(f"Input too short. Skipping. Length: {input_ids.size(1)}")
                continue

            seq_len = input_ids.size(1)
            print(f"Processing sequence of length: {seq_len}")

            past_key_values = None
            
            # Iterate through the sequence
            for idx in range(seq_len - 1):
                # Prepare input
                current_input = input_ids[:, idx:idx+1]
                target = input_ids[:, idx+1:idx+2]

                # Forward pass
                with torch.no_grad():
                    outputs = model(
                        current_input,
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
        with open("output/ppl.txt", "w") as f:
            f.write(f"{ppl.item()}\n")
        
        return ppl.item()
    else:
        print("No negative log-likelihoods collected. Check your data and model.")
        return None

def main():
    # Load model and tokenizer
    model_name = "lmsys/vicuna-13b-v1.3"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")

    # Evaluate perplexity
    evaluate_perplexity(model, tokenizer, dataset)

if __name__ == "__main__":
    main()