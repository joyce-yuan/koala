import string

import re
import json

# Define a function to parse the log file
def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        logs = file.readlines()
    
    # Initialize variables
    examples = []
    current_example = {}
    pattern_question = re.compile(r'\[([\d]+)\] Question: (.+)')
    pattern_gold_answers = re.compile(r'\[([\d]+)\] Gold Answers: (.+)')
    pattern_non_rag_answer = re.compile(r'\[([\d]+)\] Non-RAG Answer: (.+)')
    pattern_rag_answer = re.compile(r'\[([\d]+)\] RAG Answer: (.+)')
    
    # Parse the log lines
    for line in logs:
        question_match = pattern_question.search(line)
        gold_answers_match = pattern_gold_answers.search(line)
        non_rag_answer_match = pattern_non_rag_answer.search(line)
        rag_answer_match = pattern_rag_answer.search(line)
        
        if question_match:
            if current_example:
                examples.append(current_example)
                current_example = {}
            example_id = int(question_match.group(1))
            current_example['id'] = example_id
            current_example['question'] = question_match.group(2)
        
        if gold_answers_match:
            example_id = int(gold_answers_match.group(1))
            if 'id' in current_example and current_example['id'] == example_id:
                current_example['gold_answers'] = eval(gold_answers_match.group(2))
        
        if non_rag_answer_match:
            example_id = int(non_rag_answer_match.group(1))
            if 'id' in current_example and current_example['id'] == example_id:
                current_example['non_rag_answer'] = non_rag_answer_match.group(2).strip()
        
        if rag_answer_match:
            example_id = int(rag_answer_match.group(1))
            if 'id' in current_example and current_example['id'] == example_id:
                current_example['rag_answer'] = rag_answer_match.group(2).strip()
    
    # Append the last example
    if current_example:
        examples.append(current_example)
    
    return examples

# Save the parsed examples to a JSON file
def save_examples_to_json(examples, output_file):
    with open(output_file, 'w') as file:
        json.dump(examples, file, indent=4)


def read_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data


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


def main():
    # # Path to your log file
    # log_file_path = 'examples/logging/2024-12-10_04:43:27_haystack.log'
    # # Path to output JSON file
    # output_json_path = 'parsed_examples.json'

    # # Parse the log and save the output
    # examples = parse_log_file(log_file_path)
    # save_examples_to_json(examples, output_json_path)

    # print(f"Parsed examples saved to {output_json_path}")
    print("Loading parsed examples...")

    examples = read_json_file('parsed_examples.json')
    print(f"Loaded {len(examples)} examples")

    rag_correct_count = 0
    non_rag_correct_count = 0
    rag_f1_sum = 0.0
    non_rag_f1_sum = 0.0

    total = len(examples)

    for example in examples: 
        try:
            if exact_match_score(example["non_rag_answer"], example["gold_answers"]):
                non_rag_correct_count += 1
            if exact_match_score(example["rag_answer"], example["gold_answers"]):
                rag_correct_count += 1

            non_rag_f1 = f1_score(example["non_rag_answer"], example["gold_answers"])
            rag_f1 = f1_score(example["rag_answer"], example["gold_answers"])
            non_rag_f1_sum += non_rag_f1
            rag_f1_sum += rag_f1
        except Exception as e:
            print(f"Error processing example {example['id']}: {e}")
            total -= 1

    print(total)

    print(f"Non-RAG Exact Match Accuracy: {non_rag_correct_count / total:.2f}")
    print(f"RAG Exact Match Accuracy: {rag_correct_count / total:.2f}")
    print(f"Non-RAG F1: {non_rag_f1_sum / total:.2f}")
    print(f"RAG F1: {rag_f1_sum / total:.2f}")


if __name__ == "__main__":
    main()
