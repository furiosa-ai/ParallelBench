import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from dataset.parallel_bench import ParallelBench
from typing import Dict, List, Any

# --- Configuration ---

# Model identifier from the Hugging Face Hub
MODEL_NAME = "Dream-org/Dream-v0-Instruct-7B"

# Configuration for the maximum number of new tokens to generate for each task.
# This helps control the length of the model's output to match task requirements.
TASKS_MAX_TOKENS = {
    # Tasks involving list manipulation
    'waiting_line/copy': 32,
    'waiting_line/insert_index': 32,
    'waiting_line/insert_random': 32,
    'waiting_line/remove_index': 32,
    'waiting_line/remove_random': 32,
    'waiting_line/replace_index': 32,
    'waiting_line/replace_random': 32,
    'waiting_line/reverse': 32,
    'waiting_line/shuffle': 32,
    'waiting_line/sort': 32,

    # Text generation and summarization tasks
    'paraphrase_summarize/chatgpt-paraphrases': 64,
    'paraphrase_summarize/samsum': 64,
    'words_to_sentence/easy': 64,
    'words_to_sentence/hard': 64,
    'words_to_sentence/medium': 64,

    # Logic puzzle tasks
    'puzzle/latin_square_n4': 64,
    'puzzle/sudoku_n4_12': 64,
}


def load_model_and_tokenizer(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads the model and tokenizer from the Hugging Face Hub.

    Args:
        model_name (str): The identifier of the model to load.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    print(f"Loading model: {model_name}...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        device_map="auto"           # Automatically distribute model across available devices
    )
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    return model, tokenizer


def evaluate_on_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task_name: str,
    num_samples: int,
    task_configs: Dict[str, int]
) -> None:
    """
    Evaluates the model on a specific task from the ParallelBench dataset.

    Args:
        model: The pre-trained language model.
        tokenizer: The tokenizer corresponding to the model.
        task_name (str): The name of the task to evaluate.
        num_samples (int): The number of samples from the dataset to test.
        task_configs (Dict[str, int]): A dictionary mapping task names to max tokens.
    """
    if task_name not in task_configs:
        raise ValueError(f"Task '{task_name}' not found in TASKS_MAX_TOKENS configuration.")

    print(f"\n--- Evaluating task: {task_name} ---")
    dataset = ParallelBench(task_name)
    max_tokens = task_configs[task_name]

    # Store model outputs and ground truth labels
    outputs: List[str] = []
    references: List[str] = []

    for i in range(num_samples):
        print(f"\nProcessing sample {i+1}/{num_samples}...")
        sample: Dict[str, Any] = dataset[i]

        # Prepare the input prompt using the chat template
        messages: List[Dict[str, str]] = sample["input"]["messages"]
        input_ids: torch.Tensor = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        )

        # Generate the output using the model's diffusion-based generation method
        generated_ids: torch.Tensor = model.diffusion_generate(
            input_ids.to(model.device),
            max_tokens=max_tokens,
            block_length=4,  # Parameters specific to the generation method
            steps=4,
            temperature=0.0  # Use 0.0 for deterministic, greedy output
        )

        # Decode only the newly generated tokens, skipping special tokens
        output_str: str = tokenizer.decode(
            generated_ids[0][len(input_ids[0]):],
            skip_special_tokens=True
        )

        # Store results for final metric computation
        references.append(sample["label"])
        outputs.append(output_str)

        print(f"  Input: {messages[-1]['content']}")
        print(f"  Reference Label: {sample['label']}")
        print(f"  Model Output: {output_str}")

    # Compute and display the evaluation metrics for the task
    print("\n--- Task complete. Computing metrics... ---")
    metrics = dataset.compute_metrics(outputs, references)
    print(f"Final Metrics for '{task_name}': {metrics}")


def main():
    """
    Main function to run the model evaluation script.
    """
    # --- Parameters ---
    task_to_run = "waiting_line/copy"
    num_samples_to_evaluate = 3
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Run the evaluation
    evaluate_on_task(
        model=model,
        tokenizer=tokenizer,
        task_name=task_to_run,
        num_samples=num_samples_to_evaluate,
        task_configs=TASKS_MAX_TOKENS
    )


if __name__ == "__main__":
    main()