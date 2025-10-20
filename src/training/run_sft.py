import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import wandb
import os

def main(args):
    """
    Main function to run the SFT training.
    """
    
    print(f"Starting SFT for task: {args.task_name}")

    # --- 1. Initialize W&B ---
    # Log in from your terminal first: `wandb login`
    run_name = f"sft-{args.task_name}-{wandb.util.generate_id()}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # --- 2. Load Tokenizer ---
    # We use the Llama 3.1 instruct tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    # Llama 3.1 uses a new EOS token but TRL/SFTTrainer expects a standard PAD token.
    # We can set the pad token to a new special token or an existing one like EOS.
    # Using EOS as PAD token is a common practice for autoregressive models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Tokenizer loaded.")

    # --- 3. Load Dataset ---
    # SFTTrainer is smart and can handle different dataset formats.
    # For HH-RLHF, it expects a column 'chosen' (the good response)
    # For OpenBookQA, we need to format it into a single 'text' column.
    
    def format_openbookqa(example):
        # This function formats the OpenBookQA dataset into a single text string
        # that the SFTTrainer can use.
        # Format: "Q: [Question]\nA: [Correct Answer]"
        question = example['question_stem']
        correct_answer = example['choices']['text'][ord(example['answerKey']) - ord('A')]
        return {"text": f"Q: {question}\nA: {correct_answer}"}

    if args.task_name == "openbookqa":
        dataset = load_dataset(args.dataset_name, args.dataset_subset, split="train")
        dataset = dataset.map(format_openbookqa)
    elif args.task_name == "safety":
        dataset = load_dataset(args.dataset_name, data_dir=args.dataset_subset, split="train")
        # Rename 'chosen' column to 'text' for SFTTrainer
        dataset = dataset.rename_column("chosen", "text")
    else:
        raise ValueError(f"Unknown task name: {args.task_name}")
        
    print(f"Dataset '{args.task_name}' loaded and formatted.")

    # --- 4. Load Model (with QLoRA) ---
    # This is the core of QLoRA: loading the model in 4-bit.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto", # Automatically maps model layers to devices
        trust_remote_code=True,
    )
    model.config.use_cache = False # Disable caching for training
    
    # Prep model for k-bit training (essential step)
    model = prepare_model_for_kbit_training(model)
    
    print("Base model loaded in 4-bit.")

    # --- 5. Configure LoRA ---
    # This tells PEFT which layers to apply the adapters to.
    # For Llama 3.1, 'q_proj', 'k_proj', 'v_proj', 'o_proj' are common targets.
    peft_config = LoraConfig(
        r=16,  # LoRA attention dimension
        lora_alpha=32,  # Alpha for scaling
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # See how few params we're training!

    # --- 6. Configure Training ---
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_adapter_dir, args.task_name),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        fp16=False, # Use bf16 if your GPU supports it (Ampere+)
        bf16=True,
        report_to="wandb",
        run_name=run_name,
    )

    # --- 7. Initialize SFTTrainer ---
    # SFTTrainer from TRL is a wrapper that simplifies everything.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text", # Tell the trainer which column has our text
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Trainer initialized. Starting training...")
    
    # --- 8. Train ---
    trainer.train()

    # --- 9. Save Adapter ---
    # This only saves the tiny LoRA adapter, not the whole model.
    adapter_save_path = os.path.join(args.output_adapter_dir, args.task_name)
    trainer.save_model(adapter_save_path)
    
    wandb.finish()
    print(f"Training complete. Adapter saved to {adapter_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT with QLoRA for a specific task.")
    
    # Pulling arguments from our (future) shell script
    parser.add_argument("--task_name", type=str, required=True, help="Task identifier (e.g., 'openbookqa', 'safety')")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset on Hugging Face Hub")
    parser.add_argument("--dataset_subset", type=str, default=None, help="Subset/config of the dataset (if any)")
    parser.add_argument("--output_adapter_dir", type=str, default="models/adapters")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--wandb_project", type=str, default="llm-alignment-pipeline")

    args = parser.parse_args()
    main(args)
