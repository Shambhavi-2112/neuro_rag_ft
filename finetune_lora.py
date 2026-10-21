"""
Optional LoRA fine-tuning scaffold.
This is intentionally separate from the Streamlit app. Use a permitted, curated dataset
and a compatible model/runtime. Validate module names for the selected architecture.
Expected JSONL: each row has {"text": "training example"}.
"""
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", default="./adapter-output")
    parser.add_argument("--epochs", type=float, default=1.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # GPT-2 uses c_attn; other architectures require different target module names.
    targets = ["c_attn"] if "gpt2" in args.model.lower() else ["q_proj", "v_proj"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        lora_dropout=0.05, target_modules=targets
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.train_file, split="train")
    if "text" not in dataset.column_names:
        raise ValueError("Each training JSONL record must contain a 'text' field.")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=10, save_strategy="epoch",
        report_to="none"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized,
                      data_collator=collator)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter/model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
