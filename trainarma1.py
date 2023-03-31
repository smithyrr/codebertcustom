import json
import os
import wandb
from transformers import AutoTokenizer, AutoModelForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Set up Weights and Biases (wandb) for experiment tracking
wandb.login()
wandb.init(project="arma3_codebert", name="training_run")

# Load JSON file
with open("/home/cognitron/codebert/arma3_commands_with_descriptions.json") as f:
    data = json.load(f)

# Extract code snippets from JSON file
code_snippets = []
for item in data:
    code_snippets.append(item["description"])

# Tokenize code snippets using CodeBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
tokenized_code = tokenizer.batch_encode_plus(
    code_snippets,
    add_special_tokens=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt"
)

# Create training dataset from tokenized code
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=None,
    batch_size=16,
    encoding="utf-8",
    limit_length=2048,
    max_length=512,
    lines=tokenized_code["input_ids"],
)

# Set up data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Load CodeBERT model
model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="/home/cognitron/codebert/training",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=500,
    save_steps=1000,
    eval_steps=1000,
    learning_rate=5e-5,
    warmup_steps=500
)

# Set up wandb logging for experiment tracking
training_args.report_to = "wandb"
training_args.log_train_metrics = True
training_args.log_eval_metrics = True
training_args.evaluate_during_training = True

# Set up wandb logging for model checkpointing
training_args.save_strategy = "steps"
training_args.save_steps = 1000
training_args.save_total_limit = 2
training_args.save_steps = 500

# Set up wandb logging for alerting and notifications
training_args.enable_app_mode = True
training_args.notify_email = "smithyyr@gmail.com"
training_args.early_stopping_patience = 5
training_args.early_stopping_threshold = 0.01

# Set up Trainer object and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)
trainer.train()

# Finish wandb run
wandb.finish()
