import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base")

# Load your custom dataset
code_names_file = "/home/cognitron/codebertcustom/arma3/data/ready/code_names.txt"
descriptions_file = "/home/cognitron/codebertcustom/arma3/data/ready/descriptions.txt"

with open(code_names_file, "r") as f:
    code_names = f.readlines()

with open(descriptions_file, "r") as f:
    descriptions = f.readlines()

# Tokenize the data
input_texts = [code.strip() for code in code_names]
target_texts = [desc.strip() for desc in descriptions]

inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
targets = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True)

# Create a dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx]
        attention_mask = self.inputs["attention_mask"][idx]
        target_ids = self.targets["input_ids"][idx]
        target_attention_mask = self.targets["attention_mask"][idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_ids,
            "labels_attention_mask": target_attention_mask,
        }

dataset = CustomDataset(inputs, targets)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="/home/cognitron/codebertcustom/endmachine",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=5000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
)

# Define a data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Create the learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Update the Trainer to use the learning rate scheduler
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
)

# Train the model with the scheduler
trainer.train()


# Save the trained model in the specified directory
if not os.path.exists("/home/cognitron/codebertcustom/endmachine"):
    os.makedirs("/home/cognitron/codebertcustom/endmachine")
trainer.save_model("/home/cognitron/codebertcustom/endmachine")
