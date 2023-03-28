from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base")
