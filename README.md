CodeBERT Pre-training Project

This project focuses on pre-training a CodeBERT model for generating and completing code snippets in a specific programming language.

 CodeBERT is a versatile language model that can be fine-tuned for various software engineering tasks such as code generation and code completion.


Key Features

•	Pre-train CodeBERT on custom datasets

•	Scrape code snippets from websites

•	Optimize memory usage during training with gradient accumulation or small batch sizes

•	Utilize the fine-tuned model for code generation and completion tasks


My Machine Used

•	Ubuntu-based machine

•	At least 10GB of RAM

•	8 virtual processors

•	Python 3

•	Python libraries: transformers, pandas, beautifulsoup4


Dataset

•	Small dataset of code snippets for pre-training is included in smalldataset.py

•	Obtain larger datasets from sources such as GitHub Python Code Snippets, Java Code Snippets, Python Code Snippets, and C++ Code Snippets


Pre-training CodeBERT

1.	Fine-tune the CodeBERT model on the selected dataset

2.	Model configuration and training script can be found in codebertmodel.py

3.	Choose between train_with_gradient_accumulation.py and train_with_small_batch_size.py for optimizing memory usage during training


Website Scraping

•	Use websiterip.py to scrape code snippets related to a specific programming language

•	Modify the script to target other websites as needed


Results
•	Use the fine-tuned CodeBERT model for various software engineering tasks, such as code generation and code completion


Extra Info

•	codebertmodel.py: Python script defining the CodeBERT model

•	README.md: Markdown file containing project information and documentation

•	smalldataset.py: Python script defining a small dataset for training or evaluation

•	websiterip.py: Python script for downloading and extracting code blocks from websites

•	train_with_small_batch_size.py: Python script for training with a small batch size to reduce memory usage

•	train_with_gradient_accumulation.py: Python script for training with gradient accumulation to improve stability


Note: The train_data folder has been added, but the data inside requires preprocessing before use.

generate_description.py script summery

Import the necessary libraries, which include the GPT-2 tokenizer and model from the Hugging Face Transformers library, as well as JSON and PyTorch.

Load the pretrained GPT-2 tokenizer and model.

Define a function called generate_description that takes a code example as input and returns a generated description. The function:

a. Prepares an input text by adding a prompt to describe the given Arma 3 code.

b. Tokenizes the input text and creates an attention mask (a tensor of ones with the same shape as the input_ids tensor).

c. Generates a description using the GPT-2 model, taking into account the input_ids and attention_mask tensors.

d. Decodes the generated output and returns the description as a string.

Read the "arma3_commands.json" file and load its content into a variable called commands_data.

Iterate through the commands_data list, and for each command with the description "No description available," use the generate_description function to generate a new description based on the command's example code. The generated description replaces the existing description in the command dictionary.

Print the code example and the corresponding generated description.

Write the updated commands_data list to a new JSON file called "arma3_commands_with_descriptions.json".










