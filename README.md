CodeBERT Pre-training Project
This project focuses on pre-training a CodeBERT model for generating and completing code snippets in a specific programming language. CodeBERT is a versatile language model that can be fine-tuned for various software engineering tasks such as code generation and code completion.
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

