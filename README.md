# Fake_News_Detection

This script uses Longformer transformer over BERT.

The dataset is from https://www.kaggle.com/datasets/rajatkumar30/fake-news
 
Longformer is a transformer model for long documents.

longformer-base-4096 is a BERT-like model started from the RoBERTa checkpoint and pretrained for MLM on long documents. It supports sequences of length up to 4,096.

Longformer uses a combination of a sliding window (local) attention and global attention. Global attention is user-configured based on the task to allow the model to learn task-specific representations. Please refer to the examples in modeling_longformer.py and the paper for more details on how to set global attention.



This Python script demonstrates how to perform fake news detection using a Longformer-based model. The script follows these main steps:

- Data Loading and organization
   
- Data Visualization by using Matplotlib to plot a bar chart of fake news counts.
- Defines a mapping dictionary to convert 'REAL' and 'FAKE' labels to 1 and 0 in the label column

- Data Tokenization: Tokenizing a dataset from huggingface and a dataset from a csv file follows a slightly different process. Tokenizes the text data using a Longformer-based tokenizer.
- Splits the dataset into training, validation, and test sets.
- Creates data loaders for each set to feed data into the model.

- Defines a custom FakeNewsDetectionModel that uses a pre-trained Longformer model.

- Trains the model for one epoch, calculating accuracy and loss on the training set.

- Evaluates the model on the validation set, calculating accuracy and loss.

- Tests the model on the test set, calculating accuracy and loss.

- Demonstrates how to use the trained model to predict the label (FAKE or REAL) and provide confidence scores for a sample text.


Initial training period for the script was supposed to be 83 hours per epoch, so the max_length was reduced from 4096 to 512 and the efficiency of the model also reduced so it did not train very well.