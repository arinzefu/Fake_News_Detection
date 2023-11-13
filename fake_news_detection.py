

import pandas as pd
import numpy as np

data = pd.read_csv('news.csv')

data.head()

data.shape

data.drop('Unnamed: 0', axis=1)

from matplotlib import pyplot as plt

fake_news_counts = data['label'].value_counts()

# Plot the counts as a bar chart
fake_news_counts.plot.bar()

# Add axis labels and title
plt.xlabel('Fake News')
plt.ylabel('Count')
plt.title('Fake News Counts')

# Display the plot
plt.show()

fake_news_counts

# Define a dictionary to map 'Real' and 'Fake' values to 1 and 0
mapping = {'REAL': 1, 'FAKE': 0}

# Apply mapping to label column
cols_to_map = ['label']
for col in cols_to_map:
    data[col] = data[col].map(mapping)

# Display the updated dataframe
print(data.head())

data.drop('Unnamed: 0', axis=1)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import LongformerTokenizer

# Load a pre-trained Longformer tokenizer
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Tokenize the dataset
input_ids = []
attention_mask = []

for index, row in data.iterrows():
    input_text = row["title"] + " " + row["text"]
    tokenized = tokenizer(input_text, padding="max_length", truncation=True, max_length=512)
    input_ids.append(tokenized["input_ids"])
    attention_mask.append(tokenized["attention_mask"])

input_ids = torch.tensor(input_ids, dtype=torch.long)
attention_mask = torch.tensor(attention_mask, dtype=torch.long)
labels = torch.tensor(data["label"].values, dtype=torch.long)

print(input_ids)

print(attention_mask)

# Split set
batch_size = 16
tokenized_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
train_data, test_data = train_test_split(tokenized_dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Create DataLoaders for training, validation, and test
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for batch_idx, batch in enumerate(train_dataloader):
    if batch_idx == 0:
        input_ids, attention_mask, labels = batch
        # Print or process the first batch here
        print("Batch 0 - Input IDs:", input_ids)
        print("Batch 0 - Attention Mask:", attention_mask)
        print("Batch 0 - Labels:", labels)
        break  # Stop after processing the first batch

for batch_idx, batch in enumerate(test_dataloader):
    if batch_idx == 0:
        input_ids, attention_mask, labels = batch
        # Print or process the first batch here
        print("Batch 0 - Input IDs:", input_ids)
        print("Batch 0 - Attention Mask:", attention_mask)
        print("Batch 0 - Labels:", labels)
        break  # Stop after processing the first batch

from transformers import LongformerModel, LongformerTokenizer

class FakeNewsDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FakeNewsDetectionModel, self).__init__()
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.longformer.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Using the [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

device = torch.device("cpu")

# instantiate your model
fake_news_model = FakeNewsDetectionModel(num_classes=2).to(device)

print(fake_news_model)

# define your loss function
criterion = nn.CrossEntropyLoss()

# define your optimizer
optimizer = torch.optim.Adam(fake_news_model.parameters(), lr=0.001)

from tqdm import tqdm
num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        # Unpack the batch into input_ids, attention_mask, and labels
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = fake_news_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # print running loss for each batch
        running_loss += loss.item()
        avg_loss = running_loss / len(train_dataloader)
        avg_acc = correct_predictions / total_predictions
        tqdm.write(f'Train Loss: {avg_loss:.3f}, Train Acc: {avg_acc:.3f}', end='\r')
    tqdm.write(f'Epoch {epoch+1}, Train Loss: {avg_loss:.3f}, Train Acc: {avg_acc:.3f}')

    print(f"Epoch {epoch+1} finished")

# Validation loop
with torch.no_grad():
    fake_news_model.eval()  # Set the model to evaluation mode
    valid_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for batch in val_dataloader:
        # Unpack the batch into input_ids, attention_mask, and labels
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

        # forward
        outputs = fake_news_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        # calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # calculate running loss
        valid_loss += loss.item()

    avg_loss = valid_loss / len(val_dataloader)
    avg_acc = correct_predictions / total_predictions
    print(f'Validation Loss: {avg_loss:.3f}, Validation Acc: {avg_acc:.3f}')

# Test loop
with torch.no_grad():
    fake_news_model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for batch in test_dataloader:
        # Unpack the batch into input_ids, attention_mask, and labels
        input_ids, attention_mask, labels = [tensor.to(device) for tensor in batch]

        # forward
        outputs = fake_news_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        # calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # calculate running loss
        test_loss += loss.item()

    avg_loss = test_loss / len(test_dataloader)
    avg_acc = correct_predictions / total_predictions
    print(f'Test Loss: {avg_loss:.3f}, Test Acc: {avg_acc:.3f}')

# save the model
torch.save(fake_news_model.state_dict(), 'fake_news_model.pth')

# Sample text to evaluate
sample_text = "Drinking Age at Disney World May be Lowered to 18-- The National Minimum Drinking Age Act was passed by congress and signed into law by President Ronald Reagan in 1984. It “requires that States prohibit persons under 21 years of age from purchasing or publicly possessing alcoholic beverages as a condition of receiving State highway funds.” This was an act to encourage states to raise the minimum drinking age to 21. As it states, it is not mandatory that states set the drinking age at 21, but if a state doesn’t implement 21 as a minimum, the government will withhold state highway funds. Didn’t think you would get a history lesson from us, did you? Now that we have set up the act, we have some Disney news to go with it. Disney World is looking to defy the minimum drinking age act. The Walt Disney Company is currently battling the state of Florida in the courts over the minimum drinking age. Disney is attempting to lower the minimum drinking age on Disney property to 18. They are clearly doing this to increase their revenue at EPCOT and across Disney World. We all know how popular drinks are at EPCOT. Whether you are having a few different concoctions or drinking around the world, alcoholic drinks are a big part of the EPCOT culture."

# List to store input IDs and attention masks
input_ids = []
attention_mask = []

# Tokenize and preprocess the sample text
tokenized = tokenizer(sample_text, padding="max_length", truncation=True, max_length=512)
input_ids.append(tokenized["input_ids"])
attention_mask.append(tokenized["attention_mask"])

# Convert input_ids and attention_mask to PyTorch Tensors
input_ids = torch.tensor(input_ids, dtype=torch.long)
attention_mask = torch.tensor(attention_mask, dtype=torch.long)

# Set the model to evaluation mode
fake_news_model.eval()

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Forward pass
with torch.no_grad():
    outputs = fake_news_model(input_ids=input_ids, attention_mask=attention_mask)
    predicted_class = torch.argmax(outputs, dim=1).item()

# Define class labels (0 for fake, 1 for real)
class_labels = ["FAKE", "REAL"]

# Get the predicted label
predicted_label = class_labels[predicted_class]

# Get the probability scores
probability_scores = torch.softmax(outputs, dim=1)
fake_probability = probability_scores[0][0].item()
real_probability = probability_scores[0][1].item()

# Print the result
print(f"Sample text: {sample_text}")
print(f"Predicted label: {predicted_label}")
print(f"Confidence - FAKE: {fake_probability * 100:.2f}%")
print(f"Confidence - REAL: {real_probability * 100:.2f}%")