import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

# Load the dataset (replace 'train.csv' with the actual file path)
df = pd.read_csv('train.csv')

# Check the first few rows
print("Initial Data:")
print(df.head())

# Define a function to clean and preprocess text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''

# Apply text cleaning
df['cleaned_text'] = df['crimeaditionalinfo'].apply(clean_text)

# Define labels and split data
df['label'] = df['category'].astype('category').cat.codes  # Convert category to numeric labels
X = df['cleaned_text'].values
y = df['label'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Create a DataLoader for our training data
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Parameters
MAX_LEN = 128
BATCH_SIZE = 8  # Reduced batch size to avoid memory issues
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create DataLoader
train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

# Define MLP Model
class MLP(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(MLP, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(bert_model.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Get the pooled output
        x = self.dropout(pooled_output)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model and move it to the device (GPU or CPU)
model = MLP(bert_model, num_classes=len(df['label'].unique())).to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training loop with debug statements
model.train()
for epoch in range(5):  # For simplicity, training for 5 epochs
    print(f"Starting epoch {epoch + 1}")
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Calculate loss
        loss = loss_fn(outputs, labels)
        
        # Print loss for debugging
        print(f"Batch {i+1}, Loss: {loss.item()}")

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} completed")

torch.save(model.state_dict(), 'model.pth')

print("Success")
