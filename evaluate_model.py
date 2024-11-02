import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer
from main import MLP  # Replace with the actual name of your training script

# Load the test dataset
df_test = pd.read_csv('test.csv')

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''

df_test['cleaned_text'] = df_test['crimeaditionalinfo'].apply(clean_text)
X_test = df_test['cleaned_text'].values
y_test = df_test['category'].astype('category').cat.codes.values

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = MLP(bert_model, num_classes=len(df_test['category'].unique()))
model.load_state_dict(torch.load('model.pth'))
model.eval()

all_preds = []
all_labels = []

# Evaluate the model
with torch.no_grad():
    for text in X_test:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)

        all_preds.append(preds.numpy())
        all_labels.append(y_test)

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
