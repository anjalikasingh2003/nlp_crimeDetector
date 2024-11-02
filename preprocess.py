import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary resources for nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset (replace 'train.csv' with the actual file path)
df = pd.read_csv('train.csv')

# Check the first few rows
print("Initial Data:")
print(df.head())

# Define a function to clean and preprocess text
def clean_text(text):
    if isinstance(text, str):  # Check if text is a string
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    return ''  # Return empty string for NaN values

# Apply text cleaning
df['cleaned_text'] = df['crimeaditionalinfo'].apply(clean_text)

# Remove stop words
stop_words = set(stopwords.words('english'))
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

# Apply stemming
ps = PorterStemmer()
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join([ps.stem(word) for word in word_tokenize(x)]))

# Check the cleaned data
print("Cleaned Data:")
print(df[['crimeaditionalinfo', 'cleaned_text']].head())
