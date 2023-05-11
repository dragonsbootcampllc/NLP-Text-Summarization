# Import the necessary libraries

import nltk
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ----------------------------------Data Loading--------------------------------------------
data = pd.read_csv('../Datasets/news_summary.csv', encoding='latin-1')

# Print the first 5 rows of the dataset
print(data.head())

# ----------------------------------Data Cleaning--------------------------------------------
# Remove the unwanted columns
data = data[['text', 'ctext']]
print(data.head())

# Check for null values in the dataset
print(data.isnull().sum())

# Remove the null values
data.dropna(inplace=True)

# print data after removing null values
print(data.isnull().sum())

# Check for duplicate values
print(data.duplicated().sum())

# Remove the duplicate values
data.drop_duplicates(inplace=True)

# print data after removing duplicate values
print(data.duplicated().sum())

# ----------------------------------Data Preprocessing--------------------------------------------
# Convert the text to lowercase
data['text'] = data['text'].apply(lambda x: x.lower())
data['ctext'] = data['ctext'].apply(lambda x: x.lower())

# Remove the punctuation
data['text'] = data['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
data['ctext'] = data['ctext'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Remove the stopwords
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
data['ctext'] = data['ctext'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# Lemmatization
lemmatizer = WordNetLemmatizer()
data['text'] = data['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
data['ctext'] = data['ctext'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
print(data.head())

# Stemming
ps = PorterStemmer()
data['text'] = data['text'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
data['ctext'] = data['ctext'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
print(data.head())

# rename the columns
data.rename(columns={'text': 'summary', 'ctext': 'text'}, inplace=True)

# ----------------------------------Save the cleaned data--------------------------------------------
# Save the cleaned data to a csv file
data.to_csv('../Datasets/cleaned_data.csv', index=False)
