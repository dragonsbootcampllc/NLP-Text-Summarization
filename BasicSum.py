import chardet
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from bs4 import BeautifulSoup
import requests
import re
import nltk


def get_tokinizedData_content():
    with open('tokinized.csv', 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    with open("tokinized.csv", 'r', encoding=encoding) as f:
        _data_tokinized = pd.read_csv(f)
        return _data_tokinized


def get_CleanData_content():
    with open('data_cleaned.csv', 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    with open("data_cleaned.csv", 'r', encoding=encoding) as f:
        _data_cleaned = pd.read_csv(f)
        return _data_cleaned


def get_ctext_CleanData_content():
    with open('data_cleaned.csv', 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    with open("data_cleaned.csv", 'r', encoding=encoding) as f:
        _data_cleaned = pd.read_csv(f)
        content = ''
        for i in range(len(_data_cleaned)):
            content += _data_cleaned['ctext'][i]
        return content


def clean(sentences):
    lemmatizer = WordNetLemmatizer()
    cleaned_sentences = []
    print("loading... it will take a while(~2min)")
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
        sentence = sentence.split()
        sentence = [lemmatizer.lemmatize(word) for word in sentence if word not in set(stopwords.words('english'))]
        sentence = ' '.join(sentence)
        cleaned_sentences.append(sentence)
    print("cleaning done")
    return cleaned_sentences


def init_probability(sentences):
    probability_dict = {}
    words = word_tokenize('. '.join(sentences))
    total_words = len(set(words))
    for word in words:
        if word != '.':
            if not probability_dict.get(word):
                probability_dict[word] = 1
            else:
                probability_dict[word] += 1

    for word, count in probability_dict.items():
        probability_dict[word] = count / total_words

    return probability_dict


def update_probability(probability_dict, word):
    if probability_dict.get(word):
        probability_dict[word] = probability_dict[word] ** 2
    return probability_dict


def average_sentence_weights(sentences, probability_dict):
    sentence_weights = {}
    for index, sentence in enumerate(sentences):
        if len(sentence) != 0:
            average_proba = sum([probability_dict[word] for word in sentence if word in probability_dict.keys()])
            average_proba /= len(sentence)
            sentence_weights[index] = average_proba
    return sentence_weights


def generate_summary(sentence_weights, probability_dict, cleaned_article, tokenized_article, summary_length=30):
    summary = []
    summary_length = min(summary_length, len(tokenized_article))
    while len(summary) < summary_length:
        max_weight = max(sentence_weights.values())
        for index, weight in sentence_weights.items():
            if weight == max_weight:
                summary.append(tokenized_article[index])
                sentence_weights.pop(index)
                break

    summary = ' '.join(summary)
    summary = clean([summary])[0]
    summary = summary.split()
    for index, word in enumerate(summary):
        summary[index] = update_probability(probability_dict, word)
    summary = ' '.join(summary)
    return summary

def main():
    print("BasicSum algorithm is running...")
    topic = get_CleanData_content().iloc[0]['headlines']
    print("Topic: ", topic)
    article = get_ctext_CleanData_content()
    print("Article Extracted")
    required_length = 2
    print("Required Length: ", required_length)
    tokenized_article = sent_tokenize(article)
    print("Article Tokenized")
    cleaned_article = clean(tokenized_article)
    print("Article Cleaned")
    probability_dict = init_probability(cleaned_article)
    print("Probability Dictionary Initialized")
    sentence_weights = average_sentence_weights(cleaned_article, probability_dict)
    print("Sentence Weights Calculated")
    summary = generate_summary(sentence_weights, probability_dict, cleaned_article, tokenized_article, required_length)
    print("Summary Generated")
    print("Summary: ")
    print(summary)


if __name__ == "__main__":
    main()
