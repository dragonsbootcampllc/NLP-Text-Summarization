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
    return pd.read_csv("tokinized.csv")


def get_CleanData_content():
    return pd.read_csv("data_cleaned.csv")


def get_ctext_CleanData_content():
    _data_cleaned = pd.read_csv("data_cleaned.csv")
    content = ''
    for i in range(len(_data_cleaned)):
        content += _data_cleaned['ctext'][i]
    return content


def clean(sentences):
    lemmatizer = WordNetLemmatizer()
    cleaned_sentences = [' '.join([lemmatizer.lemmatize(word) for word in sentence.lower().split()
                                   if word not in set(stopwords.words('english'))])
                         for sentence in sentences]
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


def average_sentence_weights(sentences, probability_dict):
    sentence_weights = {}
    for index, sentence in enumerate(sentences):
        if len(sentence) != 0:
            average_proba = sum([probability_dict[word] for word in sentence if word in probability_dict.keys()])
            average_proba /= len(sentence)
            sentence_weights[index] = average_proba
    return sentence_weights


def generate_summary(sentence_weights, probability_dict, cleaned_sentences, sentences):
    summary = ""
    summary_length = 0
    for index, sentence in enumerate(cleaned_sentences):
        if summary_length < 300:
            if sentence_weights[index] > 0.5:
                summary += sentences[index]
                summary_length += len(sentences[index].split())

    # Calculate the probability of each word in the summary.
    words = word_tokenize(summary)
    total_words = len(set(words))
    if total_words == 0:
        total_words = 1
    for word in words:
        if word != ".":
            if not probability_dict.get(word):
                probability_dict[word] = 1
            else:
                probability_dict[word] += 1

    # Update the probability of each word in the probability dictionary.
    for word, count in probability_dict.items():
        probability_dict[word] = count / total_words

    return summary


def main():
    print("BasicSum algorithm is running...")
    topic = get_CleanData_content().iloc[0]['headlines']
    print("Topic: ", topic)
    article = get_ctext_CleanData_content()
    print("Article Extracted")
    tokenized_article = sent_tokenize(article)
    print("Article Tokenized")
    cleaned_article = clean(tokenized_article)
    print("Article Cleaned")
    probability_dict = init_probability(cleaned_article)
    print("Probability Dictionary Initialized")
    sentence_weights = average_sentence_weights(cleaned_article, probability_dict)
    print("Sentence Weights Calculated")
    summary = generate_summary(sentence_weights, probability_dict, cleaned_article, tokenized_article)
    print("Summary Generated")
    print("Summary: ")
    print(summary)


if __name__ == "__main__":
    main()
