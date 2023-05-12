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


def generate_summary(sentence_weights, probability_dict, cleaned_article, tokenized_article, summary_length=1):
    summary = []
    summary_length = min(summary_length, len(tokenized_article))
    while len(summary) < summary_length:
        max_weight = max(sentence_weights.values())
        for index, weight in sentence_weights.items():
            if weight == max_weight:
                summary.append(str(tokenized_article[index]))
                sentence_weights.pop(index)
                break

    summary = ' '.join(summary)
    summary = clean([summary])[0]
    summary_prob = sum([probability_dict[word] for word in summary.split() if word in probability_dict.keys()])
    return summary, summary_prob


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
    summary, summary_prob = generate_summary(sentence_weights, probability_dict, cleaned_article, tokenized_article, required_length)
    print("Summary Generated")
    print("Summary: ")
    print(summary)
    print("Summary Probability: ", summary_prob)


if __name__ == "__main__":
    main()
