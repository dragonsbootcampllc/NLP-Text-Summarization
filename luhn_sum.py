# luhn_sum algorithm for text summarization.
import re

import chardet
import content as content
import pandas as pd
from nltk import word_tokenize, sent_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

word_limit = 300


def get_tokinizedData_content():
    with open('tokinized.csv', 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    with open("tokinized.csv", 'r', encoding=encoding) as f:
        _data_tokinized = pd.read_csv(f)
        return _data_tokinized


def clean(article):
    lem = WordNetLemmatizer()
    article = re.sub(r'\[[0-9]*\]', ' ', article)
    article = sent_tokenize(article)
    cleaned_list = []
    for sent in article:
        sent = sent.lower()
        word_list = []
        words = word_tokenize(sent)
        for word in words:
            word_list.append(lem.lemmatize(word.lower()))
        cleaned_list.append(' '.join(word_list))
    return cleaned_list


def get_CleanData_content():
    with open('data_cleaned.csv', 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    with open("data_cleaned.csv", 'r', encoding=encoding) as f:
        _data_tokinized = pd.read_csv(f)
        return _data_tokinized


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


def get_frequency_dictionary(cleaned_content):
    frequency_dictionary = {}
    for sentence in cleaned_content:
        word_list = word_tokenize(sentence)
        print("loading... it will take a while(~2min)")
        for word in word_list:
            if word not in set(stopwords.words('english')).union(
                    {',', '.', ';', '%', ')', '(', '``'}):
                if word not in frequency_dictionary.keys():
                    frequency_dictionary[word] = 1
                else:
                    frequency_dictionary[word] += 1
    return frequency_dictionary


def get_score(cleaned_content, frequency_dictionary):
    sentence_scores = {}
    print("the process is starting... it will take a while(~2min)")
    for index, sentence in enumerate(cleaned_content):
        word_list = word_tokenize(sentence)
        for word in word_list:
            if word in frequency_dictionary.keys():
                if len(sentence.split(' ')) < word_limit:
                    if sentence not in sentence_scores.keys():
                        sentence_scores[index] = frequency_dictionary[word]
                    else:
                        sentence_scores[index] += frequency_dictionary[word]
    return sentence_scores


def get_summary(sentence_scores, cleaned_content, threshold):
    summary = ''
    print("the process is starting... it will take a while(~2min)")
    for index in sorted(sentence_scores, reverse=True):
        if len(summary.split(' ')) < threshold:
            summary += cleaned_content[index]

    return summary


def main():
    print("Luhn Summarizer")
    topic_name = get_CleanData_content().iloc[0]['headlines']

    print("Topic: ", topic_name)
    print("Cleaned Content: ")
    cleaned_content = get_tokinizedData_content()['summary']
    print("cleaned_content Done")
    threshold = len(cleaned_content) // 40
    print("Threshold: ", threshold)
    # frequency_dictionary
    print("Frequency Dictionary Phase: ")
    frequency_dictionary = get_frequency_dictionary(cleaned_content)
    print("frequency_dictionary Done")
    # sentence_scores
    print("Sentence Scores Phase: ")
    sentence_scores = get_score(cleaned_content, frequency_dictionary)
    print("Sentence Scores is Done")
    print(sentence_scores)
    print("sentence_scores Done")
    # use sentence_scores to get summary
    summary = get_summary(sentence_scores, cleaned_content, threshold)
    print("Summary Phase: ")

    outer_summary = re.sub(r'[^a-zA-Z0-9]', ' ', summary)
    # remove extra spaces
    outer_summary = re.sub(r'\s+', ' ', outer_summary)
    print("Summary: ", outer_summary)
    print("Summary Done - Luhn Summarizer")
    print("Luhn Summarizer Done")


if __name__ == '__main__':
    main()
