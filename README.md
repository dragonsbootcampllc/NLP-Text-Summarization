# NLP-Text-Summarization

A Program that takes a text and gives a summary of the text. Generate a summary of the text by selecting the most
important sentences in the text.
# Video Record

[![Luhn Algo](https://i.imgur.com/vKb2F1B.png)]([https://youtu.be/vt5fpE0bzSY](https://youtu.be/ej-HhM2Yo2I))
https://youtu.be/ej-HhM2Yo2I

## Types of Summarization

### Extractive Summarization

- Simple technique, easy to understand and implement.

- Effective for short text.

- Fast and Efficient.

### Abstractive Summarization

- Can generate summaries in a more natural way.

- Can generate summaries for long text.

- Can generate more human-like summaries.

## The Algorithms

### Extractive Summarization Algorithms

for example:

- Luhn’s Algorithm
- TextRank Algorithm
- TF-IDF Algorithm
- LexRank Algorithm
- SumBasic Algorithm

### Abstractive Summarization Algorithms

for example:

- Deep Learning Based Algorithms
- Sequence-to-Sequence Models
- Encoder-Decoder Models

## Algoritms Used in this Project

- TextRank Algorithm

- LexRank Algorithm

- SumBasic Algorithm

- Luhn’s Algorithm

## TextRank Algorithm

The TextRank algorithm is a graph-based algorithm for automatic text summarization. It works by analyzing the
relationships between words in a given text to identify the most important sentences that best represent the content of
the original text. Here are the key steps involved in the TextRank algorithm summarization process:

### How TextRank Works

1. The text is first preprocessed by removing stop words, punctuation, and other unnecessary characters. The remaining
   words are then stemmed or lemmatized to reduce the number of variations of the same word.

2. Building a graph: The TextRank algorithm builds a graph representation of the text, where each sentence is
   represented as a node and the edges between the nodes represent the similarity between the sentences. The similarity
   between two sentences is calculated based on the number of common words and their distance in the text.

3. Calculating scores: The TextRank algorithm calculates a score for each sentence based on its relationship with other
   sentences in the graph. The score is calculated using an iterative process, where the score of a sentence is updated
   based on the scores of the sentences that are connected to it.

4. Selecting top sentences: The TextRank algorithm then selects the top-ranked sentences as the summary of the text. The
   number of sentences selected can be adjusted based on the desired length of the summary.

## LexRank Algorithm

LexRank is an extractive text summarization algorithm that is based on the PageRank algorithm used by Google Search. It
is an unsupervised approach to summarization that builds a graph representation of the text and performs centrality
calculations to determine the most important sentences in the text.

### How LexRank Works

1. Preprocessing: The text is first preprocessed by removing stop words, punctuation, and other unnecessary characters.
   The remaining words are then stemmed or lemmatized to reduce the number of variations of the same word.

2. Building a graph: The LexRank algorithm builds a graph representation of the text, where each sentence is represented
   as a node and the edges between the nodes represent the similarity between the sentences. The similarity between two
   sentences is calculated based on the number of common words and their distance in the text, as well as the importance
   of the words in each sentence.

3. Calculating scores: The LexRank algorithm calculates a score for each sentence based on its relationship with other
   sentences in the graph. The score is calculated using an iterative process, where the score of a sentence is updated
   based on the scores of the sentences that are connected to it. The importance of the words in each sentence is also
   taken into account when calculating the sentence scores.

4. Selecting top sentences: The LexRank algorithm then selects the top-ranked sentences as the summary of the text. The
   number of sentences selected can be adjusted based on the desired length of the summary.

## SumBasic Algorithm

SumBasic is an extractive text summarization algorithm that is based on the idea of word frequency. It works by
assigning a score to each sentence based on the frequency of the words in the sentence. The sentences with the highest
scores are then selected as the summary of the text.

### How SumBasic Works

1. Preprocessing: The text is first preprocessed by removing stop words, punctuation, and other unnecessary characters.
   The remaining words are then stemmed or lemmatized to reduce the number of variations of the same word.

2. Assigning probabilities: The SumBasic algorithm assigns a robability distribution to each word in the text based on
   its frequency in the text. The probability of a word is calculated as the number of times it appears in the text
   divided by the total number of words in the text.

3. Selecting sentences: The SumBasic algorithm then selects sentences from the text that contain the most probable
   words. The algorithm starts by selecting the most probable word in the text and then selects sentences that contain
   that word. It then updates the probability distribution of the words in the selected sentences and repeats the
   process until the desired length of the summary is reached.

4. Generating summary: Finally, the SumBasic algorithm generates a summary by concatenating the selected sentences.

## Luhn’s Algorithm

Luhn’s algorithm is an extractive text summarization algorithm that is based on the idea of word frequency. It works by
assigning a score to each sentence based on the frequency of the words in the sentence. The sentences with the highest
scores are then selected as the summary of the text.

### How Luhn’s Algorithm Works

1. Preprocessing: The text is first preprocessed by removing stop words, punctuation, and other unnecessary characters.
   The remaining words are then stemmed or lemmatized to reduce the number of variations of the same word.

2. Identifying important words: Luhn's Algorithm identifies the most important words in the text by assigning each word
   a weight based on its frequency of occurrence. Words that occur frequently in the text but are not stop words or
   common words are assigned a higher weight.

3. Scoring sentences: Luhn's Algorithm then scores each sentence based on the frequency of occurrence of the important
   words within the sentence. Sentences that contain more important words are assigned a higher score.

4. Selecting top sentences: Luhn's Algorithm then selects the top-ranked sentences based on their score. The number of
   sentences selected can be adjusted based on the desired length of the summary.

5. Generating summary: Finally, Luhn's Algorithm generates a summary by concatenating the selected sentences.
