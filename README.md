# NLP-Text-Summarization

A Program that takes a text and gives a summary of the text. Generate a summary of the text by selecting the most important sentences in the text.

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

TextRank is an extractive text summarization algorithm that is based on the PageRank algorithm used by Google Search.

### Steps

#### Tokenize the Text

- Split the text into sentences.
- Split the sentences into words.
- Calculate the similarity between words.
- Calculate the similarity between sentences.

### Calculate A graph of Sentences

- Create a graph where the vertices are the sentences and the edges are the similarity scores between the sentences.
  
### Rank the Sentences

- Rank the sentences using the PageRank algorithm. The score of each sentence is based on the sum of the similarity scores that connect to it.
