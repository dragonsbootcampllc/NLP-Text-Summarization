import chardet
import networkx as nx
import numpy as np
import pandas as pd
from nltk.corpus import stopwords


def lexrank(documents, stopwords=None):
    """
    Computes the LexRank scores for the sentences in the given documents.

    Args:
        documents: A list of strings, where each string is a document.
        stopwords: A list of strings, where each string is a stopword.

    Returns:
        A list of floats, where each float is the LexRank score for the corresponding sentence.
    """
    # Create a graph where each node represents a sentence.
    graph = nx.Graph()
    for document in documents:
        for sentence in document.split('.'):
            if stopwords is not None and sentence in stopwords:
                continue
            graph.add_node(sentence)

    # Add edges to the graph, where the weight of each edge is the cosine similarity between the two sentences.
    for sentence1 in graph.nodes:
        for sentence2 in graph.nodes:
            if sentence1 == sentence2:
                continue
            similarity = np.dot(
                np.array(sentence1.split()),
                np.array(sentence2.split()).reshape(1, -1)
            ) / (
                np.linalg.norm(np.array(sentence1.split())) * np.linalg.norm(np.array(sentence2.split()).reshape(1, -1))
            )
            graph.add_edge(sentence1, sentence2, weight=similarity)

    # Calculate the PageRank scores for the sentences in the graph.
    scores = nx.pagerank(graph, alpha=0.85)

    return scores


def get_summary(documents, summary_size=3, stopwords=None):
    """
    Generates a summary of the given documents, using the LexRank algorithm.

    Args:
        documents: A list of strings, where each string is a document.
        summary_size: The number of sentences to include in the summary.
        stopwords: A list of strings, where each string is a stopword.

    Returns:
        A string, representing the summary of the given documents.
    """
    # Calculate the LexRank scores for the sentences in the documents.
    scores = lexrank(documents, stopwords=stopwords)

    # Sort the sentences by their LexRank scores, in descending order.
    sentences = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Get the top `summary_size` sentences.
    summary = sentences[:summary_size]

    # Join the sentences together, separated by a period.
    summary = ". ".join(summary)

    return summary


def main():
    # Load the documents.
    with open('data_cleaned.csv', 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    with open("data_cleaned.csv", 'r', encoding=encoding) as f:
        _data_cleaned = pd.read_csv(f)
        content = ''
        for i in range(len(_data_cleaned)):
            content += _data_cleaned['ctext'][i]
    documents = content.split('.')
    print("This is the summary of the article: ")
    # Generate a summary of the documents.
    summary = get_summary(documents, summary_size=3, stopwords=set(stopwords.words('english')))

    # Print the summary.
    print(summary)


if __name__ == '__main__':
    main()
