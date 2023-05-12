import chardet
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf

# Read the data file.
with open('tokinized.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

with open("tokinized.csv", 'r', encoding=encoding) as f:
    _data_tokinized = pd.read_csv(f)

# Read the data file.
with open('data_cleaned.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

with open("data_cleaned.csv", 'r', encoding=encoding) as f:
    _data_cleaned = pd.read_csv(f)

# Get the number of sentences in all the data.
k = 1


# Class for calculating tf-idf weights.
class TfIdf:
    # Constructor.
    def __init__(self, documents):
        self.documents = documents
        self.tfidf = tfidf()
        self.document_vectors = self.tfidf.fit_transform(documents)
        

    # Method for calculating tf-idf weights.
    def get_tfidf_weights(self, query):
        query_vector = self.tfidf.transform([query])
        return query_vector

    # Method for getting the top k sentences based on TF-IDF scores.
    def get_top_k_sentences(self, query, k):
        # Get the TF-IDF weights for the query.
        query_vector = self.get_tfidf_weights(query)

        # Calculate the cosine similarity between the query and the document.
        cosine_similarity = (self.document_vectors * query_vector.T).toarray()

        # Get the indices of the top k sentences based on the cosine similarity.
        top_k_indices = cosine_similarity.argsort()[::-1][:k]

        # Get the actual sentences corresponding to the top k indices.
        top_k_sentences = [self.documents[i].values[0] for i in top_k_indices]

        # Return the top k sentences.
        return top_k_sentences


# Main function to run the summarizer and return the summary of the document.
def run_summarizer(documents, query, k):
    # create a sammary object. This object will be used to calculate the tf-idf weights. The constructor takes the list of documents as input. The documents are the sentences in the document. The documents are passed as a list of lists. Each list contains a single sentence. The list of lists is passed as a list of strings. Each string contains a single sentence. The list of strings is passed as a list of dictionaries. Each dictionary contains a single key-value pair. The key is 'sentence' and the value is the sentence. The list of dictionaries is passed as a pandas dataframe. The dataframe has a single column named 'sentence'.
    summarizer = TfIdf(documents)

    # Get the top k sentences based on the TF-IDF scores.
    top_k_sentences = summarizer.get_top_k_sentences(query, k)

    # Return the top k sentences as a single string.
    return ' '.join(top_k_sentences)


summary = run_summarizer(_data_tokinized['summary'], _data_cleaned['headlines'][7], k)
# Print the summary.
print(summary)

print("Done")
