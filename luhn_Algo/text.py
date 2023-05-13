import re
import nltk
import heapq
from nltk.corpus import stopwords
nltk.download('stopwords')

def preprocess(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize text into sentences
    sentences = nltk.sent_tokenize(text)
    # Tokenize each sentence into words
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    # Remove stop words and short words
    stop_words = set(stopwords.words('english'))
    words = [[word for word in sentence if word not in stop_words and len(word) > 2] for sentence in words]
    return words

def luhn_summarize(text, summary_size):
    # Preprocess text
    words = preprocess(text)
    # Calculate word frequency
    word_freq = {}
    for sentence in words:
        for word in sentence:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    # Calculate sentence scores
    sentence_scores = {}
    for i, sentence in enumerate(words):
        for word in sentence:
            if word in word_freq:
                if i not in sentence_scores:
                    sentence_scores[i] = word_freq[word]
                else:
                    sentence_scores[i] += word_freq[word]
    # Select top sentences
    summary_sentences = heapq.nlargest(summary_size, sentence_scores, key=sentence_scores.get)
    summary_sentences.sort()
    # Generate summary
    summary = ''
    for i in summary_sentences:
        summary += ' '.join(words[i]) + '. '
    return summary

# Read text file
with open('input.txt', 'r') as f:
    text = f.read()

# Generate summary
summary = luhn_summarize(text, 3)

# Print summary
print(summary)