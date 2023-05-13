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

def sumbasic_summarize(text, summary_size):
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
    # Calculate sentence probabilities
    sentence_probs = []
    for i, sentence in enumerate(words):
        sentence_prob = 0
        for word in sentence:
            sentence_prob += word_freq[word]
        sentence_prob /= len(sentence)
        sentence_probs.append((i, sentence_prob))
    # Select top sentences
    summary_sentences = []
    while len(summary_sentences) < summary_size:
        top_sentence = max(sentence_probs, key=lambda x: x[1])
        summary_sentences.append(top_sentence[0])
        for word in words[top_sentence[0]]:
            for i, sentence in enumerate(words):
                if i not in summary_sentences:
                    if word in sentence:
                        sentence_probs[i] *= word_freq[word] / len(sentence)
        sentence_probs[top_sentence[0]] = (top_sentence[0], -1)
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
summary = sumbasic_summarize(text, 3)

# Print summary
print(summary)