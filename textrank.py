import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np

nltk.download("punkt")
nltk.download("stopwords")

def read_transcript(transcript_file):
    with open(transcript_file, 'r', encoding='utf-8') as file:
        transcript = file.read()
    return transcript

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    words1 = word_tokenize(sent1)
    words2 = word_tokenize(sent2)

    words1 = [word.lower() for word in words1 if word.isalnum() and word.lower() not in stopwords]
    words2 = [word.lower() for word in words2 if word.isalnum() and word.lower() not in stopwords]

    all_words = list(set(words1 + words2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in words1:
        vector1[all_words.index(word)] += 1

    for word in words2:
        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stopwords=None):
    if stopwords is None:
        stopwords = []

    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stopwords)

    return similarity_matrix

def generate_summary(transcript, top_n=5):
    sentences = sent_tokenize(transcript)
    stop_words = set(stopwords.words("english"))

    similarity_matrix = build_similarity_matrix(sentences, stop_words)

    scores = np.array([np.sum(similarity_matrix[i]) for i in range(len(sentences))])
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[::-1][:top_n]]

    return " ".join(ranked_sentences)

if __name__ == "__main__":
    transcript_file = input("Enter the file path for the transcript: ")
    summary = generate_summary(read_transcript(transcript_file), top_n=5)
    print("Generated Summary:")
    print(summary)
    print(len(summary))
