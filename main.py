import numpy as np
import pandas as pd
from preprocessing import get_data, get_data_tokenized
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from collections import Counter
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import spacy
import matplotlib.pyplot as plt

# implement model that finds the most common topics among all the text messages

data_path = "D:/Projects/Codecademy/NaturalLanguageProcessing/PortfolioProject/clean_nus_sms.csv"
data_frame = get_data(data_path)

# tokenized_sents = [word_tokenize(sent) for sent in data_frame['Message']]
tokenized_sents = get_data_tokenized(data_path)

# bag_of_words = dict()
# for sent in tokenized_sents:
#     for word in sent:
#         if word in bag_of_words:
#             bag_of_words[word] += 1
#         else:
#             bag_of_words[word] = 1
           
# simple analysis of most frequent words
def most_frequent_words(list_of_sentences:list)->list:
    """
        Returns a list of the words that occur the most given a series of tokenized sentences.

        For optimal efficiency, preprocess the sentences first to removal as much unhelpful information as possible.
    """
    all_words = [word for sentence in list_of_sentences for word in sentence]
    return Counter(all_words).most_common()          

# Part 1: Most frequent words
most_freq_words = most_frequent_words(tokenized_sents)
print("The most frequently occuring words among the corpus.\n")
print(most_freq_words)
print("\n\n**************************\n\n")
# Part 2: Similarity to top 25 relevant words
all_text_embeddings = Word2Vec(tokenized_sents, vector_size=128, window=4, min_count=1, workers=2, sg=1, epochs=10)
# save Word2Vec model for future use
all_text_embeddings.save("D:/Projects/Codecademy/NaturalLanguageProcessing/PortfolioProject/word2vec_model.word2vec")
for i in range(25, 51):
    print(most_freq_words[i][0])
    most_sim = all_text_embeddings.wv.most_similar(most_freq_words[i][0], topn=20)
    print(most_sim) #Show text version of most similar data
    # Show graph of most similar data
    x_data = [point[0] for point in most_sim]
    y_data = [point[1] for point in most_sim]
    plt.plot(x_data, y_data)
    plt.title(most_freq_words[i][0])
    plt.show()
    print("\n#####\n")
# ###########################################

# Part 3: frequency distributions of words using tf-idf
print("\n\n###########################################\n\n")

tfidf_vec = TfidfVectorizer(norm=None)
tfidf_scores = tfidf_vec.fit_transform(data_frame['Message'])
# dataframe of tfidf data for each word
df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=tfidf_vec.get_feature_names(), columns=range(len(data_frame['Message'])))
print(df_tf_idf) #may cut out important information on display
