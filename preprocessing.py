import numpy as np
import re
# import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas

# TODO: analysis: most common topics, frequency distributions, optional: sentiment analysis

# data_path = "D:/Projects/Codecademy/NaturalLanguageProcessing/PortfolioProject/clean_nus_sms.csv"

def get_data(data_path:str)->pandas.DataFrame:
    """
        Receives a string to retrieve the data from a csv file, then proceeds to perform preprocessing on the message data

        Path must lead to a csv file of similar style to clean_nus_sms.csv (File courtesy of Codecademy)

        Returns: DataFrame object with the text data preprocessed and ready to analyze
    """
    pandas_dataframe = pandas.read_csv(data_path)
    dataframe_msgs = pandas_dataframe['Message']
    for i in range(len(dataframe_msgs)):
    # preprocess message column data
        result = re.sub(r'[\.\?\!\,\:\;\"\<\>\(\)\&\*\#\@]', '', str(dataframe_msgs[i])) #remove noise
        # result = re.sub(r'\s\w\s', '', result)
        # make lowercase
        result = result.lower()
        # tokenize for preprocessing methods
        res_tokenized = word_tokenize(result)
        # lemmatization
        lemmatizer = WordNetLemmatizer()
        res_tokenized = [lemmatizer.lemmatize(token) for token in res_tokenized]
        # stopword removal
        stop_words = set(stopwords.words('english'))
        res_tokenized = [token for token in res_tokenized if token not in stop_words]
        result = ""
        for word in res_tokenized:
            result += " " + word
        dataframe_msgs[i] = result.strip() #TODO: fix error that this is throwing
    pandas_dataframe['Message'] = dataframe_msgs
    return pandas_dataframe

def get_data_tokenized(data_path:str)->list:
    return_list = list()
    pandas_dataframe = pandas.read_csv(data_path)
    dataframe_msgs = pandas_dataframe['Message']
    for i in range(len(dataframe_msgs)):
    # preprocess message column data
        result = re.sub(r'[\.\?\!\,\:\;\"\<\>\(\)\&\*\#\@]', '', str(dataframe_msgs[i])) #remove noise from symbols
        # result = re.sub(r'\s\w\s', '', result)
        # make lowercase
        result = result.lower()
        # tokenize for preprocessing methods
        res_tokenized = word_tokenize(result)
        # lemmatization
        lemmatizer = WordNetLemmatizer()
        res_tokenized = [lemmatizer.lemmatize(token) for token in res_tokenized]
        # stopword removal
        stop_words = set(stopwords.words('english'))
        res_tokenized = [token for token in res_tokenized if token not in stop_words]
        return_list.append(res_tokenized)
    return return_list
        

if __name__ == "__main__":
    data_path = "D:/Projects/Codecademy/NaturalLanguageProcessing/PortfolioProject/clean_nus_sms.csv"
    # data_frame = get_data(data_path)
    # print(data_frame)
    tokenized_sents = get_data_tokenized(data_path)
    print(tokenized_sents)