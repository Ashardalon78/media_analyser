import pickle
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.feature_extraction.text import CountVectorizer
from comment_analysis_trainer import class_comment_analyser as ca
from instalyser import class_instagram_client as ic

class media_analyser():

    def __init__(self, modelpath, datapath):

        with open(modelpath, 'rb') as filein:
            self.model = pickle.load(filein)

        with open(datapath, 'rb') as filein:
            #self.df_main = pickle.load(filein)
            self.df_comments = pickle.load(filein)

        self.ic = ic.InstagramClient.from_pickle('instalyser/saved_data/ashardalon78_instadata.pkl')

    def prepare_comment_df(self, comments_col='Comments_text', owner='ashardalon78'):
        comments = self.df_main[comments_col].explode().dropna()
        users = comments.apply(lambda x: x.user.username)
        users.name = 'users'
        #print(comments[102].values[1].user.username)
        comments = comments.apply(lambda x: x.text)

        self.df_comments = pd.DataFrame(pd.concat([comments, users],axis=1))
        self.df_comments = self.df_comments[self.df_comments['users']!=owner]
        self.df_comments = ca.CommmentAnalyser.transform_comments(self.df_comments, colname=comments_col)
        #print(self.df_comments)
        #print(self.df_main[comments_col][102])

    def predict_comment_sentiment(self, cvpath, comments_col='Comments_text'):
        self.df_comments = ca.CommmentAnalyser.transform_comments(self.df_comments, colname=comments_col)
        #self.df_main = ca.CommmentAnalyser.transform_comments(self.df_main, colname='Comments_text')
        #self.df_comments = ca.CommmentAnalyser.transform_comments(self.df_comments, colname='Comments_text')

        #print(self.df_main.columns)
        X = self.df_comments['Texts_Transformed']

        with open(cvpath, 'rb') as filein:
            cv = pickle.load(filein)
        # cv = CountVectorizer()
        # cv.fit(X)
        X = cv.transform(X)

        y_pred = self.model.predict(X.toarray())

        self.df_comments['Rating'] = y_pred

    def rate_comment(self, rating_factor=1, negative_is_negative=True):
        if negative_is_negative:
            conversion = {'negative': -rating_factor, 'neutral': 0, 'positive': rating_factor}
        else:
            conversion = {'negative': 0, 'neutral': rating_factor, 'positive': 2*rating_factor}

        self.df_comments['Rating'] = self.df_comments['Rating'].apply(lambda x: conversion[x])

    def write_rated_to_df_main(self):
        self.df_main['Comments_Rating'] = self.df_comments.groupby(self.df_comments.index).sum()['Rating']
        self.df_main['Comments_Rating'].fillna(0.0, inplace=True)
        self.df_main['Rating_Total'] = self.df_main['Likes'] + self.df_main['Comments_Rating']

    def write_rated_to_specific_df(self):
        self.ic.write_rated_to_df_user_data(self.df_comments)

    def plot_rating(self, yname, xlabel=None, ylabel=None, show=True):
        x, y = self.ic.get_plotting_data(yname)

        plt.plot(x, y, label=yname)
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
        plt.legend()
        if show: plt.show()
