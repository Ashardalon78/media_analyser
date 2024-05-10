from class_media_analyser import media_analyser

ma = media_analyser('comment_analysis_trainer/saved_data/best_model_RF.pkl', 'instalyser/saved_data/ashardalon78_insta_comments.pkl')
#ma = media_analyser('comment_analysis_trainer/saved_data/best_model_RF.pkl', 'saved_datasets/ashardalon78_instadata.pkl')
#ma = media_analyser('comment_analysis_trainer/saved_data/best_model_RF.pkl', 'comment_analysis_trainer/saved_data/df_main.pkl')

#ma.prepare_comment_df() #to instalyser
ma.predict_comment_sentiment('comment_analysis_trainer/saved_data/cv.pkl', comments_col='Comments_text')
ma.rate_comment()
#ma.write_rated_to_df_main() #to instalyser
ma.write_rated_to_specific_df()
ma.plot_rating('Likes', xlabel='Post ID', ylabel='Popularity', show=False)
#ma.plot_rating('Comments_Rating')
ma.plot_rating('Rating_Total', xlabel='Post ID', ylabel='Popularity', show=True)
