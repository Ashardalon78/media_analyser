from class_media_analyser import media_analyser

ma = media_analyser('comment_analysis_trainer/saved_data/best_model_RF.pkl', 'saved_datasets/ashardalon78_instadata.pkl')
#ma = media_analyser('comment_analysis_trainer/saved_data/best_model_RF.pkl', 'comment_analysis_trainer/saved_data/df_main.pkl')

ma.prepare_comment_df()
ma.predict_comment_sentiment('comment_analysis_trainer/saved_data/cv.pkl')
ma.rate_comment()
ma.write_rated_to_df_main()
#ma.plot_rating('Likes')
ma.plot_rating('Comments_Rating')
#ma.plot_rating('Rating_Total')
