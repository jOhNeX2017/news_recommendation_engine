import pandas as pd
import numpy as np
from zipfile import ZipFile 
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from scipy.sparse import  csr_matrix
from sklearn.neighbors import NearestNeighbors

zip_filename = 'news-category-dataset.zip'

with ZipFile(zip_filename, 'r') as zip:
    zip.extractall()

df = pd.read_json('News_Category_Dataset_v2.json',lines=True)
df.to_csv('data.csv')

data = pd.read_csv('data.csv')
data.rename( columns={'Unnamed: 0':'id'}, inplace=True )
data=data[0:5000]

total_no_of_category = len(list(set(data['category'])))

# combine_news_category = data.dropna(axis=0, subset=['headline'])
# news_rating_count = combine_news_category.groupby(by=['headline'])['category'].count().reset_index()[['headline','category']]

# news_category_count =  news_rating_count.rename(columns={'category':'totalRatingCount'})

# #print(news_category_count.head())

# rating_with_totalRatingCount = pd.merge(combine_news_category,news_rating_count,left_on="headline",right_on="headline",how="left")
# #print(rating_with_totalRatingCount.head())

# pd.set_option('display.float_format', lambda x: '%3f' %x)
# #print(news_rating_count['category'].describe())

# popularity_threshold = 2
# rating_popular_news = rating_with_totalRatingCount.query('category_y >= @popularity_threshold')

# news_features_df = rating_popular_news.pivot_table(index='headline',columns='id',values='category_y').fillna(0)
# #print(news_features_df.head())

# news_features_df_matrix = csr_matrix(news_features_df.values)

# knn_model = NearestNeighbors(metric='cosine', algorithm = 'brute')
# knn_model.fit(news_features_df_matrix)

# query_index=np.random.choice(news_features_df.shape[0])
# distance,indices = knn_model.kneighbors(news_features_df.iloc[query_index,:].values.reshape(1,-1),n_neighbors=total_no_of_category)

# for i in range(0,len(distance.flatten())):
#     if i==0:
#         print('Recoomendation for {0}:\n'.format(news_features_df.index[query_index]))
#     else :
#         print('{0}: {1}, with disatnce of {2}'.format(i,news_features_df.index[indices.flatten()[i]],distance.flatten()[i]))

# dropping irrelevant columns
data_cleaned = data.drop(columns=['category','authors','link'])

# applying tfidfvecorizer and removing the stop words in order to create document matrix
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern='\w{1,}',ngram_range=(1,3),stop_words='english')

# filling NaN with blank
data_cleaned['headline'] = data_cleaned['headline'].fillna('')

# applying fit transform
tfv_matrix =  tfv.fit_transform(data_cleaned['headline'])

# computing the sigomoid value using sigmoid matrix
sig = sigmoid_kernel(tfv_matrix,tfv_matrix)

indices = pd.Series(data_cleaned.index,index=data_cleaned['short_description']).drop_duplicates()

def give_recommendation(headline, sig=sig):
    idx = indices[headline]

    sig_scores = list(enumerate(sig[idx]))

    sig_scores = sorted(sig_scores,key=lambda x: x[1], reverse=True)

    sig_scores=sig_scores[1:11]

    news_indices = [i[0] for i in sig_scores]

    return data_cleaned['headline'].iloc[news_indices]

print(give_recommendation('There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV'))



#removed extracted file becoz more than 25 mb we can't upload in git hub
os.remove('News_Category_Dataset_v2.json')
#removed the converted CSV file 
os.remove('data.csv')