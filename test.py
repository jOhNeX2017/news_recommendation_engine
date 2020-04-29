import pandas as pd
from zipfile import ZipFile
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import os

zip_filename = 'news-category-dataset.zip'

with ZipFile(zip_filename, 'r') as zip:
    zip.extractall()

df = pd.read_json('News_Category_Dataset_v2.json',lines=True)
df.to_csv('data.csv')


data = pd.read_csv('data.csv')
data.rename( columns={'Unnamed: 0':'id'}, inplace=True )
#data = data[0:5000]

data = data.dropna(how='any')

data['headline'] = data['headline'].replace({"'ll":""},regex=True)
data['headline'] = data['headline'].replace({"-":""},regex=True)

data['short_description'] = data['short_description'].replace({"'ll":""},regex=True)
data['short_description'] = data['short_description'].replace({"-":""},regex=True)

comb_frame = data.headline.str.cat(" "+ data.short_description)

comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+":""},regex=True)



vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame)

true_k = len(list(set(data['category'])))

model = KMeans(n_clusters=true_k,init='k-means++', max_iter=500,n_init=15)
model.fit(X)

# print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names()

# for i in range(true_k):
#     print("Cluster %d"%i),
#     for ind in order_centroids[i,:15]:
#         print(' %s' %terms[ind]),
#     print

# print("centroids Length:",len(order_centroids))
# print(order_centroids)

# Y = vectorizer.transform(["Man Claims He Walked In On Co-Worker Engaging in Sexual Act With Patient In Hospital"])
# prediction = model.predict(Y)
# print(prediction)

import pickle

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

#removed extracted file becoz more than 25 mb we can't upload in git hub
os.remove('News_Category_Dataset_v2.json')
#removed the converted CSV file 
os.remove('data.csv')



with ZipFile('finalized_model.zip', 'w') as zipObj2:
   # Add files to the zip
   zipObj2.write('finalized_model.sav')

os.remove('finalized_model.sav')
