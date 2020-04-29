import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from zipfile import ZipFile
import os

zip_filename='finalized_model.zip'

with ZipFile(zip_filename, 'r') as zip:
    zip.extractall()

with open('finalized_model.sav','rb') as fid:
    model = pickle.load(fid)

zip_filename = 'news-category-dataset.zip'

with ZipFile(zip_filename, 'r') as zip:
    zip.extractall()

df = pd.read_json('News_Category_Dataset_v2.json',lines=True)
df.to_csv('data.csv')

data = pd.read_csv('data.csv')
data.rename( columns={'Unnamed: 0':'id'}, inplace=True )
data = data.dropna(how='any')

data['headline'] = data['headline'].replace({"'ll":""},regex=True)
data['headline'] = data['headline'].replace({"-":""},regex=True)

data['short_description'] = data['short_description'].replace({"'ll":""},regex=True)
data['short_description'] = data['short_description'].replace({"-":""},regex=True)

comb_frame = data.headline.str.cat(" "+ data.short_description)

comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+":""},regex=True)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame)

data = pd.read_csv('data.csv')
data.rename( columns={'Unnamed: 0':'id'}, inplace=True )
data = data.dropna(how='any')

data['inp_string']=data.headline.str.cat(" "+ data.short_description)

data['clusterprediction'] = ""

def cluster_predict(str_input):
    Y = vectorizer.transform(list(str_input))
    prediction = model.predict(Y)
    return prediction

data['clusterprediction']=data.apply(lambda x: cluster_predict(data['inp_string']),axis=0)

# print(type(data['clusterprediction'][0]))

str_input = "There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV"

def recommend_util(str_input):
    temp_data = data.loc[data.headline==str_input]
    predection_inp = cluster_predict(str_input)
    predection_inp = int(predection_inp[0] if len(predection_inp)>1 else predection_inp)

    temp_data=data.loc[data.clusterprediction == predection_inp]

    temp_data = temp_data.sample(10)
    return temp_data

# def recommend_util(str_input):
#     temp_data = data.loc[data['headline'] == str_input]
#     temp_data['inp_string']= temp_data.headline.str.cat(" "+ temp_data.short_description)
#     str_input = temp_data['inp_string']

#     predction_inp = cluster_predict(str_input)
#     predction_inp = int(predction_inp)

#     temp_data =  data.loc[data['clusterprediction'] == predction_inp]
#     temp_data = temp_data.sample(10)
    
#     return list(temp_data)

res = recommend_util("There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV")
print(res[['headline','short_description']])


#removed extracted file becoz more than 25 mb we can't upload in git hub
os.remove('News_Category_Dataset_v2.json')
#removed the converted CSV file 
os.remove('data.csv')
os.remove('finalized_model.sav')