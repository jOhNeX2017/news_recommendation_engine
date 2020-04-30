from flask import Flask,request,render_template,redirect,url_for,session,jsonify
from flaskext.mysql import MySQL
from datetime import datetime
import os,time
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from zipfile import ZipFile
import os

############################ Recommedation Part #################################################

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

    temp_data = temp_data.sample(15)
    return temp_data

##########################################  WebSite Part  #################################

mysql=MySQL()
app = Flask(__name__)
app.secret_key = 'recommend'

app.config['MYSQL_DATABASE_USER'] = 'root' 
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'recommend'
app.config['MYSQL_DATABASE_HOST'] = '127.0.0.1'

mysql.init_app(app)
conn = mysql.connect()
cur = conn.cursor()

news_categories = ['SCIENCE', 'LATINO VOICES', 'MEDIA', 'RELIGION', 'THE WORLDPOST', 'WORLDPOST', 'ARTS & CULTURE', 'HEALTHY LIVING', 'PARENTS', 'WORLD NEWS', 'BLACK VOICES', 'SPORTS', 'TASTE', 'ENTERTAINMENT', 'COLLEGE', 'IMPACT', 'BUSINESS', 'STYLE', 'QUEER VOICES', 'FIFTY', 'WEIRD NEWS', 'CRIME', 'GOOD NEWS', 
'GREEN', 'TRAVEL', 'WOMEN', 'ARTS', 'EDUCATION', 'COMEDY', 'TECH', 'POLITICS']



@app.route('/', methods=['GET','POST'])
def index():
    cur.execute("Select * from data order by date LIMIT 39")
    data = cur.fetchall()

    # print(data[0])
    return render_template('index.html',data=data,data_length=len(data),category=news_categories)

@app.route("/category/<variable>")
def user(variable):
    cur.execute("Select * from data  where category = %s order by date LIMIT 39;",(variable))
    data = cur.fetchall()
    return render_template('category.html',data=data,data_length=len(data),category=news_categories)

@app.route("/recommend/<var_id>")
def recommend(var_id):
    cur.execute("Select * from data where id = {} ".format(var_id))
    news_data=cur.fetchone()
    var = news_data[2]
    recommend_data = recommend_util(var)['id']
    #print(recommend_data)
    ids = (list(set(recommend_data)))
    print(ids)
    data = []
    for id in ids:
        cur.execute("Select * from data where id = {} ".format(id))
        data.append(cur.fetchone())
        #print(data)
    return render_template('recommend.html',data=data,data_length=len(data),category=news_categories,news=news_data)


if __name__=="__main__":
	app.run(debug=True)

#removed extracted file becoz more than 25 mb we can't upload in git hub
os.remove('News_Category_Dataset_v2.json')
#removed the converted CSV file 
os.remove('data.csv')
#removed the extracted fianlized_model
os.remove('finalized_model.sav')