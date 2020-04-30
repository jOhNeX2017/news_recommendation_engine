from sqlalchemy import create_engine
import pandas as pd
from zipfile import ZipFile
import os



zip_filename = 'news-category-dataset.zip'

with ZipFile(zip_filename, 'r') as zip:
    zip.extractall()

df = pd.read_json('News_Category_Dataset_v2.json',lines=True)
df.to_csv('data.csv')

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root",pw="",db="recommend"), pool_size=10, max_overflow=20)

df = pd.read_csv('data.csv')
df.rename( columns={'Unnamed: 0':'id'}, inplace=True )
df.set_index(['id'], inplace=True)

df = df.dropna(how='any')

# print(len(df['category']))

# this will load the csv file in the sql
df.to_sql('data',con = engine, if_exists = 'append', chunksize = 100000)

os.remove('News_Category_Dataset_v2.json')
#removed the converted CSV file 
os.remove('data.csv')