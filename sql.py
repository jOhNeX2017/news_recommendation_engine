from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root",pw="",db="recommend"), pool_size=10, max_overflow=20)

df = pd.read_csv('data.csv')
df.rename( columns={'Unnamed: 0':'id'}, inplace=True )
df.set_index(['id'], inplace=True)

df = df.dropna(how='any')

# print(len(df['category']))

# this will load the csv file in the sql
df.to_sql('data',con = engine, if_exists = 'append', chunksize = 100000)