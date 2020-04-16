import pandas as pd
import uuid
data =  pd.read_csv('dataset.csv')
data['id']='abc'
for i in range(0,len(data)):
    data['id'][i]=uuid.uuid4()

print(data.shape())

data.to_csv('dataset.csv')
