import pandas as pd
import numpy as np

dataset = pd.read_csv('2019VAERSData.csv', encoding='latin-1')

dataset["SERIOUS"] = np.where((dataset["DIED"] == "Y" )
    | (dataset["ER_VISIT"] == "Y") | (dataset["HOSPITAL"] == "Y") 
    | (dataset["DISABLE"] == "Y"), 'Y', 'N')


print(dataset.head())