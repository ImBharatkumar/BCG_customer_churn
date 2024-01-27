import pandas as pd
import os


#function to load data
def load_data(path):
    data= pd.read_csv(path)
    return data

# function to save df to csv file
import os

def save_to_csv(df, path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except:
            print("Error: unable to create directory for csv")
    df.to_csv(path, index=False)