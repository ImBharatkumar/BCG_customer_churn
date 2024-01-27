#load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.utils import load_data


#loading data
file_path=os.path.join('src//data','clean_data_after_eda.csv')
data=load_data(file_path)


