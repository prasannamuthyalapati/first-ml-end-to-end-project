
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import cross_val_score 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
df=pd.read_csv(r'ml.project\files\application_record.csv')
print(df.head())