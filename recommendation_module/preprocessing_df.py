import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import silhouette_score
import pickle

def read_get_df():
    df = pd.read_csv("features_data.csv")
    df = df.drop(columns = ['Unnamed: 0'])
    return df

def safe_literal_eval(x):
  try:
    return ast.literal_eval(x)
  except (SyntaxError, ValueError):
    return x
  
def pad_df(df):
    df['mfcc_mean'] = df['mfcc_mean'].apply(safe_literal_eval)
    df['mfcc_std'] = df['mfcc_std'].apply(safe_literal_eval)

    max_len_mfcc_mean = max(len(x) for x in df['mfcc_mean'])
    max_len_mfcc_std = max(len(x) for x in df['mfcc_std'])

    # Manually pad sequences
    def pad_sequence(seq, max_len):
        """Pads a sequence with zeros to the specified maximum length."""
        return seq + [0.0] * (max_len - len(seq))

    df['mfcc_mean_padded'] = df['mfcc_mean'].apply(lambda x: pad_sequence(x, max_len_mfcc_mean))
    df['mfcc_std_padded'] = df['mfcc_std'].apply(lambda x: pad_sequence(x, max_len_mfcc_std))

    return df

def list_to_features(df, column_name, prefix):
    """Converts a list column to multiple feature columns."""
    max_len = max(len(x) for x in df[column_name])
    for i in range(max_len):
        df[f'{prefix}_{i}'] = df[column_name].apply(lambda x: x[i] if i < len(x) else 0)
    return df

def convert_list_to_features(df):
    df = list_to_features(df, 'mfcc_mean_padded', 'mfcc_mean')
    df = list_to_features(df, 'mfcc_std_padded', 'mfcc_std')

    # Drop original list columns
    df = df.drop(columns=['mfcc_mean', 'mfcc_std', 'mfcc_mean_padded', 'mfcc_std_padded'])
    return df

def scale_df(df):
    df = df.drop(columns = ["track_name"])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def get_labels(df):
    model = pickle.load(open('kmean_model.pkl', 'rb'))
    result = model.predict(df)
    return result