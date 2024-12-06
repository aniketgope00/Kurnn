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
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import silhouette_score
import pickle
import joblib

def read_get_df():
    df = pd.read_csv("features_data.csv")
    df = df.drop(columns = ['Unnamed: 0'])
    return df

def safe_literal_eval(x):
  try:
    return ast.literal_eval(x)
  except (SyntaxError, ValueError):
    return x
  
def manual_pad_sequences(sequences, maxlen, dtype='int32', padding='post', truncating='post', value=0.):
    """
    Pads sequences to the same length.

    Args:
        sequences: A list of sequences.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
        padding: String, 'pre' or 'post', pad either before or after each sequence.
        truncating: String, 'pre' or 'post', truncate either before or after each sequence.
        value: Float or Int, value to pad the sequences with.

    Returns:
        x: Numpy array with shape (len(sequences), maxlen)
    """

    x = np.zeros((len(sequences), maxlen), dtype=dtype)
    for i, seq in enumerate(sequences):
        if truncating == 'pre':
            trunc = seq[-maxlen:]
        elif truncating == 'post':
            trunc = seq[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not recognized' % truncating)

        if padding == 'post':
            x[i, :len(trunc)] = trunc
        elif padding == 'pre':
            x[i, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not recognized' % padding)
    return x
  
def pad_df(df):
    df['mfcc_mean'] = df['mfcc_mean'].apply(safe_literal_eval)
    df['mfcc_std'] = df['mfcc_std'].apply(safe_literal_eval)

    max_len_mfcc_mean = max(len(x) for x in df['mfcc_mean'])
    max_len_mfcc_std = max(len(x) for x in df['mfcc_std'])
    print(f"max_len mfcc_mean: {max_len_mfcc_mean}")
    print(f"max_len mfcc_std: {max_len_mfcc_std}")

    # Manually pad sequences
    def pad_sequence(seq, max_len):
        """Pads a sequence with zeros to the specified maximum length."""
        return seq + [0.0] * (max_len - len(seq))

    df['mfcc_mean_padded'] = df['mfcc_mean'].apply(lambda x: manual_pad_sequences(x, max_len_mfcc_mean))
    df['mfcc_std_padded'] = df['mfcc_std'].apply(lambda x: manual_pad_sequences(x, max_len_mfcc_std))

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
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def get_labels(df):
    #df_padded = pad_df(df)
    #df_listed = convert_list_to_features(df_padded)
    df = df.drop(columns = ["mfcc_mean", "mfcc_std"])
    df_scaled = scale_df(df)
    model = joblib.load("kmeans.pkl")
    result = model.predict(df_scaled)
    return result