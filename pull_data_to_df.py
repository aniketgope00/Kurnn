import os
import supabase
import pandas as pd
import batch_processor
import feature_extractor_module

PROJECT_URL = "https://qbmoyulmzltkzvtqslnl.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFibW95dWxtemx0a3p2dHFzbG5sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzI0MzI3MDMsImV4cCI6MjA0ODAwODcwM30.Kg0APL06JN3Wa4Zd7J_uDM3nOoEclpcKYOA71QYN2n8"

supabase_client = supabase.create_client(PROJECT_URL, API_KEY)
df = pd.DataFrame(columns=["track_name" ,"rms_level , spectral_centroid , bandwidth , zero_crossing_rate , band_energy_ratio , delta_spectrum_magnitude , pitch , pitch_strength , mfcc_mean , mfcc_std , average_roughness , std_roughness , one_2hz_loudness , three_15hz_loudness , twenty_43hz_loudness , sharpness , gamma_0Hz_energy , gamma_3_15Hz_energy , gamma_20_150Hz_energy , gamma_150_1000Hz_energy"])


def send_to_supabase(feature_dict):
    serialized_dict = batch_processor.make_json_serializable(feature_dict)
    supabase_client.table('audio_features').insert(serialized_dict).execute()


file_list = os.listdir("data_loader_spotify/Downloaded_Songs")
for file in file_list:
    path = "data_loader_spotify/Downloaded_Songs/"
    feature_dict = feature_extractor_module.extract_audio_features(path + file)
    feature_dict["track_name"] = file
    send_to_supabase(feature_dict)
    feature_df = pd.DataFrame([feature_dict])
    df = pd.concat([df, feature_df])

df = df.drop(columns=['rms_level , spectral_centroid , bandwidth , zero_crossing_rate , band_energy_ratio , delta_spectrum_magnitude , pitch , pitch_strength , mfcc_mean , mfcc_std , average_roughness , std_roughness , one_2hz_loudness , three_15hz_loudness , twenty_43hz_loudness , sharpness , gamma_0Hz_energy , gamma_3_15Hz_energy , gamma_20_150Hz_energy , gamma_150_1000Hz_energy'])
df.to_csv("features_data.csv")
    
