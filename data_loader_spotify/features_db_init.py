import sqlite3

connector = sqlite3.connect("spotify_features_db.sqlite3")
cursor = connector.cursor()

query = """
CREATE TABLE FEATURES_DATA (
    feature_index FLOAT PRIMARY KEY,
    rms_level FLOAT,
    spectral_centroid FLOAT,
    bandwidth FLOAT,
    zero_crossing_rate FLOAT,
    band_energy_ratio FLOAT,
    delta_spectrum_magnitude FLOAT,
    pitch FLOAT,
    pitch_strength FLOAT,
    mfcc_mean FLOAT,
    mfcc_std FLOAT,
    average_roughness FLOAT,
    std_roughness FLOAT,
    one_2hz_loudness FLOAT,
    three_15hz_loudness FLOAT,
    twenty_43hz_loudness FLOAT,
    sharpness FLOAT,
    gamma_0Hz_energy FLOAT,
    gamma_3_15Hz_energy FLOAT,
    gamma_20_150Hz_energy FLOAT,
    gamma_150_1000Hz_energy FLOAT
)
"""
cursor.execute(query)
connector.close()