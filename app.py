from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import flash
from flask import url_for
from db_checker_module import generate_rows
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, ValidationError
import supabase
from db_checker_module import generate_rows
from recommendation_module import feature_extractor_module
from recommendation_module import preprocessing_df
import numpy as np
import soundfile as sf
from llm_module import models
import os
import joblib
import pandas as pd

PROJECT_URL = "https://qbmoyulmzltkzvtqslnl.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFibW95dWxtemx0a3p2dHFzbG5sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzI0MzI3MDMsImV4cCI6MjA0ODAwODcwM30.Kg0APL06JN3Wa4Zd7J_uDM3nOoEclpcKYOA71QYN2n8"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

supabase_client = supabase.create_client(PROJECT_URL, API_KEY)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


df_songs = pd.read_csv("recommendation_module/features_data.csv")

class Search_Form(FlaskForm):
    song_name = StringField('search song name')
    submit = SubmitField('search')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recommend_songs(filepath, num_recommendations):
    get_features = feature_extractor_module.extract_audio_features(file_path=filepath)
    df = pd.DataFrame([get_features])
    df = df.drop(columns=["mfcc_mean", "mfcc_std"])
    model = joblib.load("recommendation_module/kmeans_model_final.pkl")
    result = model.predict(df)[0]
    df_org = pd.read_csv("recommendation_module/clustered_data.csv")
    recommendations = list(df_org[df_org["cluster_kmeans"] == result]["Unnamed: 0"][:num_recommendations])
    songs = []
    for rec in recommendations:
        songs.append(df_songs["track_name"].iloc[rec])
    files_list = os.listdir("uploads")
    for file in files_list:
        os.remove("uploads/"+file)
    os.rmdir("uploads")
    return songs

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/generate")
def generate():
    return render_template("generate.html")

@app.route("/generate_result", methods = ["GET","POST"])
def generate_results():
    if "file" not in request.files and "prompt" not in request.form:
        flash("File and prompt are required.")
        return redirect(request.url)

    # File handling (optional)
    file = request.files["file"]
    lyrics, generated_lyrics = "", ""
    if file.filename:
        file.save(os.path.join(OUTPUT_DIR, file.filename))
        lyrics = models.transcriber(OUTPUT_DIR+"/"+file.filename)
        generated_lyrics = models.get_new_lyrics(lyrics)

    
    # Get form data
    prompt = request.form["prompt"]
    model_choice = request.form.get("model_choice")
    checkbox_field = request.form.get("checkbox_field")

    # Model processing
    if model_choice == "model1":
        notes_generated = models.text_to_music(prompt)
    elif model_choice == "model2":
        audio_data_list = models.musicgen(prompt)
        audio_data = audio_data_list[0][0].numpy().astype(np.float32)
        audio_data = audio_data.squeeze()  # Remove unnecessary dimensions if present
        sampling_rate = audio_data_list[1]
        # Save as WAV file
        sf.write(OUTPUT_DIR+'/'+'generated_audio.wav', audio_data, sampling_rate)
    else:
        flash("Please select a model for generation.")
        return redirect(request.url)

    # Render results
    return render_template("result.html", text_output=notes_generated, audio_output=OUTPUT_DIR+'/'+'generated_audio.wav', lyrics = lyrics, song_data = generated_lyrics[:2], new_lyrics = generated_lyrics[-1])


@app.route("/recommend_home")
def recommend():
    return render_template("recommend.html")

@app.route("/recommend_results", methods = ["POST"])
def recommend_results():
    if 'file' not in request.files or 'num_recommendations' not in request.form:
        flash('Both file and number of recommendations are required')
        return redirect(request.url)
    
    file = request.files['file']
    num_recommendations = request.form.get('num_recommendations')

    # Validate number of recommendations
    if not num_recommendations.isdigit() or int(num_recommendations) <= 0:
        flash('Number of recommendations must be a positive integer')
        return redirect(request.url)
    
    num_recommendations = int(num_recommendations)

    # Validate and save the file
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Generate recommendations
        recommendations = recommend_songs(filepath, num_recommendations)

        return render_template('results.html', recommendations=recommendations)
    else:
        flash('Invalid file type. Only .wav and .mp3 files are allowed.')
        return redirect(request.url)

@app.route("/check", methods=["GET", "POST"])
def check():
    header_rows = generate_rows.get_header_rows()
    search_input = Search_Form()

    if request.method == "POST":
        search_string = search_input.song_name.data
        search_result_songs = generate_rows.search_results(search_string)
        return render_template('search_results.html', search_results=search_result_songs)  # Updated variable name

    return render_template("check.html", rows=header_rows, form=search_input)

if __name__ == "__main__":
    app.run(debug=True)