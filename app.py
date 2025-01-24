from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import flash
from flask import url_for

import psycopg2
from dotenv import load_dotenv

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
import models
import os
import joblib
import pandas as pd

import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth

#Fetch the data from the database
load_dotenv()

PROJECT_URL = os.getenv('PROJECT_URL')
API_KEY = os.getenv('API_KEY')
USER = os.getenv('USER')
PASSWORD = os.getenv('PASSWORD')
HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
DBNAME = os.getenv('DBNAME')

#cred = credentials.Certificate('path/to/your/credentials.json')
#firebase_admin.initialize_app(cred)

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

class LoginForm(FlaskForm):
    email = StringField('email', validators=[DataRequired()])
    password = StringField('password', validators=[DataRequired()])
    submit = SubmitField('login')

class Search_Form(FlaskForm):
    song_name = StringField('search song name')
    submit = SubmitField('search')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recommend_songs(filepath, num_recommendations):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
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
    return songs

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/login_result', methods=['GET', 'POST'])
def validate_login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        '''try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user_uid'] = user.uid
            return redirect(url_for('dashboard'))  # Redirect to a protected page
        except Exception as e:
            # Handle login errors (e.g., invalid credentials)
            return 
'''
@app.route('/checkdb')
def checkdb():
    try:
        connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
        return "connection successful"
    except Exception as error:
        return f"Error connecting to the database: {error}"

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return validate_login()
    return render_template("login.html")

@app.route("/generate")
def generate():
    return render_template("generate.html")

@app.route("/generate_result", methods = ["GET","POST"])
def generate_results():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # Get form data
    #prompt = request.form["prompt"]
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    filepath=""
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    print(filepath)

    lyrics = models.transcriber('uploads/'+os.listdir('uploads')[0])
    song_data = models.get_new_lyrics(lyrics)
    text_output = models.text_to_music(song_data[0], song_data[1])
    
    files_list = os.listdir("uploads")
    for file in files_list:
        os.remove("uploads/"+file)

    # Render results
    return render_template("generation_results.html", text_output=text_output, lyrics = lyrics, song_data = song_data[:2], new_lyrics = song_data[-1])


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


@app.route("/explore")
def explore():
    return render_template("explore.html")

@app.route('/discover')
def contribute():
    header_rows = generate_rows.get_header_rows()
    print(header_rows)
    return render_template("discover.html", header_rows = header_rows)

if __name__ == "__main__":
    app.run(debug=True)