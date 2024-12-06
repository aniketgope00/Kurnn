from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import flash
from db_checker_module import generate_rows
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, ValidationError
import supabase
from db_checker_module import generate_rows
import os

PROJECT_URL = "https://qbmoyulmzltkzvtqslnl.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFibW95dWxtemx0a3p2dHFzbG5sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzI0MzI3MDMsImV4cCI6MjA0ODAwODcwM30.Kg0APL06JN3Wa4Zd7J_uDM3nOoEclpcKYOA71QYN2n8"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

supabase_client = supabase.create_client(PROJECT_URL, API_KEY)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class Search_Form(FlaskForm):
    song_name = StringField('search song name')
    submit = SubmitField('search')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recommend_songs(filepath, num_recommendations):
    pass

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/generate")
def generate():
    return render_template("generate.html")


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

        return render_template('result.html', recommendations=recommendations)
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