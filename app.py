from flask import Flask, redirect, url_for, send_from_directory, flash
from dotenv import find_dotenv, dotenv_values

# Load environment variables
config = dotenv_values(find_dotenv())

UPLOAD_FOLDER = '/static/temp_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SECRET_KEY'] = config["FLASK_SECRET_KEY"]
app.config['TESTING'] = config["FLASK_TESTING"]

# 20 Megabyes
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

if __name__ == "__main__":
    app.run()

# Load pages after app
import views

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('./static/images/', 'favicon.jpg')

# @app.errorhandler(Exception)
# def page_not_found(e):
#     return redirect(url_for('error_page', code=e.code))
