from utlis import *
from app import *
from forms import *
from flask import render_template, request, session

@app.route('/', methods=['GET'])
def home():
    form = upload_form()
    return render_template('home.html', page_title="Homepage", form=form)


@app.route('/preprocess', methods=['POST'])
def preprocess():
    return redirect(url_for("results"))


@app.route('/results', methods=['GET'])
def results():
    ml_results = []
    return render_template('results.html', page_title="Results", results=ml_results)

@app.route('/retrain', methods=['GET'])
def retrain():
    return redirect(url_for('home'))


@app.route('/newdata', methods=["POST"])
    return {"msg":"failed"}

@app.route('/error/<int:code>', methods=['GET'])
def error_page(code):
    ERROR_MESSAGES = {
        404: "Page not found!",
        413: "Image size is too big! Please try something smaller!",
        500: "Unexpected error occurred, please try again later!",
    }
    if code not in ERROR_MESSAGES:
        code = 500
    return render_template('error.html', page_title="Error", context=ERROR_MESSAGES[code])
