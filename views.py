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
    # Check an image has been uploaded
    if ('image_upload' not in request.files) or (request.files['image_upload'].filename == ''):
        flash("No Image uploaded", "danger")
        return redirect(url_for('home'))

    file = request.files['image_upload']
    # Check extension is allowed
    if allowed_file_extension(file.filename) is False:
        message = f"Invalid file format used, only {list(ALLOWED_EXTENSIONS)} extensions are supported."
        flash(message, "danger")
        return redirect(url_for('home'))

    # Clear everything from temp_images (previous session)
    is_successful, message = empty_directory(directory_name="static/temp_images")
    if is_successful != True:
        flash(message, "danger")
        return redirect(url_for('home'))

    # Try load image
    try:
        image = request.files['image_upload']
    except Exception as message:
        flash(message, "danger")
        return redirect(url_for('home'))

    session['Filename'] = file.filename.split(".")[0]

    # Try split grid and store in temp images
    split_image(image)
    return redirect(url_for("results"))


@app.route('/results', methods=['GET'])
def results():
    ml_results = predict_image()
    return render_template('results.html', page_title="Results", results=ml_results)

@app.route('/retrain', methods=['GET'])
def retrain():
    try:
        retrain_model()
        flash("Retrain successful!", "success")
    except Exception as message:
        flash(message, "danger")
    return redirect(url_for('home'))


@app.route('/newdata', methods=["POST"])
def newdata():
    if request.method == "POST":
        try:
            data = request.get_json()['new_data']
            save_custom_data(data)
            flash("Data Added!", "success")
            return {"msg":"successfully"}
        except Exception as message:
            flash(message, "danger")
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
