from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField

class upload_form(FlaskForm):
    image_upload = FileField()
    update_btn = SubmitField('Upload')
