from tensorflow import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from app import ALLOWED_EXTENSIONS, UPLOAD_FOLDER
from shutil import rmtree
import numpy as np
import cv2
import csv
import os


def empty_directory(directory_name=UPLOAD_FOLDER):
    for file in os.listdir(directory_name):
        try:
            rmtree(os.path.join(directory_name, file))
        except Exception as e:
            return (False, e)
    return (True, "Success")


def allowed_file_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
