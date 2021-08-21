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


def open_image(img_stream, read_type):
    bytes = np.frombuffer(img_stream.read(), dtype=np.uint8)
    return cv2.imdecode(bytes, read_type)


def split_image(request_image):
    # Load image
    original_image = open_image(request_image, cv2.IMREAD_COLOR)
    gray_scale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Threshold image
    _, img_binary = cv2.threshold(
        gray_scale, 150, 225, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_binary = ~img_binary

    # Kernals for line detection
    line_min_width = 29
    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)

    # Extract Horizontal & Verticle Lines
    img_h_lines = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernal_h)
    img_v_lines = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernal_v)
    img_lines = img_h_lines | img_v_lines

    # Thicken Lines
    kernel_dilate = np.ones((3, 3), np.uint8)
    img_final_lines = cv2.dilate(img_lines, kernel_dilate, iterations=1)

    # Get the connected line components
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        ~img_final_lines, connectivity=8, ltype=cv2.CV_32S)

    # if 0, then there is no grid, must be normal number
    if len(stats[2:]) == 0:
        box_gray_scale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, box_image_thresh = cv2.threshold(
            box_gray_scale, 190, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated = cv2.dilate(box_image_thresh, kernel, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 1))
        dilated = cv2.dilate(dilated, kernel, iterations=1)

        directory_name = ".{}/1/".format(UPLOAD_FOLDER)
        os.mkdir(directory_name)

        # Get the contours for each individual elements
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        coordinates = []
        for contour in contours:
            [x, y, width, height] = cv2.boundingRect(contour)
            coordinates.append((x, y, width, height))

        coordinates.sort(key=lambda tup: tup[0])

        number = 0
        for current_coordinate in coordinates:
            number += 1
            [x, y, width, height] = current_coordinate
            # Use original image:
            segment = original_image[y:y+height, x:x+width]
            cv2.imwrite(os.path.join(directory_name,
                        "{}.png".format(number)), segment)
    else:
        # Go through each grid and get the numbers only (use a slightly higher thresh for this)
        for full_image_x, full_image_y, full_image_width, full_image_height, _ in stats[2:]:

            if full_image_height > 20:
                box_image = original_image[full_image_y:full_image_y +
                                           full_image_height, full_image_x:full_image_x+full_image_width]

                # Threshold at a lower amount to find the darker characters (most likely to contain a character)
                box_gray_scale = cv2.cvtColor(box_image, cv2.COLOR_BGR2GRAY)
                _, box_image_thresh = cv2.threshold(
                    box_gray_scale, 165, 255, cv2.THRESH_BINARY_INV)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
                dilated = cv2.dilate(box_image_thresh, kernel, iterations=1)

                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 1))
                dilated = cv2.dilate(dilated, kernel, iterations=1)

                # If not all black (might have a number inside)
                if cv2.countNonZero(dilated) != 0:

                    # Create Directory for grid
                    directory_name = ".{}/{}_{}_{}_{}/".format(
                        UPLOAD_FOLDER, full_image_y, full_image_x, full_image_width, full_image_height)
                    os.mkdir(directory_name)
                    cv2.imwrite(os.path.join(directory_name, "{}_{}_{}_{}.png".format(
                        full_image_y, full_image_x, full_image_width, full_image_height)), box_image)

                    # Get the contours for each individual elements
                    contours, _ = cv2.findContours(
                        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    coordinates = []
                    for contour in contours:
                        [x, y, width, height] = cv2.boundingRect(contour)
                        coordinates.append((x, y, width, height))

                    coordinates.sort(key=lambda tup: tup[0])

                    number = 0
                    for current_coordinate in coordinates:
                        number += 1
                        [x, y, width, height] = current_coordinate
                        # Use original image:
                        segment = box_image[y:y+height, x:x+width]
                        cv2.imwrite(os.path.join(directory_name,
                                    "{}.png".format(number)), segment)


def add_border(image):
    # Calculate padding to ensure size is same
    required_h_pxls = int((40 - image.shape[0]) / 2)
    required_w_pxls = int((40 - image.shape[1]) / 2)
    # Calculate adding required
    top = required_h_pxls
    bottom = int(40 - required_h_pxls - image.shape[0])
    left = required_w_pxls
    right = int(40 - required_w_pxls - image.shape[1])
    img = cv2.copyMakeBorder(image, top, bottom, left,
                             right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img


def load_training_data():
    data_path = "training_data/mnist/"
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

    # Load Custom Data
    custom_data_path = "training_data/further_data/"
    custom_data = np.loadtxt(custom_data_path + "train.csv", delimiter=",")

    # Split Data
    X_train = train_data[:, 1:]
    y_train = train_data[:, :1]
    X_test = test_data[:, 1:]
    y_test = test_data[:, :1]

    if len(custom_data) != 0:
        custom_X_train = custom_data[:, 1:]
        custom_y_train = custom_data[:, :1]
        X_train = np.concatenate((X_train, custom_X_train))
        y_train = np.concatenate((y_train, custom_y_train))

    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # one hot encode target values
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Store in np array
    train_imgs = np.asfarray(X_train).astype('float32')
    train_labels = np.asfarray(y_train)
    test_imgs = np.asfarray(X_test).astype('float32')
    test_labels = np.asfarray(y_test)
    return train_imgs, train_labels, test_imgs, test_labels


def save_custom_data(data_list):
    data_path = "training_data/further_data/train.csv"

    with open(data_path, 'a', newline='') as file:
        writer = csv.writer(file)

        for row in data_list:
            label = row['label']
            image_src = row['image'][1:]
            # Treat . as 10
            if label == ".":
                label = 10
            image = cv2.imread(image_src, 0)
            image = ~image

            bordered = add_border(image)

            resized_image = cv2.resize(bordered, (28, 28))
            row = list(resized_image.ravel())
            row.insert(0, label)
            writer.writerow(row)
        return True

    return False


def retrain_model():
    width = 28
    height = 28
    train_imgs, train_labels, test_imgs, test_labels = load_training_data()
    model = train_cnn(train_imgs, train_labels, test_imgs,
                      test_labels, width, height)
    filename = "training_data/pickle"
    model.save(filename)


def train_cnn(train_imgs, train_labels, test_imgs, test_labels, width, height):
    num_classes = train_labels.shape[1]
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(
        width, height, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # Fit
    model.fit(train_imgs, train_labels, validation_data=(
        test_imgs, test_labels), epochs=10, batch_size=200)
    return model


def predict_image():
    model = keras.models.load_model("training_data/pickle")
    images_dir = UPLOAD_FOLDER[1:]

    results = {} 
    directories = os.listdir(images_dir)
    for directory in directories:
        image_dir = images_dir+"/"+directory
        current_directory_images = os.listdir(image_dir)
        for image_name in current_directory_images:
            # Load Image
            image = cv2.imread(images_dir+"/"+directory+"/"+image_name)

            # Process the digit at a higher threshold
            box_gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, box_image_thresh = cv2.threshold(
                box_gray_scale, 190, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
            processed = cv2.dilate(box_image_thresh, kernel, iterations=1)

            if (processed.shape[0] <= 40 and processed.shape[1] <= 40) or (len(directories) == 1):
                if (len(directories) == 1):
                    img = processed
                else:
                    img = add_border(processed)

                resized = cv2.resize(img, (28, 28)).reshape(1, 28, 28, 1)
                predict_x = model.predict(resized)
                prediction = np.argmax(predict_x, axis=1)[0]

                # 10 is considered as .
                if prediction == 10:
                    prediction = "."

                if directory not in results:
                    results[directory] = {}

                image_name = image_name.split('.')[0]
                results[directory][int(image_name)] = prediction

    return results
