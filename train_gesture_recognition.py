import os
import csv
import numpy as np
import tensorflow as tf
from keras import Sequential, layers, activations, optimizers, losses, metrics
from matplotlib import pyplot as plt

from face_landmarks_subset import gesture_landmark_indexes
from utils import gesture2class_dict as class_dict

LANDMARKS_DATASET_PATH = 'results/landmark_data/'
MODEL_SAVE_PATH = 'models/face_gesture_recognition/'
TEST_RATIO = 0.03
VALIDATION_RATIO = 0.06


def plot_history(history):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Categorical accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def load_landmarks_dataset():
    dataset = []
    labels = []
    classes = os.listdir(LANDMARKS_DATASET_PATH)
    for c in classes:
        tensor_files = os.listdir(os.path.join(LANDMARKS_DATASET_PATH, c))
        for tf in tensor_files:
            data = []
            label_vector = [0]*7
            label_vector[class_dict[c]] = 1
            with open(os.path.join(LANDMARKS_DATASET_PATH, c, tf)) as t:
                for r in csv.reader(t): data.append([float(v) for v in r])
            data = [data[i] for i in gesture_landmark_indexes]
            dataset.append(data)
            labels.append(label_vector)
    dataset = np.array(dataset, dtype=np.float32)
    labels = np.array(labels, dtype=np.int8)
    return [dataset, labels]


inputs, labels = load_landmarks_dataset()
dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

ds_size = dataset.cardinality().numpy()
train_ds_size = ds_size*(1-TEST_RATIO)
validation_ds_size = train_ds_size*VALIDATION_RATIO
test_ds_size = ds_size*TEST_RATIO

dataset = dataset.shuffle(buffer_size=ds_size, reshuffle_each_iteration=False)
train_ds = dataset.take(train_ds_size)
validation_ds = train_ds.take(validation_ds_size)
test_ds = dataset.skip(train_ds_size)

model = Sequential([
    layers.Input((222, 3)),
    layers.Conv1D(1, 30, activation=activations.relu),
    layers.Flatten(),
    layers.Dense(units=1200, activation=activations.relu),
    layers.Dense(units=7, activation=activations.softmax),
], name='face_gesture_recognition')

model.compile(optimizer=optimizers.Adam(), loss=losses.categorical_crossentropy, metrics=[metrics.CategoricalAccuracy])
model.summary()

history = model.fit(train_ds.shuffle(train_ds_size).batch(40), epochs=60, validation_data=validation_ds.batch(test_ds_size))
plot_history(history)
model.evaluate(test_ds.batch(test_ds_size))
model.export(MODEL_SAVE_PATH)


