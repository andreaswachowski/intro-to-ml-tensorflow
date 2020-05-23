# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="k7sePAkWpLJV"
# # Saving and Loading Models
#
# In this notebook, we'll see how to save and load models with TensorFlow. This is important because you'll often want to load previously trained models to use in making predictions or to continue training on new data.

# %% [markdown] colab_type="text" id="tD856SqhH4JK"
# ## Import Resources

# %%
import warnings
warnings.filterwarnings('ignore')

# %% colab={} colab_type="code" id="Hsu5egUUqPg9"
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# %%
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="BqsrWYDKp4Fd" outputId="5fe90392-c56f-423f-bc89-b9fc985feecf"
print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

# %% [markdown] colab_type="text" id="dAe81nXoICzC"
# ## Load the Dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 207} colab_type="code" id="bxcg_ZbuLnM3" outputId="33841a52-53e6-4e8a-ecbd-b448bf3c46f5"
train_split = 60
test_val_split = 20

splits = tfds.Split.ALL.subsplit([train_split, test_val_split, test_val_split])

dataset, dataset_info = tfds.load('fashion_mnist', split=splits, as_supervised=True, with_info=True)

training_set, validation_set, test_set = dataset

# %% [markdown] colab_type="text" id="z1WhOLC7Ii3D"
# ## Explore the Dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" id="9i2586KjI4QM" outputId="24ccc17a-c3f6-44ba-edc3-ae267e628fc2"
total_examples = dataset_info.splits['train'].num_examples + dataset_info.splits['test'].num_examples

num_training_examples = (total_examples * train_split) // 100
num_validation_examples = (total_examples * test_val_split) // 100
num_test_examples = num_validation_examples

print('There are {:,} images in the training set'.format(num_training_examples))
print('There are {:,} images in the validation set'.format(num_validation_examples))
print('There are {:,} images in the test set'.format(num_test_examples))

# %% colab={} colab_type="code" id="RLMJCpppq43U"
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

# %% colab={"base_uri": "https://localhost:8080/", "height": 280} colab_type="code" id="PeU9nb_xqW98" outputId="7e35ce36-2589-4b3e-b2bf-313eaa2414f0"
for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

plt.imshow(image, cmap=plt.cm.binary)
plt.title(class_names[label])
plt.colorbar()
plt.show()


# %% [markdown] colab_type="text" id="k5rUDqxBIt5N"
# ## Create Pipeline

# %% colab={} colab_type="code" id="Ec3uphcyci3c"
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 64

training_batches = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)
validation_batches = validation_set.cache().batch(batch_size).map(normalize).prefetch(1)
testing_batches = test_set.cache().batch(batch_size).map(normalize).prefetch(1)

# %% [markdown] colab_type="text" id="ySQuJ-iPqNoR"
# ## Build and Train the Model
#
# Here we'll build and compile our model as usual.

# %% colab={"base_uri": "https://localhost:8080/", "height": 425} colab_type="code" id="47Vnu0KJMqwc" outputId="f1abd9d1-db9e-4bfd-99c7-d4376adb8745"
layer_neurons = [512, 256, 128]

dropout_rate = 0.5

model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))

for neurons in layer_neurons:
    model.add(tf.keras.layers.Dense(neurons, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))

model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.summary()

# %% colab={"base_uri": "https://localhost:8080/", "height": 153} colab_type="code" id="1qLJ-cAwnmFD" outputId="32be0a8a-5cfb-473f-872c-e09da279eae5"
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 4

history = model.fit(training_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

# %% [markdown] colab_type="text" id="jseIvfe2xb56"
# ## Saving and Loading Models
#
# In TensorFlow we can save our trained models in different formats. Here we will see how to save our models in TensorFlow's SavedModel format and as HDF5 files, which is the format used by Keras models.
#
# ### Saving and Loading Models in HDF5 Format
#
# To save our models in the format used by Keras models we use the `.save(filepath)` method. For example, to save a model called `my_model` in the current working directory with the name `test_model` we use:
#
# ```python
# my_model.save('./test_model.h5')
# ```
#
# It's important to note that we have to provide the `.h5` extension to the `filepath` in order the tell `tf.keras` to save our model as an HDF5 file. 
#
# The above command saves our model into a single HDF5 file that will contain:
#
# * The model's architecture.
# * The model's weight values which were learned during training.
# * The model's training configuration, which corresponds to the parameters you passed to the `compile` method.
# * The optimizer and its state. This allows you to resume training exactly where you left off.
#
#
# In the cell below we save our trained `model` as an HDF5 file. The name of our HDF5 will correspond to the current time stamp. This is useful if you are saving many models and want each of them to have a unique name. By default the `.save()` method will **silently** overwrite any existing file at the target location with the same name. If we want `tf.keras` to provide us with a manual prompt to whether overwrite files with the same name, you can set the argument `overwrite=False` in the `.save()` method.

# %% colab={} colab_type="code" id="G1dOvNRvrhNa"
t = time.time()

saved_keras_model_filepath = './{}.h5'.format(int(t))

model.save(saved_keras_model_filepath)

# %% [markdown] colab_type="text" id="lGNRBb1puSRg"
# Once a model has been saved, we can use `tf.keras.models.load_model(filepath)` to re-load our model. This command will also compile our model automatically using the saved training configuration, unless the model was never compiled in the first place.

# %% colab={"base_uri": "https://localhost:8080/", "height": 547} colab_type="code" id="akaAVE2js5d0" outputId="84301998-a6c3-4a55-c5f1-f76d086290bf"
reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath)

reloaded_keras_model.summary()

# %% [markdown] colab_type="text" id="xWihP1oMjNeF"
# As we can see the re-loaded model has the same architecture as our original model, as it should be. At this point, since we haven't done anything new to the re-loaded model, then both the `reloaded_keras_model` our original `model` should be identical copies. Therefore, they should make the same predictions on the same images. Let's check that this is true:

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="gLQsw7QVkElc" outputId="0d00e16b-9fdd-4d34-b9ab-96956ddbf5a5"
for image_batch, label_batch in testing_batches.take(1):
    prediction_1 = model.predict(image_batch)
    prediction_2 = reloaded_keras_model.predict(image_batch)
    difference = np.abs(prediction_1 - prediction_2)
    print(difference.max())

# %% [markdown] colab_type="text" id="K-dDOY0BmYhs"
# As we can see, the result is 0.0, which indicates that both models made the same predictions on the same images.

# %% [markdown] colab_type="text" id="lxHdz18pQUNV"
# ### Saving and Loading TensorFlow SavedModels

# %% [markdown] colab_type="text" id="OGtK83g2vVki"
# To export our models to the TensorFlow **SavedModel** format, we use the `tf.saved_model.save(model, export_dir)` function. For example, to save a model called `my_model` in a folder called `saved_models` located in the current working directory we use:
#
# ```python
# tf.saved_model.save(my_model, './saved_models')
# ```
#
# It's important to note that here we have to provide the path to the directory where we want to save our model, **NOT** the name of the file. This is because SavedModels are not saved in a single file. Rather, when you save your model as a SavedModel, `the tf.saved_model.save()` function will create an `assets` folder, a `variables` folder, and a `saved_model.pb` file inside the directory you provided.
#
# The SavedModel files that are created contain:
#
# * A TensorFlow checkpoint containing the model weights.
# * A SavedModel proto containing the underlying TensorFlow graph. Separate graphs are saved for prediction (serving), training, and evaluation. If the model wasn't compiled before, then only the inference graph gets exported.
# * The model's architecture configuration if available.
#
# The SavedModel is a standalone serialization format for TensorFlow objects, supported by TensorFlow serving as well as TensorFlow implementations other than Python. It does not require the original model building code to run, which makes it useful for sharing or deploying in different platforms, such as mobile and embedded devices (with TensorFlow Lite), servers (with TensorFlow Serving), and even web browsers (with TensorFlow.js).
#
# In the cell below we save our trained model as a SavedModel. The name of the folder where we are going to save our model will correspond to the current time stamp. Again, this is useful if you are saving many models and want each of them to be saved in a unique directory.

# %% colab={"base_uri": "https://localhost:8080/", "height": 173} colab_type="code" id="V2C0F3luxzlI" outputId="80a362e5-008f-4f54-ffb5-0b5577b5c46b"
t = time.time()

savedModel_directory = './{}'.format(int(t))

tf.saved_model.save(model, savedModel_directory)

# %% [markdown] colab_type="text" id="DBY1j0QEyjPi"
# Once a model has been saved as a SavedModel, we can use `tf.saved_model.load(export_dir)` to re-load our model. 

# %% colab={} colab_type="code" id="rRx2y2M4AtKl"
reloaded_SavedModel = tf.saved_model.load(savedModel_directory)

# %% [markdown] colab_type="text" id="wJwmzT1gAwew"
# It's important to note that the object returned by `tf.saved_model.load` is **NOT** a Keras object. Therefore, it doesn't have `.fit`, `.predict`, `.summary`, etc. methods. It is 100% independent of the code that created it. This means that in order to make predictions with our `reloaded_SavedModel` we need to use a different method than the one used with the re-loaded Keras model.
#
# To make predictions on a batch of images with a re-loaded SavedModel we have to use:
#
# ```python
# reloaded_SavedModel(image_batch, training=False)
# ```
#
# This will return a tensor with the predicted label probabilities for each image in the batch. Again, since we haven't done anything new to this re-loaded SavedModel, then both the `reloaded_SavedModel` and our original `model` should be identical copies. Therefore, they should make the same predictions on the same images. Let's check that this is true:

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="ozMqD1ZoER5g" outputId="17769afa-1a1f-48c4-80e5-a389c80f4062"
for image_batch, label_batch in testing_batches.take(1):
    prediction_1 = model.predict(image_batch)
    prediction_2 = reloaded_SavedModel(image_batch, training=False).numpy()
    difference = np.abs(prediction_1 - prediction_2)
    print(difference.max())

# %% [markdown] colab_type="text" id="3QZNNPkYFH3D"
# We can also get back a full Keras model, from a TensorFlow SavedModel, by loading our SavedModel with the `tf.keras.models.load_model` function. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 425} colab_type="code" id="0BxFJcGLyMTD" outputId="a2fefa76-57b5-4a9e-8c05-b8a1ae7f31a2"
reloaded_keras_model_from_SavedModel = tf.keras.models.load_model(savedModel_directory)

reloaded_keras_model_from_SavedModel.summary()

# %% [markdown] colab_type="text" id="FomAlrxnQnm8"
# ## Saving Models During Training
#
# We have seen that when we train a model with a validation set, the value of the validation loss changes through the training process. Since the value of the validation loss is an indicator of how well our model will generalize to new data, it will be great if could save our model at each step of the training process and then only keep the version with the lowest validation loss. 
#
# We can do this in `tf.keras` by using the following callback:
#
# ```python
# tf.keras.callbacks.ModelCheckpoint('./best_model.h5', monitor='val_loss', save_best_only=True)
# ```
# This callback will save the model as a Keras HDF5 file after every epoch. With the `save_best_only=True` argument, this callback will first check the validation loss of the latest model against the one previously saved. The callback will only save the latest model and overwrite the old one, if the latest model has a lower validation loss than the one previously saved. This will guarantee that will end up with the version of the model that achieved the lowest validation loss during training.

# %% colab={"base_uri": "https://localhost:8080/", "height": 765} colab_type="code" id="vvsuAeUQ1WKR" outputId="b8ee7834-f46e-4141-d61c-83cd7d72a333"
model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Stop training when there is no improvement in the validation loss for 10 consecutive epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Save the Model with the lowest validation loss
save_best = tf.keras.callbacks.ModelCheckpoint('./best_model.h5',
                                               monitor='val_loss',
                                               save_best_only=True)

history = model.fit(training_batches,
                    epochs = 100,
                    validation_data=validation_batches,
                    callbacks=[early_stopping, save_best])

# %% colab={} colab_type="code" id="sz4snGQsR2Mg"
