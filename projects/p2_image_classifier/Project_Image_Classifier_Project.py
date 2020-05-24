# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [markdown] colab_type="text" id="y00b5TQZnqs_"
# # Your First AI application
#
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.
#
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from Oxford of 102 flower categories, you can see a few examples below.
#
# <img src='assets/Flowers.png' width=500px>
#
# The project is broken down into multiple steps:
#
# * Load the image dataset and create a pipeline.
# * Build and Train an image classifier on this dataset.
# * Use your trained model to perform inference on flower images.
#
# We'll lead you through each part which you'll implement in Python.
#
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

# %% [markdown] colab_type="text" id="kKnPjnLAftRV"
# ## Import Resources

# %% colab={} colab_type="code" id="2dCk6873paNW"
# TODO: Make all necessary imports.
import warnings

warnings.filterwarnings("ignore")

# %%
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

# Note: As of 2020-05-24, tfds-nightly is required.
# See https://www.tensorflow.org/datasets/catalog/oxford_flowers102
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

# %%
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# %%
print("Using:")
print("\t\u2022 TensorFlow version:", tf.__version__)
print("\t\u2022 tf.keras version:", tf.keras.__version__)

print(
    "\t\u2022 Running on GPU"
    if tf.test.is_gpu_available()
    else "\t\u2022 GPU device not found. Running on CPU"
)

# %% [markdown] colab_type="text" id="tWKF0YOarpCx"
# ## Load the Dataset
#
# Here you'll use `tensorflow_datasets` to load the [Oxford Flowers 102 dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102). This dataset has 3 splits: `'train'`, `'test'`, and `'validation'`.  You'll also need to make sure the training data is normalized and resized to 224x224 pixels as required by the pre-trained networks.
#
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet, but you'll still need to normalize and resize the images to the appropriate size.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="vXISRjfdrrQ6" outputId="6edf59b2-b468-4c4a-cff4-7cc7cfcc3c2d"
# TODO: Load the dataset with TensorFlow Datasets.
# TODO: Create a training set, a validation set and a test set.

dataset, dataset_info = tfds.load(
    "oxford_flowers102", as_supervised=True, with_info=True
)

# %%
(training_set, validation_set, test_set) = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)

# %% [markdown] colab_type="text" id="S5pdQnDbf0-j"
# ## Explore the Dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" id="CWR9ScCbPI_D" outputId="fdf01c8d-2db9-4d7c-a566-4db2599fd1ab"
# TODO: Get the number of examples in each set from the dataset info.
# TODO: Get the number of classes in the dataset from the dataset info.
for split in dataset_info.splits:
    print(
        f"The {split} dataset contains {dataset_info.splits[split].num_examples} examples."
    )

# TODO: Get the number of classes in the dataset from the dataset info.
num_classes = dataset_info.features["label"].num_classes

print(f"\nThere are {num_classes} classes in the dataset.")

# %% [markdown]
# Remark: It looks surprising that the test dataset is so much larger than the training dataset, but this is exactly how the authors designed the dataset. From the original paper (https://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/nilsback08.pdf):
#
# > The dataset is divided into a training set, a validation set and a test set. The training set and validation set each consist of 10 images per class (totalling 1030 images each). The test set consist of the remaining 6129 images (minimum 20 per class).

# %% colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" id="CWR9ScCbPI_D" outputId="fdf01c8d-2db9-4d7c-a566-4db2599fd1ab"
# TODO: Print the shape and corresponding label of 3 images in the training set.
for image, label in training_set.take(3):
    print("Image dtype:", image.dtype, ", shape:", image.shape)

# %% [markdown]
# Apparently, the images have different shapes, which we want to change during preprocessing. The same goes for the 8-bit integer pixel values which are better converted to floating point numbers for our purposes

# %% colab={"base_uri": "https://localhost:8080/", "height": 280} colab_type="code" id="DQbnq8htRTnl" outputId="32a0e1af-2b04-440e-ddb4-835732be3e83"
# TODO: Plot 1 image from the training set. Set the title
# of the plot to the corresponding image label.

for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

plt.imshow(image, cmap=plt.cm.binary)
plt.title(label)
plt.colorbar()
plt.show()

# %% [markdown] colab_type="text" id="zuh1841cs-j1"
# ### Label Mapping
#
# You'll also need to load in a mapping from label to category name. You can find this in the file `label_map.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/3.7/library/json.html). This will give you a dictionary mapping the integer coded labels to the actual names of the flowers.

# %% colab={} colab_type="code" id="JoVzdO3KsdSk"
with open("label_map.json", "r") as f:
    class_names = json.load(f)

# %% colab={"base_uri": "https://localhost:8080/", "height": 280} colab_type="code" id="fc6pMUZgEvUo" outputId="4274fd43-5cee-4523-885f-a18f6f277dd6"
# TODO: Plot 1 image from the training set. Set the title
# of the plot to the corresponding class name.

for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

plt.imshow(image, cmap=plt.cm.binary)
plt.title(class_names[str(label)])
plt.colorbar()
plt.show()

# %% [markdown] colab_type="text" id="0gL7AaqNf-NC"
# ## Create Pipeline

# %% colab={"base_uri": "https://localhost:8080/", "height": 88} colab_type="code" id="5hNznLbPNZxS" outputId="7c114910-b75f-4220-cda9-f84426ec2728"
# TODO: Create a pipeline for each set.

batch_size = 32
image_size = 224 # corresponding to the MobileNet image sizes

num_training_examples = dataset_info.splits['train'].num_examples

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image, label


training_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)
testing_batches = test_set.map(format_image).batch(batch_size).prefetch(1)


# %% [markdown] colab_type="text" id="gR9gtRbeXPYx"
# # Build and Train the Classifier
#
# Now that the data is ready, it's time to build and train the classifier. You should use the MobileNet pre-trained model from TensorFlow Hub to get the image features. Build and train a new feed-forward classifier using those features.
#
# We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students!
#
# Refer to the rubric for guidance on successfully completing this section. Things you'll need to do:
#
# * Load the MobileNet pre-trained network from TensorFlow Hub.
# * Define a new, untrained feed-forward network as a classifier.
# * Train the classifier.
# * Plot the loss and accuracy values achieved during training for the training and validation set.
# * Save your trained model as a Keras model.
#
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
#
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right.
#
# **Note for Workspace users:** One important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module. Also, If your model is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# %% colab={} colab_type="code" id="4zElEHViXLni"
# TODO: Build and train your network.

module_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(module_url, input_shape=(image_size, image_size,3))

feature_extractor.trainable = False


# %% colab={} colab_type="code" id="4zElEHViXLni"
model = tf.keras.Sequential(
    [
        feature_extractor,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(
            num_classes,
            activation="softmax",
        ),
    ]
)

model.summary()


# %%
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
save_best = tf.keras.callbacks.ModelCheckpoint('./best_model.h5',
                                               monitor='val_loss',
                                               save_best_only=True)

EPOCHS = 30

# %%
history = model.fit(
    training_batches,
    epochs=EPOCHS,
    validation_data=validation_batches,
    callbacks=[early_stopping, save_best],
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 498} colab_type="code" id="VU6sWzx4e7Yb" outputId="f7b5c7c5-683a-463c-9228-68c4918bdd5b"
# TODO: Plot the loss and accuracy values achieved during training for the training and validation set.

training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range=range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# %% [markdown] colab_type="text" id="qcTDnyvop3ky"
# ## Testing your Network
#
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="79l7-HM1cafO" outputId="6cf468a4-1e27-4f20-d63a-a8bdd78bcdbe"
# TODO: Print the loss and accuracy values achieved on the entire test set.

loss, accuracy = model.evaluate(testing_batches)

print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))


# %% [markdown] colab_type="text" id="pLsIDWnuqfkl"
# ## Save the Model
#
# Now that your network is trained, save the model so you can load it later for making inference. In the cell below save your model as a Keras model (*i.e.* save it as an HDF5 file).

# %% colab={} colab_type="code" id="7XOwdOjSptp-"
# TODO: Save your trained model as a Keras model.

# In fact, we already saved the best model by using a callback,
# but here goes how we can save a model afterwards:

saved_keras_model_filepath = './flower_model.h5'

model.save(saved_keras_model_filepath)


# %% [markdown] colab_type="text" id="rbeLSRC1rxuj"
# ## Load the Keras Model
#
# Load the Keras model you saved above.

# %% colab={"base_uri": "https://localhost:8080/", "height": 394} colab_type="code" id="3T6Dgc7Nrzds" outputId="f5d356dc-183f-4cd3-f15d-88ebb4966082"
# TODO: Load the Keras model

# reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath)
reloaded_keras_model = tf.keras.models.load_model(
    "./flower_model.h5", custom_objects={"KerasLayer": hub.KerasLayer}
)

reloaded_keras_model.summary()


# %% [markdown] colab_type="text" id="ZjucwuFrsyhJ"
# # Inference for Classification
#
# Now you'll write a function that uses your trained network for inference. Write a function called `predict` that takes an image, a model, and then returns the top $K$ most likely class labels along with the probabilities. The function call should look like:
#
# ```python
# probs, classes = predict(image_path, model, top_k)
# ```
#
# If `top_k=5` the output of the `predict` function should be something like this:
#
# ```python
# probs, classes = predict(image_path, model, 5)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
#
# Your `predict` function should use `PIL` to load the image from the given `image_path`. You can use the [Image.open](https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.open) function to load the images. The `Image.open()` function returns an `Image` object. You can convert this `Image` object to a NumPy array by using the `np.asarray()` function.
#
# The `predict` function will also need to handle pre-processing the input image such that it can be used by your model. We recommend you write a separate function called `process_image` that performs the pre-processing. You can then call the `process_image` function from the `predict` function.
#
# ### Image Pre-processing
#
# The `process_image` function should take in an image (in the form of a NumPy array) and return an image in the form of a NumPy array with shape `(224, 224, 3)`.
#
# First, you should convert your image into a TensorFlow Tensor and then resize it to the appropriate size using `tf.image.resize`.
#
# Second, the pixel values of the input images are typically encoded as integers in the range 0-255, but the model expects the pixel values to be floats in the range 0-1. Therefore, you'll also need to normalize the pixel values.
#
# Finally, convert your image back to a NumPy array using the `.numpy()` method.

# %% colab={} colab_type="code" id="oG7mJ1-5s1qe"
# TODO: Create the process_image function

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()




# %% [markdown]
# To check your `process_image` function we have provided 4 images in the `./test_images/` folder:
#
# * cautleya_spicata.jpg
# * hard-leaved_pocket_orchid.jpg
# * orange_dahlia.jpg
# * wild_pansy.jpg
#
# The code below loads one of the above images using `PIL` and plots the original image alongside the image produced by your `process_image` function. If your `process_image` function works, the plotted image should be the correct size.

# %%
from PIL import Image

image_path = "./test_images/hard-leaved_pocket_orchid.jpg"
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), ncols=2)
ax1.imshow(test_image)
ax1.set_title("Original Image")
ax2.imshow(processed_test_image)
ax2.set_title("Processed Image")
plt.tight_layout()
plt.show()


# %% [markdown]
# Once you can get images in the correct format, it's time to write the `predict` function for making inference with your model.
#
# ### Inference
#
# Remember, the `predict` function should take an image, a model, and then returns the top $K$ most likely class labels along with the probabilities. The function call should look like:
#
# ```python
# probs, classes = predict(image_path, model, top_k)
# ```
#
# If `top_k=5` the output of the `predict` function should be something like this:
#
# ```python
# probs, classes = predict(image_path, model, 5)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
#
# Your `predict` function should use `PIL` to load the image from the given `image_path`. You can use the [Image.open](https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.open) function to load the images. The `Image.open()` function returns an `Image` object. You can convert this `Image` object to a NumPy array by using the `np.asarray()` function.
#
# **Note:** The image returned by the `process_image` function is a NumPy array with shape `(224, 224, 3)` but the model expects the input images to be of shape `(1, 224, 224, 3)`. This extra dimension represents the batch size. We suggest you use the `np.expand_dims()` function to add the extra dimension.

# %% colab={} colab_type="code" id="SBnPKFJuGB32"
# TODO: Create the predict function
  
def predict_single_image(image_path, model):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = np.expand_dims(process_image(image), axis=0)
    predictions = model.predict(processed_image)
    return predictions[0]


# %%
def predict(image_path, model, top_k):
    predictions = predict_single_image(image_path, model)
    top_k_indices = np.argsort(-predictions)[:top_k]
    classes = list(map(str, top_k_indices + 1)) # +1 because labels are 1-based, arrays are 0-based
    probs = predictions[top_k_indices]
    return probs, classes


# %% [markdown] colab_type="text" id="aft8f_n5C7Co"
# # Sanity Check
#
# It's always good to check the predictions made by your model to make sure they are correct. To check your predictions we have provided 4 images in the `./test_images/` folder:
#
# * cautleya_spicata.jpg
# * hard-leaved_pocket_orchid.jpg
# * orange_dahlia.jpg
# * wild_pansy.jpg
#
# In the cell below use `matplotlib` to plot the input image alongside the probabilities for the top 5 classes predicted by your model. Plot the probabilities as a bar graph. The plot should look like this:
#
# <img src='assets/inference_example.png' width=600px>
#
# You can convert from the class integer labels to actual flower names using `class_names`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 336} colab_type="code" id="I_tBH8xGGVxQ" outputId="ef0fe795-65f3-49c5-fab0-086fac7d409d"
# TODO: Plot the input image along with the top 5 classes
import glob, os

for image_path in glob.glob('./test_images/*.jpg'):
    image_name, _ = os.path.splitext(os.path.basename(image_path))
    image = Image.open(image_path)
    image = process_image(np.asarray(image))
    top_k = 5
    probs, classes = predict(image_path, reloaded_keras_model, top_k)
    # processed_test_image = process_image(test_image)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(8,9), ncols=2)
    ax1.imshow(image, cmap = plt.cm.binary)
    ax1.axis('off')
    ax1.set_title(image_name)
    ax2.barh(np.arange(top_k), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(top_k))
    ax2.set_yticklabels([class_names[str(label)] for label in classes], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()
