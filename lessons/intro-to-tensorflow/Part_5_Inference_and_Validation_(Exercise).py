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

# %% [markdown] colab_type="text" id="zqcFCOmzHdDu"
# # Inference and Validation
#
# Now that you have a trained network, you can use it for making predictions. This is typically called **inference**, a term borrowed from statistics. However, neural networks have a tendency to perform *too well* on the training data and aren't able to generalize to data that hasn't been seen before. This is called **overfitting** and it impairs inference performance. To test for overfitting while training, we measure the performance on data not in the training set called the **validation** set. We avoid overfitting through regularization such as dropout while monitoring the validation performance during training.

# %% [markdown] colab_type="text" id="bmWozIIC46NB"
# ## Import Resources

# %%
import warnings
warnings.filterwarnings('ignore')

# %% colab={} colab_type="code" id="FVP2jRaGJ9qu"
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# %%
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="hYwKhqk_h6-y" outputId="72f54d14-8abc-4fc9-95bf-dd5c2cc1b56b"
print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

# %% [markdown] colab_type="text" id="tPFaSI1S5CM3"
# ## Load the Dataset
#
# We are now going to load the Fashion-MNIST dataset using tensorflow_datasets as we've done before. In this case, however, we are going to define how to split the dataset ourselves. We are going to split the dataset such that 60\% of the data will be used for training, 20\% of the data will be used for validation, and the remaining 20\% of the data will be used for testing. 
#
# To do this, we are going to do two things in succession. First we are going to combine (merge) the `train` and `test` splits. After the splits are merged into a single set, we are going to sub-split it into three sets, where the first set has 60\% of the data, the second set has 20\% of the data, and the third set has the remaining 20\% of the data. 
#
# To merge all the splits of a dataset together, we can use `split=tfds.Split.ALL`. For example,
#
# ```python
# dataset = tfds.load('fashion_mnist', split=tfds.Split.ALL)
# ```
#
# will return a `dataset` that contains a single set with 70,000 examples. This is because the Fashion-MNIST dataset has a `train` split with 60,000 examples and a `test` split with 10,000 examples. The `tfds.Split.ALL` keyword merged both splits into a single set containing the combined data from both splits. 
#
# After we have merged the splits into a single set, we need to sub-split it. We can sub-split our dataset by using the `.subsplit()` method. There are various ways in which we can use the `.subsplit()` method. Here we are going to sub-split the data by providing the percentage of data we want in each set. To do this we just pass a list with percentages we want in each set. For example,
#
# ```python
# split=tfds.Split.ALL.subsplit([60,20,20])
# ```
#
# will sub-split our dataset into three sets, where the first set has 60\% of the data, the second set has 20\% of the data, and the third set has the remaining 20\% of the data. A word of **caution**, TensorFlow Datasets does not guarantee the reproducibility of the sub-split operations. That means, that two different users working on the same version of a dataset and using the same sub-split operations could end-up with two different sets of examples. Also, if a user regenerates the data, the sub-splits may no longer be the same. To learn more about `subsplit` and other ways to sub-split your data visit the [Split Documentation](https://www.tensorflow.org/datasets/splits#subsplit).

# %% colab={"base_uri": "https://localhost:8080/", "height": 207} colab_type="code" id="bgIzJ4oRLQpd" outputId="a79abec7-96bb-4b4b-cf34-399447bb9814"
# tfds.Split.ALL has been removed, see
# https://github.com/tensorflow/datasets/issues/1455
#
# subsplit was deprecated and has also been removed,
# https://github.com/tensorflow/datasets/issues/1998
#
# For the new way, see
# https://www.tensorflow.org/datasets/splits

train_split = 60
test_val_split = 20

def splitstr(from_percent, to_percent):
    return f"train[{from_percent}%:{to_percent}%]+test[{from_percent}%:{to_percent}%]"
    
splits = [f"{splitstr(0,train_split)}",
          f"{splitstr(train_split,train_split+test_val_split)}",
          f"{splitstr(train_split+test_val_split,100)}"]

print(splits)
dataset, dataset_info = tfds.load(
    "fashion_mnist", split=splits, as_supervised=True, with_info=True
)

training_set, validation_set, test_set = dataset

# %% [markdown] colab_type="text" id="jNJA3Xe-A4q_"
# When we use `split=tfds.Split.ALL.subsplit([60,20,20])`, `tensorflow_datasets` returns a tuple with our sub-splits. Since we divided our dataset into 3 sets, then, in this case, `dataset` should be a tuple with 3 elements.

# %% colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" id="3OBvm_yf5ijj" outputId="d239ea33-ce86-4817-fe04-2e4785d7fd5d"
# Check that dataset is a tuple
print('dataset has type:', type(dataset))

# Print the number of elements in dataset
print('dataset has {:,} elements '.format(len(dataset)))

# %% colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" id="MGwZVYsj6OXe" outputId="605bf42a-34cd-4a36-9907-ec2c16beeb11"
# Display dataset
dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 598} colab_type="code" id="2HL4BL_lnz2z" outputId="a7558de0-93b8-4f40-80e9-e378ce248fc4"
# Display dataset_info
dataset_info

# %% [markdown] colab_type="text" id="14QBpBlU5sOm"
# ## Explore the Dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" id="_GnHgnh-eSuf" outputId="156da5c6-bc6c-4d8a-8768-26d4f48c2e13"
total_examples = dataset_info.splits['train'].num_examples + dataset_info.splits['test'].num_examples

num_training_examples = (total_examples * train_split) // 100
num_validation_examples = (total_examples * test_val_split) // 100
num_test_examples = num_validation_examples

print('There are {:,} images in the training set'.format(num_training_examples))
print('There are {:,} images in the validation set'.format(num_validation_examples))
print('There are {:,} images in the test set'.format(num_test_examples))

# %% colab={} colab_type="code" id="4WMKWKxPcgOU"
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']


# %% [markdown] colab_type="text" id="xlHOpMIq5yYa"
# ## Create Pipeline

# %% colab={} colab_type="code" id="mBAzrt_nUfNZ"
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 64

training_batches = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)
validation_batches = validation_set.cache().batch(batch_size).map(normalize).prefetch(1)
testing_batches = test_set.cache().batch(batch_size).map(normalize).prefetch(1)

# %% [markdown] colab_type="text" id="39MO_CpdneIY"
# ## Build the Model
#
# Here we'll build and compile our model as usual.

# %% colab={} colab_type="code" id="agzupDJxnekW"
model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
])

# %% colab={} colab_type="code" id="uI0kZt-cpbXO"
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %% [markdown] colab_type="text" id="-cyd0DQSoazb"
# ## Evaluate Loss and Accuracy on the Test Set
#
# The goal of validation is to measure the model's performance on data that isn't part of the training set. Performance here is up to the developer to define though. Typically this is just accuracy, the percentage of classes the network predicted correctly. Other options are [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)) and top-5 error rate. We'll focus on accuracy here. Let's see how the model performs on our test set.

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="P3kE7BEAobKs" outputId="7e140d18-5cdc-404c-f3da-21310456965e"
loss, accuracy = model.evaluate(testing_batches)

print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))

# %% [markdown] colab_type="text" id="mx52hCxlp27g"
# The network is untrained so it's making random guesses and we should see an accuracy around 10%.

# %% [markdown] colab_type="text" id="ziyVd9R76H25"
# ## Train the Model with the Validation Set
#
# Now let's train our network as usual, but this time we are also going to incorporate our validation set into the training process. 
#
# During training, the model will only use the training set in order to decide how to modify its weights and biases. Then, after every training epoch we calculate the loss on the training and validation sets. These metrics tell us how well our model is "learning" because it they show how well the model generalizes to data that is not used for training. It's important to remember that the model does not use any part of the validation set to tune its weights and biases, therefore it can tell us if we're overfitting the training set.
#
# We can incorporate our validation set into the training process by including the `validation_data=validation_batches` argument in the `.fit` method.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="GFmdnOz1pNoa" outputId="538cb017-140c-4ece-e515-08083b3eaf29"
EPOCHS = 30

history = model.fit(
    training_batches,
    epochs=EPOCHS,
    validation_data=validation_batches
)

# %% [markdown] colab_type="text" id="CMQnPRTwPZbU"
# ## Loss and Validation Plots
#
# If we look at the training and validation losses achieved on epoch 30 above, we see that the loss on the training set is much lower than that achieved on the validation set. This is a clear sign of overfitting. In other words, our model has "memorized" the training set so it performs really well on it, but when tested on data that it wasn't trained on (*i.e.* the validation dataset) it performs poorly. 
#
# Let's take a look at the model's loss and accuracy values obtained during training on both the training set and the validation set. This will allow us to see how well or how bad our model is "learning". We can do this easily by using the `history` object returned by the `.fit` method. The  `history.history` attribute is a **dictionary** with a record of training accuracy and loss values at successive epochs, as well as validation accuracy and loss values when applicable. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" id="EYjSIPoO6hXC" outputId="bc4ff436-db55-4d55-f6d4-82003b9ecd50"
# Check that history.history is a dictionary
print('history.history has type:', type(history.history))

# Print the keys of the history.history dictionary
print('\nThe keys of history.history are:', list(history.history.keys()))

# %% [markdown] colab_type="text" id="QyUfyPzD9hLA"
# Let's use the `history.history` dictionary to plot our model's loss and accuracy values obtained during training.

# %% colab={"base_uri": "https://localhost:8080/", "height": 498} colab_type="code" id="wDFZCZnArx1T" outputId="1f696a01-ceaf-4a65-b04a-cb33ddc8d09a"
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

# %% [markdown] colab_type="text" id="MzkpAw5SP4cI"
# ## Early Stopping
#
# If we look at the training and validation losses as we train the network, we can see a phenomenon known as overfitting. 
# This happens when our model performs really well on the training data but it fails to generalize well enough to also perform well on the validation set. We can tell that this is happening because when we finished training the validation loss is higher than the training loss.
#
# One way to prevent our model from overfitting is by stopping training when we achieve the lowest validation loss. If we take a look at the plots we can see that at the beginning of training the validation loss starts decreasing, then after some epochs it levels off, and then it just starts increasing. Therefore, we can stop training our model when the validation loss levels off, such that our network is accurate but it's not overfitting.
#
# This strategy is called **early-stopping**. We can implement early stopping in `tf.keras` by using a **callback**. A callback is a set of functions to be applied at given stages of the training process. You can pass a list of callbacks to the `.fit()` method by using the `callbacks` keyword argument. 
#
# To implement early-stopping during training we will use the callback:
#
#
# ```python
# tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# ```
#
# The `monitor` argument specifies the quantity we want to monitor during training to determine when to stop training. The `patience` argument determines the number of consecutive epochs with no significant improvement after which training will be stopped. We can also specify the minimum change in the monitored quantity to qualify as an improvement, by specifying the `min_delta` argument. For more information on the early-stopping callback check out the [EarlyStopping 
# documentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/callbacks/EarlyStopping#class_earlystopping).

# %% colab={"base_uri": "https://localhost:8080/", "height": 425} colab_type="code" id="1Ch_iMqOQ6K3" outputId="5717abb0-17ca-48cc-b4f5-b1bca74af9c0"
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

# Stop training when there is no improvement in the validation loss for 5 consecutive epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(training_batches,
                    epochs = 100,
                    validation_data=validation_batches,
                    callbacks=[early_stopping])

# %% colab={"base_uri": "https://localhost:8080/", "height": 498} colab_type="code" id="mcyx8WQHVxEW" outputId="07ce0c7e-4263-4af9-91ba-cb5d82dd694f"
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range=range(len(training_accuracy))

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

# %% [markdown] colab_type="text" id="dSNjBbarspdj"
# ## Dropout
#
# Another common method to reduce overfitting is called **dropout**, where we randomly drop neurons in our model during training. This forces the network to share information between weights, increasing its ability to generalize to new data. We can implement dropout in `tf.keras` by adding `tf.keras.layers.Dropout()` layers to our models. 
#
# ```python
# tf.keras.layers.Dropout(rate)
# ```
# randomly sets a fraction `rate` of the dropout layer's input units to 0 at each update during training. The `rate` argument is a float between 0 and 1, that determines the fraction of neurons from the previous layer that should be turned off. For example,  `rate =0.5` will drop 50\% of the neurons. 
#
# It's important to note that we should never apply dropout to the input layer of our network. Also, remember that during training we want to use dropout to prevent overfitting, but during inference we want to use all the neurons in the network. `tf.keras` is designed to care of this automatically, so it uses the dropout layers during training, but automatically ignores them during inference.

# %% [markdown] colab_type="text" id="ABj6x-zss0I1"
# > **Exercise:** Add 3 dropout layers with a `rate=0.2` to our previous `model` and train it on Fashion-MNIST again. See if you can get a lower validation loss.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="IvSOnFdBsfbL" outputId="83adf78f-bbaf-456f-ce94-2449834e5864"
## Solution

model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(10, activation = 'softmax')
])


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

history = model.fit(training_batches,
                   epochs=100,
                   validation_data=validation_batches)

# %% [markdown] colab_type="text" id="Eqc6YFpFvwIq"
# ## Inference
#
# Now that the model is trained, we can use it to perform inference. Here we are going to perform inference on 30 images and print the labels in green if our model's prediction matches the true label. On the other hand, if our model's prediction doesn't match the true label, we print the label in red. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 858} colab_type="code" id="_bA9AnH9vq2m" outputId="f4a0767d-a17a-4ddf-facd-4caa5aa3aac9"
for image_batch, label_batch in testing_batches.take(1):
    ps = model.predict(image_batch)
    images = image_batch.numpy().squeeze()
    labels = label_batch.numpy()


plt.figure(figsize=(10,15))

for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(images[n], cmap = plt.cm.binary)
    color = 'green' if np.argmax(ps[n]) == labels[n] else 'red'
    plt.title(class_names[np.argmax(ps[n])], color=color)
    plt.axis('off')

# %% [markdown] colab_type="text" id="DcBmIg4Sdri_"
# ## Next Up!
#
# In the next lesson, we'll see how to save our trained models. In general, you won't want to train a model every time you need it. Instead, you'll train once, save it, then load the model when you want to train more or use it for inference.

# %% colab={} colab_type="code" id="nQdG3m0N9yDl"
