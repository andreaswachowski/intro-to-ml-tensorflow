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

# %% [markdown] colab_type="text" id="Lg2hLK7hlWdb"
# # Classifying Fashion-MNIST
#
# Now it's your turn to build and train a neural network. You'll be using the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), a drop-in replacement for the MNIST dataset. MNIST is actually quite trivial with neural networks where you can easily achieve better than 97% accuracy. Fashion-MNIST is a set of 28x28 greyscale images of clothes. It's more complex than MNIST, so it's a better representation of the actual performance of your network, and a better representation of datasets you'll use in the real world.
#
# <img src='assets/fashion-mnist-sprite.png' width=500px>
#
# In this notebook, you'll build your own neural network. For the most part, you could just copy and paste the code from Part 3, but you wouldn't be learning. It's important for you to write the code yourself and get it to work. Feel free to consult the previous notebooks though as you work through this.
#
# First off, let's import our resources and download the Fashion-MNIST dataset from `tensorflow_datasets`. 

# %% [markdown] colab_type="text" id="EMflYTIOtOPf"
# ## Import Resources

# %%
import warnings
warnings.filterwarnings('ignore')

# %% colab={} colab_type="code" id="U0n2QWj1p2fG"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="FwP1_Qw-cCsY" outputId="5cc63000-690c-4063-d0c4-2f242819ccac"
print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

# %% [markdown] colab_type="text" id="Vr2SOjl8txrZ"
# ## Load the Dataset
#
# We are now going to load the Fashion-MNIST dataset using `tensorflow_datasets` as we've done before. In this case, however, we are going to omit the `split` argument.  This means that `tensorflow_datasets` will use the default value for `split` which is `split=None`. When `split=None`, `tensorflow_datasets` returns a **dictionary** with all the splits available for the dataset you are loading. However, if the split is given explicitly, such as `split='train'`, then `tensorflow_datasets` returns a `tf.data.Dataset` object.
#
# In our case, we are going to load the `fashion_mnist` dataset. If we look at the [documentation](https://www.tensorflow.org/datasets/catalog/fashion_mnist#statistics) we will see that this particular dataset has 2 splits, namely a `train` and a `test` split. We also see that the `train` split has 60,000 examples, and that the `test` split has 10,000 examples. 
#
# Now, let's load the `fashion_mnist` dataset and inspect the returned values.

# %% colab={"base_uri": "https://localhost:8080/", "height": 54} colab_type="code" id="1kn4Op7dXCnk" outputId="cd83ee11-b25e-4df2-dbf7-2026fd2049da"
dataset, dataset_info = tfds.load('fashion_mnist', as_supervised = True, with_info = True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" id="2_vT6HUUXg05" outputId="fcdf4d7e-d14b-491b-b6c1-235823d67875"
# Check that dataset is a dictionary
print('dataset has type:', type(dataset))

# Print the keys of the dataset dictionary
print('\nThe keys of dataset are:', list(dataset.keys()))

# %% [markdown] colab_type="text" id="6S4f2J9jbpak"
# In the cell below, we are going to save the training data and the test data into different variables.

# %% colab={} colab_type="code" id="kxo7PHJys18t"
training_set, test_set = dataset['train'], dataset['test']

# %% [markdown] colab_type="text" id="zzZciG_KcHbI"
# Now, let's take a look at the `dataset_info`

# %% colab={"base_uri": "https://localhost:8080/", "height": 598} colab_type="code" id="7jFE3vbebU-A" outputId="faaf389e-4d0b-4d51-f565-34aba4ae5cfd"
# Display the dataset_info
dataset_info

# %% [markdown] colab_type="text" id="0_If36cti685"
# We can access the information in `dataset_info` very easily. As we can see, the `features` and `splits` info are contained in dictionaries. We can access the information we want by accessing the particular key and value in these dictionaries. We start by looking at the values of particular keys in these dictionaries:

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="6KtD7j5HgTkn" outputId="926d32e3-644b-45ff-c86e-119663fcabc6"
dataset_info.features['image']

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="l_QXhcTOiQ1a" outputId="53da5e37-9e6e-45ee-c395-81166c3e6e5c"
dataset_info.features['label']

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="gGn6yzTxgKwj" outputId="442c79f6-a5c6-4d4c-8b84-7f2d93778d81"
dataset_info.splits['train']

# %% [markdown] colab_type="text" id="MFwhpPOijumG"
# We can now use dot notation to access the information we want. Below are some examples. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" id="m9_OYPHsbbcl" outputId="9b7e79ce-1932-443c-85b9-1dca8b55eade"
shape_images = dataset_info.features['image'].shape
num_classes = dataset_info.features['label'].num_classes

num_training_examples  = dataset_info.splits['train'].num_examples
num_test_examples = dataset_info.splits['test'].num_examples

print('There are {:,} classes in our dataset'.format(num_classes))
print('The images in our dataset have shape:', shape_images)

print('\nThere are {:,} images in the test set'.format(num_test_examples))
print('There are {:,} images in the training set'.format(num_training_examples))

# %% [markdown] colab_type="text" id="nfMgIb3PvWXo"
# ## Explore the Dataset
#
# The images in this dataset are 28 $\times$ 28 arrays, with pixel values in the range `[0, 255]`. The *labels* are an array of integers, in the range `[0, 9]`. These correspond to the *class* of clothing the image represents:
#
# <table>
#   <tr>
#     <th>Label</th>
#     <th>Class</th> 
#   </tr>
#   <tr>
#     <td>0</td>
#     <td>T-shirt/top</td> 
#   </tr>
#   <tr>
#     <td>1</td>
#     <td>Trouser</td> 
#   </tr>
#     <tr>
#     <td>2</td>
#     <td>Pullover</td> 
#   </tr>
#     <tr>
#     <td>3</td>
#     <td>Dress</td> 
#   </tr>
#     <tr>
#     <td>4</td>
#     <td>Coat</td> 
#   </tr>
#     <tr>
#     <td>5</td>
#     <td>Sandal</td> 
#   </tr>
#     <tr>
#     <td>6</td>
#     <td>Shirt</td> 
#   </tr>
#     <tr>
#     <td>7</td>
#     <td>Sneaker</td> 
#   </tr>
#     <tr>
#     <td>8</td>
#     <td>Bag</td> 
#   </tr>
#     <tr>
#     <td>9</td>
#     <td>Ankle boot</td> 
#   </tr>
# </table>
#
# Each image is mapped to a single label. Since the *class names* are not included with the dataset, we create them here to use later when plotting the images:

# %% colab={} colab_type="code" id="odzN3aJjusED"
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

# %% colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" id="RoY1HeJJyces" outputId="c6d817e3-4150-4f8f-8b28-298b0936e794"
for image, label in training_set.take(1):
    print('The images in the training set have:\n\u2022 dtype:', image.dtype, '\n\u2022 shape:', image.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 301} colab_type="code" id="CInprnnJ1_gk" outputId="aa2945e1-9f33-4d2e-8191-a47dd7dbb29f"
for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

plt.imshow(image, cmap= plt.cm.binary)
plt.colorbar()
plt.show()

print('The label of this image is:', label)
print('The class name of this image is:', class_names[label])


# %% [markdown] colab_type="text" id="Hb-lmuTM35C9"
# ## Create Pipeline

# %% colab={} colab_type="code" id="3gq-_mXl3ZFG"
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

batch_size = 64

training_batches = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)
testing_batches = test_set.cache().batch(batch_size).map(normalize).prefetch(1)

# %% [markdown] colab_type="text" id="LviX4-ii8js7"
# ## Build the Model
#
# > **Exercise:** Here you should define your own neural network. Feel free to create a model with as many layers and neurons as you like. You should keep in mind that as with MNIST, each image is 28 $\times$ 28 which is a total of 784 pixels, and there are 10 classes. Your model should include at least one hidden layer. We suggest you use ReLU activation functions for the hidden layers and a softmax activation function for the output layer.

# %% colab={} colab_type="code" id="OYzFZ3jQ8azd"
## Solution

my_model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# %% [markdown] colab_type="text" id="CYhwsFzA-Aah"
# ## Train the Model
#
# > **Exercise:** Compile the model you created above using an `adam` optimizer, a `sparse_categorical_crossentropy` loss function, and the `accuracy` metric. Then train the model for 5 epochs. You should be able to get the training loss below 0.4.

# %% colab={"base_uri": "https://localhost:8080/", "height": 187} colab_type="code" id="Cyy9SqTU91IS" outputId="e8823c12-e7c0-4397-8126-2cb29e8be66a"
## Solution

my_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy",metrics=['accuracy']
)

for image_batch, label_batch in training_batches.take(1):
    loss, accuracy = my_model.evaluate(image_batch, label_batch)
    
print("\nLoss before training: {:,.3f}".format(loss))
print("\nAccuracy before training: {:,.3%}".format(accuracy))

EPOCHS = 5

history = my_model.fit(training_batches, epochs = EPOCHS)

# %% [markdown] colab_type="text" id="REJbwplUBoRT"
# ## Evaluate Loss and Accuracy on the Test Set
#
# Now let's see how the model performs on the test set. This time, we will use all the examples in our test set to assess the loss and accuracy of our model. Remember, the images in the test are images the model has never seen before.

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="q76aDGGl_xp4" outputId="d1ee69fc-874c-4985-cbd3-5bae323f64fb"
loss, accuracy = my_model.evaluate(testing_batches)

print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))

# %% [markdown] colab_type="text" id="PnpZWDQp2Zaq"
# ## Check Predictions

# %% colab={"base_uri": "https://localhost:8080/", "height": 225} colab_type="code" id="kqUzc4pYAe7Z" outputId="faa09287-401f-478d-85c1-6eb59eb748cd"
for image_batch, label_batch in testing_batches.take(1):
    ps = my_model.predict(image_batch)
    first_image = image_batch.numpy().squeeze()[0]
    first_label = label_batch.numpy()[0]

fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
ax1.imshow(first_image, cmap = plt.cm.binary)
ax1.axis('off')
ax1.set_title(class_names[first_label])
ax2.barh(np.arange(10), ps[0])
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(10))
ax2.set_yticklabels(class_names, size='small');
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()
