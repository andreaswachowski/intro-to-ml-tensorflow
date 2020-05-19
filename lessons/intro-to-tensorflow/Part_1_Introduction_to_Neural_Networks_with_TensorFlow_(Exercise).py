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

# %% [markdown] colab_type="text" id="q9aaxiSJg60h"
# # Introduction to Neural Networks with TensorFlow
#
# In this notebook, you'll get introduced to [TensorFlow](https://www.tensorflow.org/), an open source library to help you develop and train machine learning models.
#
# TensorFlow in a lot of ways behaves like the arrays you love from NumPy. NumPy arrays, after all, are just tensors. TensorFlow takes these tensors and makes it simple to move them to GPUs for the faster processing needed when training neural networks. TensorFlow can also calculate gradients automatically, which is perfect for backpropagation, and has  intuitive high-level APIs specifically for building neural networks. 

# %% [markdown] colab_type="text" id="Q6WXEV-OiFY7"
# ## Neural Networks
#
# Deep Learning is based on artificial neural networks which have been around in some form since the late 1950s. The networks are built from individual parts approximating neurons, typically called units or simply "neurons." Each unit has some number of weighted inputs. These weighted inputs are summed together (a linear combination) then passed through an activation function to get the unit's output.
#
# <img src="assets/simple_neuron.png" width=400px>
#
# Mathematically, the above looks like: 
#
# $$
# y = f(h)
# $$
#
# where,
#
# $$
# h = w_1 x_1 + w_2 x_2 + b = \sum_{i=1}^2 w_i x_i + b
# $$
#
# If we let $b = w_0x_0$, then we can express $h$ as the dot/inner product of two vectors:
#
# $$
# h = \begin{bmatrix}
# x_0 \, x_1 \, x_2
# \end{bmatrix}
# \cdot 
# \begin{bmatrix}
#            w_0 \\
#            w_1 \\
#            w_2
# \end{bmatrix} = w_0 x_0 + w_1 x_1 + w_2 x_2
# $$

# %% [markdown] colab_type="text" id="AcBD-UAaiaUc"
# ## Tensors
#
# It turns out neural network computations are just a bunch of linear algebra operations on *tensors*, a generalization of matrices. A vector is a 1-dimensional tensor, a matrix is a 2-dimensional tensor, an array with three indices is a 3-dimensional tensor (RGB color images for example). The fundamental data structure for neural networks are tensors and TensorFlow is built around tensors.
#
# <img src="assets/tensor_examples.svg" width=600px>
#
# With the basics covered, it's time to explore how we can use TensorFlow to build a simple neural network.

# %% [markdown] colab_type="text" id="pt6qo5hiesm7"
# ## Import Resources

# %% colab={} colab_type="code" id="adTiqrSxkPvL"
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

# %%
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# %% colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" id="jyZjjOb-V4eo" outputId="0bb960f6-e973-46af-eccb-a923475310e3"
print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)

# %% [markdown] colab_type="text" id="IVTYgypBfx2a"
# ## Single Layer Neural Network

# %% colab={"base_uri": "https://localhost:8080/", "height": 153} colab_type="code" id="UDJArguZkLPJ" outputId="8f187357-5089-49ab-8112-2819fa3ec786"
# Set the random seed so things are reproducible
tf.random.set_seed(7) 

# Create 5 random input features
features = tf.random.normal((1, 5))

# Create random weights for our neural network
weights = tf.random.normal((1, 5))

# Create a random bias term for our neural network
bias = tf.random.normal((1, 1))

print('Features:\n', features)
print('\nWeights:\n', weights)
print('\nBias:\n', bias)


# %% [markdown] colab_type="text" id="7o-2IomPkZx6"
# Above we generated data we can use to get the output of our simple network. This is all just random for now, going forward we'll start using real data. 
#
# As we can see above, in TensorFlow, tensors are `tf.Tensor` objects that have a data type and a shape. Now let's go through each relevant line:
#
# * `features = tf.random.normal((1, 5))` creates a tensor with shape `(1, 5)`, one row and five columns, that contains random values from a normal distribution with a mean of zero and standard deviation of one.
#
# * `weights = tf.random.normal((1, 5))` creates a tensor with shape `(1, 5)`, one row and five columns, again containing random values from a normal distribution with a mean of zero and standard deviation of one.
#
# * `bias = tf.random.normal((1, 1))` creates a single random value from a normal distribution.
#
# TensorFlow's tensors can be added, multiplied, subtracted, etc., just like NumPy arrays. In general, you'll use TensorFlow's tensors pretty much the same way you'd use NumPy arrays. They come with some nice benefits though such as GPU acceleration which we'll get to later. For now, we will use the generated random data to calculate the output of this simple single layer network.
#
# Now we create the activation function we are going to use:

# %% colab={} colab_type="code" id="wmOi7zjFgsMM"
def sigmoid_activation(x):
    """ Sigmoid activation function
    
        Arguments
        ---------
        x: tf.Tensor. Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.
    """
    return 1/(1+tf.exp(-x))


# %% [markdown] colab_type="text" id="57QrwP-0gny1"
# Now, let's calculate the output of our single layer neural network.
#
# > **Exercise**: Calculate the output of the network with input features `features`, weights `weights`, and bias `bias`. Similar to NumPy, TensorFlow has a `tf.multiply(a, b)` that multiplies tensors `a` and `b` element-wise. It also has `tf.reduce_sum(x)` that computes the sum of the elements of tensor `x`. Use the `sigmoid_activation` function defined above as the activation function.

# %% colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" id="qujbYK1nkaW-" outputId="d69a6561-5414-49e3-e8f3-419892991533"
## Solution
y = tf.reduce_sum(tf.multiply(features,weights)) + bias

print('label:\n', y)

# %% [markdown] colab_type="text" id="gx6LpDRakgOD"
# You can do the multiplication and sum in the same operation using a matrix multiplication. In general, you'll want to use matrix multiplications since they are more efficient and accelerated using modern libraries and high-performance computing on GPUs.
#
# Here, we want to do a matrix multiplication of the features and the weights. For this we can use `tf.matmul()`. If we try to do it with `features` and `weights` as they are, we'll get an error
#
# ```python
# >> tf.matmul(features, weights)
#
# ---------------------------------------------------------------------------
# InvalidArgumentError                      Traceback (most recent call last)
# <ipython-input-7-66a4fe44f20b> in <module>()
#       1 
# ----> 2 y = sigmoid_activation(tf.matmul(features, weights) + bias)
#
# 2 frames
# /usr/local/lib/python3.6/dist-packages/six.py in raise_from(value, from_value)
#
# InvalidArgumentError: Matrix size-incompatible: In[0]: [1,5], In[1]: [1,5] [Op:MatMul] name: MatMul/
# ```
#
# As you're building neural networks in any framework, you'll see this often. Really often. What's happening here is our tensors aren't the correct shapes to perform a matrix multiplication. Remember that for matrix multiplications, the number of columns in the first tensor must equal to the number of rows in the second tensor. Both `features` and `weights` have the same shape, `(1, 5)`. This means we need to change the shape of `weights` to get the matrix multiplication to work.
#
# **Note:** To see the shape of a tensor called `tensor`, use `tensor.shape`. If you're building neural networks, you'll be using this method often.

# %% colab={"base_uri": "https://localhost:8080/", "height": 68} colab_type="code" id="5flC0NB5uSlk" outputId="2aa713f0-3563-4845-bdd9-5331e07e694d"
print('Features Shape:', features.shape)
print('Weights Shape:', weights.shape)
print('Bias Shape:', bias.shape)

# %% [markdown] colab_type="text" id="lC0vZqQWvBon"
# Now, we need to change the shape of the `weights` tensor to get the matrix multiplication to work. However, if we use the `tf.matmul(a,b)` function, this will not be necessary because the `tf.matmul(a,b)` function has the `transpose_a` and `transpose_b` arguments that allows us to transpose the `a` or `b` tensors for the multiplication without actually changing the shape of the tensor permanently. Therefore, in this case we can use the argument `transpose_b = True` to transpose the `weights` tensor from `(1,5)` to `(5,1)` just for the multiplication.
#
# > **Exercise**: Calculate the output of our little network using matrix multiplication.

# %% colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" id="iIQOahT4kg4H" outputId="273d8dcb-ade7-46ad-f720-6c439e837d4d"
## Solution
y = tf.matmul(features,weights, transpose_b = True) + bias

print('label:\n', y)

# %% [markdown] colab_type="text" id="pxzezexmkpzs"
# We can see that we get the same value as before. Now let's check that the shape of the `weights` tensor has not been permanently changed by the `transpose_b = True` argument. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="XoESCJcLkrI5" outputId="d771c85d-ba0b-4a17-cff1-812edf3aa61b"
print('Weights Shape:', weights.shape)

# %% [markdown] colab_type="text" id="hYAyS685kv4e"
# As we can see, the shape of the `weights` tensor remains unchanged.

# %% [markdown] colab_type="text" id="ktu4Jfq3kzWx"
# ## Multi-Layer Neural Network
#
# That's how you can calculate the output for a single neuron. The real power of this algorithm happens when you start stacking these individual units into layers and stacks of layers, into a network of neurons. The output of one layer of neurons becomes the input for the next layer. With multiple input units and output units, we now need to express the weights as a matrix.
#
# <img src='assets/multilayer_diagram_weights.png' width=450px>
#
# In the diagram above, the first layer shown on the bottom are the inputs, understandably called the **input layer**. The middle layer is called the **hidden layer**, and the final layer (on the top) is the **output layer**. We can express this network mathematically with matrices and use matrix multiplication to get linear combinations for each unit in one operation. For example, the hidden layer (comprised of units $h_1$ and $h_2$) can be calculated as follows:
#
# $$
# \vec{h} = [h_1 \, h_2] = 
# \begin{bmatrix}
# x_1 \, x_2 \cdots \, x_n
# \end{bmatrix}
# \cdot 
# \begin{bmatrix}
#            w_{11} & w_{12} \\
#            w_{21} &w_{22} \\
#            \vdots &\vdots \\
#            w_{n1} &w_{n2}
# \end{bmatrix}
# $$
#
# The output for this small network is found by treating the hidden layer as inputs for the output unit. The network output is expressed simply as:
#
# $$
# y =  f_2 \! \left(\, f_1 \! \left(\vec{x} \, \mathbf{W_1}\right) \mathbf{W_2} \right)
# $$

# %% colab={} colab_type="code" id="yWRfPCIKkwZb"
# Set the random seed so things are reproducible
tf.random.set_seed(7) 

# Create 3 random input features
features = tf.random.normal((1,3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Create random weights connecting the inputs to the hidden layer
W1 = tf.random.normal((n_input,n_hidden))

# Create random weights connecting the hidden layer to the output layer
W2 = tf.random.normal((n_hidden, n_output))

# Create random bias terms for the hidden and output layers
B1 = tf.random.normal((1,n_hidden))
B2 = tf.random.normal((1, n_output))

# %% [markdown] colab_type="text" id="cCRRqGFik5N8"
# > **Exercise:** Calculate the output for this multi-layer network using the weights `W1` & `W2`, and the biases, `B1` & `B2`. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="eKKfk_jyk5_n" outputId="fc2f418d-21f7-458f-8905-36dd9bdc183e"
## Solution

output_hidden = sigmoid_activation(tf.matmul(features,W1) + B1)
output = sigmoid_activation(tf.matmul(output_hidden,W2) + B2)

print(output)

# %% [markdown] colab_type="text" id="9Q1QzBpFlAFj"
# If you did this correctly, you should see the output: `tf.Tensor([[0.10678572]], shape=(1, 1), dtype=float32)`
#
# The number of hidden units is a parameter of the network, often called a **hyperparameter** to differentiate it from the weights and biases parameters. As you'll see later when we discuss training a neural network, the more hidden layers and units a network has, the better able it will be able to learn from data and make accurate predictions.

# %% [markdown] colab_type="text" id="qASlsncVlDRi"
# ## NumPy to TensorFlow and Back
#
# This is a special bonus section! TensorFlow has a great feature for converting between NumPy's `ndarrays` and its `tf.Tensors`. To create a tensor from a NumPy ndarray, use `tf.convert_to_tensor()`. To convert a tensor to a NumPy array, use the `.numpy()` method. Below are some examples:

# %% colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" id="5MHSE0uqlAsS" outputId="22d3e319-e6c3-4bce-8f92-5414558df10c"
# Set the random seed so things are reproducible
tf.random.set_seed(7) 

a = np.random.rand(4,3)

print(type(a))
print(a)

# %% colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" id="VceuFk3GlKKL" outputId="8691f3b8-8b74-403c-cf5a-f7274aef0c9c"
b = tf.convert_to_tensor(a)

print(b)

# %% colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" id="QOh6E9SBlMzj" outputId="40de0a13-be50-4c94-cd9b-fc20e1396646"
c = b.numpy()

print(type(c))
print(c)

# %% [markdown] colab_type="text" id="Ua9iXmXBlP4M"
# If you change the values of the Tensor, the values of the ndarray will not change, and vice-versa.

# %% colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" id="BC_O_BcOlTHa" outputId="c49af615-b311-4c6c-c66d-636f32dd654b"
# Multiply TensorFlow Tensor by 40
b = b * 40

print(b)

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="R1gxSglFlV2R" outputId="3944e74f-62c4-4222-d505-b647ea24f4e1"
# NumPy array stays the same
a

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="4tkHnTSbNKdE" outputId="85afcc3d-cfdd-4db0-e246-3e05ff76a04e"
# NumPy array stays the same
c

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} colab_type="code" id="z_vdtpxnlYUN" outputId="2788ac27-bb7f-4738-9d23-c96d09f0326a"
# Add 1 to NumPy ndarray
a = a + 1

print(a)

# %% colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" id="sdtiRFFDlblU" outputId="67e8adc1-55e0-498a-de20-f4f2512ce961"
# Tensor stays the same
print(b)

# %%
# NumPy array stays the same
c
