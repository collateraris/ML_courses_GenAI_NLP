"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filesname, 'rb') as fd:
        samples = fd.read()

    with gzip.open(label_filename, 'rb') as fd:
        labels = fd.read()

    _, imgs_cnt, rows_cnt, cols_cnt = struct.unpack('>iiii', samples[:16])
    _, labels_cnt = struct.unpack('>ii', labels[:8])

    assert imgs_cnt == labels_cnt

    decoded_samples = struct.unpack(f'>{imgs_cnt*rows_cnt*cols_cnt}B',
                                    samples[16:])
    decoded_labels = struct.unpack(f'>{labels_cnt}B', labels[8:])

    samples = np.array(decoded_samples, dtype=np.float32)
    samples = samples.reshape(imgs_cnt, rows_cnt*cols_cnt)
    samples -= samples.min()
    samples /= (samples.max() - samples.min())

    labels = np.array(decoded_labels, dtype=np.uint8)

    return samples, labels


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # Compute softmax probabilities
    exp_Z = ndl.exp(Z)
    sum_exp_Z = ndl.summation(exp_Z, axes=1, keepdims=True)
    softmax_probs = exp_Z / sum_exp_Z
    
    # Compute cross-entropy loss
    log_probs = ndl.log(softmax_probs)
    losses = -ndl.summation(y_one_hot * log_probs, axes=1)
    
    # Return average loss
    return ndl.summation(losses) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_batches = (num_examples + batch - 1) // batch
    
    for i in range(num_batches):
        start_idx = i * batch
        end_idx = min((i + 1) * batch, num_examples)
        
        # Get batch data
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        # Convert to tensors
        X_tensor = ndl.Tensor(X_batch)
        
        # Create one-hot encoding for y_batch
        y_one_hot = np.zeros((y_batch.shape[0], W2.shape[1]))
        y_one_hot[np.arange(y_batch.shape[0]), y_batch] = 1
        y_tensor = ndl.Tensor(y_one_hot)
        
        # Forward pass
        # logits = ReLU(X * W1) * W2
        hidden = ndl.relu(ndl.matmul(X_tensor, W1))
        logits = ndl.matmul(hidden, W2)
        
        # Compute loss
        loss = softmax_loss(logits, y_tensor)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        # Get gradients
        W1_grad = W1.grad.realize_cached_data()
        W2_grad = W2.grad.realize_cached_data()
        
        # Update weights
        W1_new = W1.realize_cached_data() - lr * W1_grad
        W2_new = W2.realize_cached_data() - lr * W2_grad
        
        # Create new tensors with updated weights
        W1 = ndl.Tensor(W1_new)
        W2 = ndl.Tensor(W2_new)
    
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
