#!/usr/bin/env python

"""
State Farm Distracted Driver Detection
Convolutional Neural Network Script

This is a script for the kaggle competition:

https://www.kaggle.com/c/state-farm-distracted-driver-detection/
"""

from __future__ import print_function

import sys
import time
import glob
from loadingbar import loadingBar as loadbar
import numpy as np
import theano
import theano.tensor as T
from PIL import Image

import lasagne
import random


# ################## Prepare the Kaggle dataset ##################
# Function for reading in the dataset from its original .jpg format, into numpy arrays.
# Note that this script assumes the training images are pathed as 'imgs/train/c0/img_45.jpg',
# the test images are pathed as 'imgs/test/img_45.jpg'

def load_dataset():
    drivers = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022','p024','p026','p035','p039','p041','p042','p045','p047','p049','p050','p051','p052','p056','p061','p064','p066','p072','p075','p081']
    driversampledict = {'p022': 1233, 'p049': 1011, 'p021': 1237, 'p026': 1196, 'p002': 725, 'p041': 605, 'p042': 591, 'p045': 724, 'p047': 835, 'p024': 1226, 'p072': 346, 'p075': 814, 'p016': 1078, 'p015': 875, 'p014': 876, 'p039': 651, 'p012': 823, 'p035': 848, 'p052': 740, 'p051': 920, 'p050': 790, 'p056': 794, 'xsubject': 1, 'p066': 1034, 'p064': 820, 'p061': 809, 'p081': 823}
    random.shuffle(drivers)
    traindrivers = drivers[2:]
    valdrivers = drivers[:2]
    numtrainingvectors = 0
    numvalidationvectors = 0
    for key, value in driversampledict.items():
        if key in traindrivers:
            numtrainingvectors += value
        else:
            numvalidationvectors += value
    
    loadcounter = 0
    with open('data/driver_imgs_list.csv') as imglabels:
        labels = imglabels.readlines()[1:]  # the first line in metadata
        numlabels = len(labels)
        x_train = np.empty((numtrainingvectors, 1, 48, 64), np.float32)
        y_train = np.zeros(numtrainingvectors, np.int32)
        x_val = np.empty((numvalidationvectors, 1, 48, 64), np.float32)
        y_val = np.zeros(numvalidationvectors, np.int32)
        random.shuffle(labels)
        loadbar(loadcounter, numlabels, prefix = 'Progress:', suffix = 'Complete', barLength = 50, addline = True)        
        trainindex = 0
        valindex = 0
        for index, label in enumerate(labels):
            # Convert scale from integers [0, 255] to floats [0,1]
            x = np.asarray(Image.open("data/train/"+label[5:7]+"/"+label[8:].rstrip("\n")))[::10,::10]/np.float64(255)
            if label[:4] in valdrivers:
                x_val[valindex][0] = x
                y_val[valindex] = int(label[6:7])
                valindex += 1
            else:
                x_train[trainindex][0] = x
                y_train[trainindex] = int(label[6:7])
                trainindex += 1
            
            loadcounter += 1
            loadbar(loadcounter, numlabels, prefix = 'Progress:', suffix = 'Complete', barLength = 50, addline = True)        

    # 10% of the test set will be the validation set.
    vsize = int(len(y_train) * .1)

    # Reserve 10% of data set for validation.
    # x_train, x_val = x_train[:-vsize], x_train[-vsize:]
    # y_train, y_val = y_train[:-vsize], y_train[-vsize:]
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)

    return x_train, y_train, x_val, y_val


def build_cnn(input_var=None):
    # a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 48, 64), input_var=input_var)

    # Convolutional layer with 32 kernels of size 7x7. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # network = lasagne.layers.Conv2DLayer(network, num_filters=16, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, 3), nonlinearity=lasagne.nonlinearities.rectify)
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
    # lasagne.regularization.l2(network)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=500):
    # Load the dataset
    print("Loading data...")
    x_train, y_train, x_val, y_val = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs', "float32")
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # set up prediction function
    predict_probabilities = theano.function(inputs=[input_var], outputs=test_prediction)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        counter = 0
        print("Epoch {} of {}".format(epoch + 1, num_epochs))
        loadbar(counter, 103, prefix = 'Progress:', suffix = 'Complete', barLength = 50, addline = True)
        for batch in iterate_minibatches(x_train, y_train, 200, shuffle=False): 
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            counter += 1  
            loadbar(counter, 103, prefix = 'Progress:', suffix = 'Complete', barLength = 50, addline = True)

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(x_val, y_val, 200, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

    print("Building submission file.")
    submissionfile = open("submission.txt", "w")
    submissionfile.write("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n")
    
    # Finally, we build a file to submit to kaggle and test the given test images
    testfiles = glob.glob("data/test/*.jpg")
    numtestfiles = len(testfiles)
    loadcounter = 0
    loadbar(loadcounter, numtestfiles, prefix = 'Progress:', suffix = 'Complete', barLength = 50, addline = True)            
    for index, imgfile in enumerate(testfiles):
        submissionfile.write(imgfile[10:])
        testimage = np.asarray(Image.open(imgfile))[::10, ::10]/np.float32(256)
        testvector = np.empty((1, 1, 48, 64), np.float32)
        testvector[0][0] = testimage
        pred = predict_probabilities(testvector)
        # Convert scale from integers [0, 255] to floats [0,1]
        for el in pred[0]:
            submissionfile.write("," + str(el))
        submissionfile.write("\n")
        loadcounter += 1
        loadbar(loadcounter, numtestfiles, prefix = 'Progress:', suffix = 'Complete', barLength = 50, addline = True)        

    #Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    main(20)
