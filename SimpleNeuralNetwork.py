# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 19:11:35 2016

@author: Tom Blanchet
"""

import numpy as np

dot = np.dot

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Thanks Wendellllllllcow
    """
    try:
        try:
            type(x.shape[1]) == int

            scoresLong = np.reshape(x, (len(x)*len(x[0])), order = 'F')
            scoresOrdered = np.reshape(scoresLong, (len(x[0]), len(x)))

            a = np.array([])
            for i in scoresOrdered:
                a = np.append(a, softmaxCalculate(i))

            a = np.reshape(a, (len(x), len(x[0])), order = 'F')
            return a
        except:
            return softmaxCalculate(x)

    except:
        return softmaxCalculate(x)

def softmaxCalculate(x):
    num = np.exp(x)
    denom = num.sum()
    return num / denom

def differentiate(func, accuracy = 1e-5):
    """
    Given function func (numeric->numeric), returns a function that
    evaluates the pointwise approximate derivative of func.
    If None, this will assume it is the identity function f(x) -> x.
    """
    if(func == None):
        def const(x):
            return x*0 + 1 #To work with numpy
        return const

    def dfunc_over_dx(x):
        return (func(x+accuracy)-func(x-accuracy))/(accuracy*2)

    return dfunc_over_dx

class NeuronLayer:
    """
    A layer of neurons.

    __init__:
        inSize:
            number of inputs into the layer
        outSize:
            number of outputs (neurons) of the layer
        activationFunc:
            activation function (function after applying weights)
            of all the neurons in the layer. If None, defaults to
            the indentity function.
        bp:
            If true, enables backpropagation
        sampler:
            Random array generator. Takes keyword arguements "scale"
            and "size."

            scale:
                The standard deviation of the elements
            size:
                Size of random array.

    Methods:
        actvFunc
        applyWeight
        __call__
        train

    """
    def __init__(self, inSize, outSize, activationFunc = None, bp=False,\
                 sampler = np.random.normal):
        self._w = sampler(scale = np.sqrt(inSize)/inSize, size = inSize*outSize)\
                    .reshape(outSize, inSize)

        self._func = activationFunc
        self._diff = differentiate(activationFunc)
        self._bp = bp


    def actvFunc(self, inArr):
        """
        Apply the activation function to the array inArr.
        """
        if self._func != None:
            try:
                inArr = self._func(inArr)
            except TypeError:
                #Just fix it to None to skip the try next time.
                self._func = None
        #Default as "do nothing" (indentity function)
        return inArr

    def applyWeight(self, inArr):
        """
        Apply the weight matrix to the array inArr.
        """
        return dot(self._w, inArr)

    def __call__(self, inArr):
        """
        Fire the neuron layer with the input array inArr.
        """
        return self.actvFunc(self.applyWeight(inArr))

    def train(self, inArr, reinforcement = None, learnRate = 0.01):
        """
        Train the neuron based on the input array inArr.

        reinforcement can be an array, a callable, or left as None:
            as an array: This will be used as a target output in training.

            as a callable: This will be used as a reinforcement funtion, where
                            the output will be fed through and then
                            multiplied by the learning rate, where it will
                            then go through a typical unsupervised training
                            cycle with the newly modified learning rate.

            as None: Unsupervised training. Associations will arrise naturally
                    through the data based on the activation function.

        """
        if reinforcement == None:
            #Check if None
            return self._unsupervisedTrain(inArr, learnRate)
        else:
            try:
                iter(reinforcement)
                return self._trainCompareOut(inArr, reinforcement, learnRate)
            except TypeError:
                pass
            try:
                reinforcement.__call__
                return self._trainReinforceFunc(inArr, reinforcement, learnRate)
            except AttributeError:
                pass
        raise NotImplemented()

    def _unsupervisedTrain(self, inArr, learnRate):
        """
        Apply a Hebbian leaning cycle to the layer, then, if bp=True,
        return an adjusted input array that would
        produce approximately the same output given the new weights.
        """
        weighed_in = self.applyWeight(inArr)
        old_out = self.actvFunc(weighed_in)
        delta_w = learnRate * (np.outer(old_out, inArr-dot(self._w.T, old_out)))
        self._w = self._w + delta_w
        if self._bp:
            return np.dot(np.linalg.pinv(self._w), weighed_in)

    def _trainCompareOut(self, inArr, expOutArr, learnRate):
        """
        Apply a supervised learning cycle with an expected output.
        Then, if bp = True, return the input array to produce the
        result closest to the expected output of the trained layer.
        """
        weighed_in = self.applyWeight(inArr)
        point_diff = np.diag(self._diff(weighed_in))
        old_out = self.actvFunc(weighed_in)
        error = old_out - expOutArr
        pre_actv_error = np.dot(np.linalg.pinv(point_diff), error)
        delta_w = - learnRate * np.outer(pre_actv_error, inArr)
        self._w = self._w + delta_w
        if self._bp:
            weighed_approx = weighed_in - pre_actv_error
            return np.dot(np.linalg.pinv(self._w), weighed_approx)

    def _trainReinforceFunc(self, inArr, reinforcement, learnRate):
        """
        Train the layer based on a reinforcement function.
        """
        return self._unsupervisedTrain(inArr, reinforcement(self(inArr))*learnRate)

class NeuralNetwork:
    """
    Every neuralNetwork instance internally contains several NeuronLayers
    which vary depending on the parameters passed on initialization.

    layerInOutArr:
        An index (such as a list) of non-negative integers. (len > 1)
        Each sequential pair of numbers in the index denotes the inSize and
        outSize of a NeuronLayer in the network. E.G., [1, 2, 3] would mean
        that the network contained a NeuronLayer(1, 2), and a NeuronLayer(2,3),
        where the first layer feeds into the second.
    activationFuncs:
        Denotes the activation functions for each layer. This must be an index
        of length >= len(layerInOutArr)-1, or be a single function. (Defaults
        to the identity function)
    samplers:
        Random samplers for each layer. This must have
        length >= len(layerInOutArr)-1, or be a single function. (Defaults
        to the random normal distribution)
    """
    def __init__(self, layerInOutArr, activationFuncs = None,\
                 samplers = np.random.normal):
        self._numHiddenLayers = len(layerInOutArr)-1

        if activationFuncs == None:
            activationFuncs = [None]*self._numHiddenLayers
        try:
            len(activationFuncs)
        except TypeError:
            activationFuncs = [activationFuncs]*self._numHiddenLayers

        if samplers == None:
            samplers == [np.random.normal]*self._numHiddenLayers
        try:
            len(samplers)
        except TypeError:
            samplers = [samplers]*self._numHiddenLayers

        self._layers = [NeuronLayer(layerInOutArr[i],layerInOutArr[i+1],\
                                    activationFunc=activationFuncs[i],\
                                    bp=True, sampler=samplers[i])\
                        for i in range(self._numHiddenLayers)]

    def __call__(self, inArr):
        """
        Get an output from the network with the given input inArr.
        """
        ret = inArr
        for nl in self._layers:
            ret = nl(ret)
        return ret

    def train(self, inArr, reinforcement = None, learnRate = 0.01):
        """
        Train the network based on the input array inArr.

        reinforcement can be an array, a callable, or left as None:
            as an array: This will be used as a target output in training.

            as a callable: This will be used as a reinforcement funtion, where
                            the output will be fed through and then
                            multiplied by the learning rate, where it will
                            then go through a typical unsupervised training
                            cycle with the newly modified learning rate.

            as None: Unsupervised training. Associations will arrise naturally
                    through the data based on the activation function.

        """
        thisArr = inArr
        responses = [inArr]
        ##Go through all the layers except the last layer.
        for nl in self._layers[:-1]:
            thisArr = nl(thisArr)
            responses.append(thisArr)

        #Then, train the layers in reverse order.
        #Train returns the target input for the previous layer.
        self._layers.reverse()
        for nl in self._layers:
            reinforcement = nl.train(responses.pop(), reinforcement, learnRate)

        self._layers.reverse()
