import numpy as np
import os

import pydensecrf.densecrf as dcrf

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def crf_refine_(prediction):
    # print('shape of pred: ',prediction.shape)
    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(prediction.shape[1], prediction.shape[0], M)

    n_energy = -np.log((1.0 - prediction + EPSILON)) / (tau * _sigmoid(1 - prediction))
    p_energy = -np.log(prediction + EPSILON) / (tau * _sigmoid(prediction))

    U = np.zeros((M, prediction.shape[0] * prediction.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    # d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=prediction, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    # res = res * 255
    res = res.reshape(prediction.shape[:2])
    return res