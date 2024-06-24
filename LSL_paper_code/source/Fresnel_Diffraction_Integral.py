#!/bin/env python3

# ==============================================================================
# This software developed by Stellar Science Ltd Co and the U.S. Government.
# Copyright (C) 2017 Stellar Science. Unlimited Government Rights.
# Warning: May contain EXPORT CONTROLLED, FOUO, ITAR, or sensitive information.
# ------------------------------------------------------------------------------
"""
Functions for computing the Fresnel diffraction integral using a direct method.

See the accompanying IPython notebook for additional details.
"""

import collections
from contextlib import contextmanager
import numpy as np
import scipy.special
import time

Timer = collections.namedtuple('Timer', ['definiteIntegral', 'innerSum', 'outerSum'])

@contextmanager
def _accumulateTime(timeList):
    startTime = time.time()
    yield
    timeList.append(time.time() - startTime)


def computeIntegral(U, xGrid, yGrid, outputXGrid, outputYGrid, alpha, timer=Timer([], [], [])):
    """Compute the integral in N**3 steps with N**2 storage, using a single thread.

    Inputs:
        U: a complex nx*ny numpy array giving the sampled field values on the input grid
        xGrid: a real nx*1 numpy array giving grid point x locations for U
        yGrid: a real ny*1 numpy array giving grid point y locations for U
        outputXGrid: a real outputNx*1 numpy array giving output grid point x locations
        outputYGrid: a real outputNy*1 numpy array giving output grid point y locations
        alpha: a real scalar (see IPython notebook for definition)

    Returns: a complex outputNx*outputNy numpy array of field values on the output grid
    """

    _checkTwoDimensionalArray(U, 2, "U", expectedDtype=np.complex_)
    _checkOneDimensionalArrayTypeAndLength(xGrid, U.shape[0], "xGrid", expectedDtype=np.float_)
    _checkArrayStrictlyIncreasing(xGrid, "xGrid")
    nx = xGrid.size
    xGrid = xGrid.reshape((nx, 1))
    _checkOneDimensionalArrayTypeAndLength(yGrid, U.shape[1], "yGrid", expectedDtype=np.float_)
    _checkArrayStrictlyIncreasing(yGrid, "yGrid")
    ny = yGrid.size
    yGrid = yGrid.reshape((ny, 1))
    _checkOneDimensionalArrayTypeAndShape(outputXGrid, "outputXGrid", expectedDtype=np.float_)
    _checkArrayStrictlyIncreasing(outputXGrid, "outputXGrid")
    outputNx = outputXGrid.size
    outputXGrid = outputXGrid.reshape((outputNx, 1))
    _checkOneDimensionalArrayTypeAndShape(outputYGrid, "outputYGrid", expectedDtype=np.float_)
    _checkArrayStrictlyIncreasing(outputYGrid, "outputYGrid")
    outputNy = outputYGrid.size
    outputYGrid = outputYGrid.reshape((outputNy, 1))

    outputU = np.zeros((outputXGrid.size, outputYGrid.size), dtype=np.complex_)

    def expTermOfIndefiniteIntegral(uMinusB):
        """The exp() term of "f" (see IPython notebook)."""
        return 1.j / (np.pi * alpha**2) * np.exp(-1.j * np.pi / 2. * alpha**2 * (uMinusB)**2)

    def fresnelTermOfIndefiniteIntegral(uMinusB):
        """The C and S terms of "f", with (gamma - beta) factored out (see IPython notebook)."""
        (sTable, cTable) = scipy.special.fresnel(alpha * uMinusB)
        return 1. / alpha * (cTable - 1.j * sTable)

    # This code is common to the "x" and "y" directions.  Here, "w" stands for either x or y.
    def tabulateDefiniteIntegral(wGrid, outputWGrid):
        nw = wGrid.size
        outputNw = outputWGrid.size
        
        wMinusOutputW = np.tile(wGrid.T, (outputNw, 1)) - np.tile(outputWGrid, (1, nw))
        fresnelTerm = fresnelTermOfIndefiniteIntegral(wMinusOutputW)
        expTerm = expTermOfIndefiniteIntegral(wMinusOutputW)
        inverseDeltaW = (1. / (wGrid[1:] - wGrid[:-1])).reshape((nw-1))        

        definiteIntegral = np.zeros((2, outputNw, nw-1), dtype=np.complex_)

        definiteIntegral[0, :, :] += fresnelTerm[:, 1:]
        definiteIntegral[0, :, :] -= fresnelTerm[:, :-1]
        definiteIntegral[0, :, :] *= wMinusOutputW[:, 1:]
        definiteIntegral[0, :, :] -= expTerm[:, 1:]
        definiteIntegral[0, :, :] += expTerm[:, :-1]

        definiteIntegral[1, :, :] -= fresnelTerm[:, 1:]
        definiteIntegral[1, :, :] += fresnelTerm[:, :-1]
        definiteIntegral[1, :, :] *= wMinusOutputW[:, :-1]
        definiteIntegral[1, :, :] += expTerm[:, 1:]
        definiteIntegral[1, :, :] -= expTerm[:, :-1]
        
        # absorb the "delta w" factor, which only depends on the outermost (0 .. nw-1) index.
        definiteIntegral *= inverseDeltaW
        
        return definiteIntegral

    with _accumulateTime(timer.definiteIntegral):
        definiteIntegralOverY = tabulateDefiniteIntegral(yGrid, outputYGrid)

    with _accumulateTime(timer.innerSum):
        # Tabulate the inner sum, which is used repeatedly later, to get the overall N**3 complexity.
        innerSum = np.zeros((2, outputNy, nx-1), order='C', dtype=np.complex_)
        for j in range(0, nx-1):
            innerSum[0, :, j] += np.dot(definiteIntegralOverY[0, :, :], U[j, :-1])
            innerSum[0, :, j] += np.dot(definiteIntegralOverY[1, :, :], U[j, 1:])
            innerSum[1, :, j] += np.dot(definiteIntegralOverY[0, :, :], U[j+1, :-1])
            innerSum[1, :, j] += np.dot(definiteIntegralOverY[1, :, :], U[j+1, 1:])

    with _accumulateTime(timer.outerSum):
        # Now calculate outer sum
        definiteIntegralOverX = tabulateDefiniteIntegral(xGrid, outputXGrid)
        for l in range(0, outputNx):
            outputU[l, :] += np.dot(innerSum[0, :, :], definiteIntegralOverX[0, l, :])
            outputU[l, :] += np.dot(innerSum[1, :, :], definiteIntegralOverX[1, l, :])

    outputU *= alpha**2 / 2.
    return outputU


def _computeIntegralUsingNaiveAlgorithm(U, xGrid, yGrid, outputXGrid, outputYGrid, alpha):
    """Compute the integral in N**4 steps, closely following the formulation in the IPython notebook.

    Intended for testing purposes only.
    """

    _checkTwoDimensionalArray(U, 2, "U", expectedDtype=np.complex_)
    _checkOneDimensionalArrayTypeAndLength(xGrid, U.shape[0], "xGrid", expectedDtype=np.float_)
    _checkArrayStrictlyIncreasing(xGrid, "xGrid")
    nx = xGrid.size
    xGrid = xGrid.flatten()
    _checkOneDimensionalArrayTypeAndLength(yGrid, U.shape[1], "yGrid", expectedDtype=np.float_)
    _checkArrayStrictlyIncreasing(yGrid, "yGrid")
    ny = yGrid.size
    yGrid = yGrid.flatten()
    _checkOneDimensionalArrayTypeAndShape(outputXGrid, "outputXGrid", expectedDtype=np.float_)
    _checkArrayStrictlyIncreasing(outputXGrid, "outputXGrid")
    outputNx = outputXGrid.size
    outputXGrid = outputXGrid.flatten()
    _checkOneDimensionalArrayTypeAndShape(outputYGrid, "outputYGrid", expectedDtype=np.float_)
    _checkArrayStrictlyIncreasing(outputYGrid, "outputYGrid")
    outputNy = outputYGrid.size
    outputYGrid = outputYGrid.flatten()

    outputU = np.zeros((outputNx, outputNy), dtype=np.complex_)

    def f(u, b, c):
        (S, C) = scipy.special.fresnel(alpha * (u - b))
        result = (
            (1.j / (np.pi * alpha**2)
             * np.exp(-1.j * np.pi / 2. * alpha**2 * (u - b)**2))
            - (c - b) / alpha * (C - 1.j * S))
        if np.isnan(result):
            raise ValueError("f({}, {}, {}) is nan".format(u, b, c))
        return result

    for l in range(0, outputNx):
        for m in range(0, outputNy):
            for j in range(0, nx - 1):
                dx = xGrid[j+1] - xGrid[j]

                fx = np.zeros((2,), dtype=np.complex_)
                fx[0] -= f(xGrid[j+1], outputXGrid[l], xGrid[j+1])
                fx[0] += f(xGrid[j  ], outputXGrid[l], xGrid[j+1])
                fx[1] += f(xGrid[j+1], outputXGrid[l], xGrid[j  ])
                fx[1] -= f(xGrid[j  ], outputXGrid[l], xGrid[j  ])
                
                for k in range(0, ny - 1):
                    dy = yGrid[k+1] - yGrid[k]

                    fy = np.zeros((2,), dtype=np.complex_)
                    fy[0] -= f(yGrid[k+1], outputYGrid[m], yGrid[k+1])
                    fy[0] += f(yGrid[k  ], outputYGrid[m], yGrid[k+1])
                    fy[1] += f(yGrid[k+1], outputYGrid[m], yGrid[k])
                    fy[1] -= f(yGrid[k  ], outputYGrid[m], yGrid[k])

                    outputU[l, m] += (fx[0] * U[j  , k  ] * fy[0]) / (dx * dy)
                    outputU[l, m] += (fx[0] * U[j  , k+1] * fy[1]) / (dx * dy)
                    outputU[l, m] += (fx[1] * U[j+1, k  ] * fy[0]) / (dx * dy)
                    outputU[l, m] += (fx[1] * U[j+1, k+1] * fy[1]) / (dx * dy)

    outputU *= alpha**2 / 2.
    return outputU

def _checkArrayDtype(array, arrayName, expectedDtype):
    """Raises a ValueError exception if an array has the wrong dtype."""
    if array.dtype != expectedDtype:
        raise ValueError("Array {}: dtype {} does not match expected dtype {}".format(
            arrayName, array.dtype, expectedDtype))

def _checkArrayStrictlyIncreasing(array, arrayName):
    """Raises a ValueError if a an array, assumed one-dimensional, is not strictly increasing"""
    if not np.array_equal(array.flatten(), np.unique(array.flatten())):
        raise ValueError("Array {} is not strictly increasing".format(arrayName))

def _checkOneDimensionalArrayTypeAndLength(array, expectedLength, arrayName, expectedDtype=np.float_):
    """Raises a ValueError exception if a one-dimensional array has the wrong dtype or is not compatible with the expected length."""
    _checkArrayDtype(array, arrayName, expectedDtype)
    if array.ndim == 1:
        if array.size != expectedLength:
            raise ValueError("Array {}: length {} does not match expected length {}".format(
                arrayName, array.size, expectedLength))
    elif array.ndim == 2:
        if (array.shape != (expectedLength, 1)) and (array.shape != (1, expectedLength)):
            raise ValueError("Array {} of shape {} is not compatible with a 1-d array of length {}".format(
                arrayName, array.shape, expectedLength))
    else:
        raise ValueError("Array {} has wrong dimension: {}".format(arrayName, array.ndim))

def _checkOneDimensionalArrayTypeAndShape(array, arrayName, expectedDtype=np.float_):
    """Raises a ValueError exception if an array has the wrong dtype or a shape that is not compatible with a one-dimensional vector."""
    _checkArrayDtype(array, arrayName, expectedDtype)
    if array.ndim == 1:
        pass
    elif array.ndim == 2:
        if (array.shape[0] != 1) and (array.shape[1] != 1):
            raise ValueError("Array {} of shape {} is not compatible with a 1-d array".format(
                arrayName, array.shape))
    else:
        raise ValueError("Array {} has wrong dimension: {}".format(arrayName, array.ndim))

def _checkTwoDimensionalArray(array, expectedNumberOfDimensions, arrayName, expectedDtype=np.float_):
    """Raises a ValueError exception if a two-dimensional array has the wrong dtype or wrong number of dimensions."""
    _checkArrayDtype(array, arrayName, expectedDtype)
    if array.ndim != 2:
        raise ValueError("Array {} has wrong dimension: {}".format(arrayName, array.ndim))
