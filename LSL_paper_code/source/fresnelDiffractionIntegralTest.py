#!/bin/env python3

# ==============================================================================
# This software developed by Stellar Science Ltd Co and the U.S. Government.
# Copyright (C) 2017 Stellar Science. Unlimited Government Rights.
# Warning: May contain EXPORT CONTROLLED, FOUO, ITAR, or sensitive information.
# ------------------------------------------------------------------------------

import fresnelDiffractionIntegral as fdi
import numpy as np
import scipy.special
import unittest

#import matplotlib.pyplot as plt

def _computeIntegralUsingRiemannSum(U, xGrid, yGrid, outputXGrid, outputYGrid, alpha):
    """Extremely simple algorithm, for debuging only."""

    fdi._checkTwoDimensionalArray(U, 2, "U", expectedDtype=np.complex_)
    fdi._checkOneDimensionalArrayTypeAndLength(xGrid, U.shape[0], "xGrid", expectedDtype=np.float_)
    fdi._checkArrayStrictlyIncreasing(xGrid, "xGrid")
    nx = xGrid.size
    xGrid = xGrid.flatten()
    fdi._checkOneDimensionalArrayTypeAndLength(yGrid, U.shape[1], "yGrid", expectedDtype=np.float_)
    fdi._checkArrayStrictlyIncreasing(yGrid, "yGrid")
    ny = yGrid.size
    yGrid = yGrid.flatten()
    fdi._checkOneDimensionalArrayTypeAndShape(outputXGrid, "outputXGrid", expectedDtype=np.float_)
    fdi._checkArrayStrictlyIncreasing(outputXGrid, "outputXGrid")
    outputNx = outputXGrid.size
    outputXGrid = outputXGrid.flatten()
    fdi._checkOneDimensionalArrayTypeAndShape(outputYGrid, "outputYGrid", expectedDtype=np.float_)
    fdi._checkArrayStrictlyIncreasing(outputYGrid, "outputYGrid")
    outputNy = outputYGrid.size
    outputYGrid = outputYGrid.flatten()

    outputU = np.zeros((outputNx, outputNy), dtype=np.complex_)

    for l in range(0, outputNx):
        for m in range(0, outputNy):
            outputU[l, m]= 0.
            for j in range(0, nx - 1):
                dx = xGrid[j+1] - xGrid[j]
                for k in range(0, ny - 1):
                    dy = yGrid[k+1] - yGrid[k]
                    outputU[l, m] += (
                        np.exp(-1.0j * np.pi / 2. * alpha**2
                               * ((xGrid[j] - outputXGrid[l])**2 + (yGrid[k] - outputYGrid[m])**2))
                        * U[j, k]
                        * dx * dy)
    outputU *= alpha**2 / 2.
    return outputU

def _exponentialDefiniteIntegral(limits, uPrime, alpha, c):
    """See analytic test cases section of the accompanying IPython notebook."""
    return (_exponentialIndefiniteIntegral(limits[1], uPrime, alpha, c)
            - _exponentialIndefiniteIntegral(limits[0], uPrime, alpha, c))

def _exponentialIndefiniteIntegral(u, uPrime, alpha, c):
    """See analytic test cases section of the accompanying IPython notebook."""
    (S, C) = scipy.special.fresnel(alpha * (u - uPrime) - c / alpha)
    return (
        1. / alpha * np.exp( 1.j * np.pi / 2. * (c**2 / alpha**2 + 2. * c * uPrime))
        * (C - 1.j * S))

def _quadraticDefiniteIntegral(limits, uPrime, alpha, c):
    """See analytic test cases section of the accompanying IPython notebook."""
    return (_quadraticIndefiniteIntegral(limits[1], uPrime, alpha, c)
            - _quadraticIndefiniteIntegral(limits[0], uPrime, alpha, c))

def _quadraticIndefiniteIntegral(u, uPrime, alpha, c):
    """See analytic test cases section of the accompanying IPython notebook."""
    (S, C) = scipy.special.fresnel(alpha * (u - uPrime))
    return (
        (1.j / (np.pi * alpha**2)
         * (c[1] + c[2] * (u + uPrime))
         * np.exp(-1.j * np.pi / 2. * alpha**2 * (u - uPrime)**2))
        + (1. / alpha
           * (c[0] + c[1] * uPrime + c[2] * uPrime**2 - 1.j * c[2] / (np.pi * alpha**2))
           * (C - 1.j * S)))

def _maxErrorRelativeToTargetRMS(value, targetValue):
    rms = np.sqrt(np.mean(np.abs(targetValue)**2))
    return np.max(np.abs(value - targetValue) / rms)

# def _makeComparisonPlot(matrix1, label1, matrix2, label2):
#         plt.subplot(321)
#         CS1 = plt.contour(np.real(matrix1))
#         plt.clabel(CS1, inline=1, fontsize=10)
#         plt.title("Re({})".format(label1))
#         plt.subplot(322)
#         CS2 = plt.contour(np.imag(matrix1))
#         plt.clabel(CS2, inline=1, fontsize=10)
#         plt.title("Im({})".format(label1))

#         plt.subplot(323)
#         CS3 = plt.contour(np.real(matrix2))
#         plt.clabel(CS3, inline=1, fontsize=10)
#         plt.title("Re({})".format(label2))
#         plt.subplot(324)
#         CS4 = plt.contour(np.imag(matrix2))
#         plt.clabel(CS4, inline=1, fontsize=10)
#         plt.title("Im({})".format(label2))

#         plt.subplot(325)
#         CS5 = plt.contour(np.real(matrix1 - matrix2))
#         plt.clabel(CS5, inline=1, fontsize=10)
#         plt.title("Re({}-{})".format(label1, label2))
#         plt.subplot(326)
#         CS6 = plt.contour(np.imag(matrix1 - matrix2))
#         plt.clabel(CS6, inline=1, fontsize=10)
#         plt.title("Im({}-{})".format(label1, label2))

#         plt.show()

class TestLinearFunctionsWhereInterpolationIsExact(unittest.TestCase):

    def testAllZeros(self):
        nx = 32
        ny = 16
        xGrid = np.linspace(0, 1, nx)
        yGrid = np.linspace(0, 1, ny)
        U = np.zeros((nx, ny), dtype=np.complex_)
        alpha = 1.0
        numericUPrime = fdi.computeIntegral(U, xGrid, yGrid, xGrid, yGrid, alpha)
        self.assertTrue(np.array_equal(numericUPrime, np.zeros((nx, ny), dtype=np.complex_)))

    def _testSmallQuadratic(self, cx, cy, alpha = 2.0):
        nx = 25
        ny = 15
        xGrid = np.linspace(0, 1, nx)
        yGrid = np.linspace(0, 1, ny)
        outputXGrid = xGrid
        outputYGrid = yGrid
        alpha = 2.0
        U = np.zeros((nx, ny), dtype=np.complex_)
        U += np.outer(
            cx[0] + cx[1] * xGrid + cx[2] * xGrid**2,
            cy[0] + cy[1] * yGrid + cy[2] * yGrid**2)
        analyticUPrime = (
            alpha**2 / 2.
            * np.outer(
                _quadraticDefiniteIntegral([np.min(xGrid), np.max(xGrid)], outputXGrid, alpha, cx),
                _quadraticDefiniteIntegral([np.min(yGrid), np.max(yGrid)], outputYGrid, alpha, cy)))
        numericUPrime = fdi.computeIntegral(U, xGrid, yGrid, outputXGrid, outputYGrid, alpha)
        #_makeComparisonPlot(analyticUPrime, "analytic", numericUPrime, "numeric")
        self.assertLess(_maxErrorRelativeToTargetRMS(numericUPrime, analyticUPrime), 1e-6)

    def testConstant(self):
        self._testSmallQuadratic([1., 0., 0], [1., 0., 0])

    def testLinearInX(self):
        self._testSmallQuadratic([1., 3., 0], [1., 0., 0])

    def testLinearInY(self):
        self._testSmallQuadratic([1., 0, 0], [1., 3., 0])

    def testBilinear1(self):
        self._testSmallQuadratic([1., -2, 0], [5., 3., 0])

    def testBilinear2(self):
        self._testSmallQuadratic([5.j, -2, 0], [-4., 3.j, 0])

    def testUnequalGridSpacing(self):
        cx = [1., 2., 0]
        cy = [1., -3., 0]
        nx = 25
        ny = 15
        xGrid = np.linspace(0, 1, nx)**2
        yGrid = np.linspace(0, 1, ny)**2
        outputXGrid = xGrid
        outputYGrid = yGrid
        alpha = 2.0
        U = np.zeros((nx, ny), dtype=np.complex_)
        U += np.outer(
            cx[0] + cx[1] * xGrid + cx[2] * xGrid**2,
            cy[0] + cy[1] * yGrid + cy[2] * yGrid**2)
        analyticUPrime = (
            alpha**2 / 2.
            * np.outer(
                _quadraticDefiniteIntegral([np.min(xGrid), np.max(xGrid)], outputXGrid, alpha, cx),
                _quadraticDefiniteIntegral([np.min(yGrid), np.max(yGrid)], outputYGrid, alpha, cy)))
        numericUPrime = fdi.computeIntegral(U, xGrid, yGrid, outputXGrid, outputYGrid, alpha)
        #_makeComparisonPlot(analyticUPrime, "analytic", numericUPrime, "numeric")
        self.assertLess(_maxErrorRelativeToTargetRMS(numericUPrime, analyticUPrime), 1e-6)

    def testRangeOfAlphas(self):
        for alpha in [0.001, 0.01, 0.1, 1., 10., 100., 1000.]:
            self._testSmallQuadratic([5.j, -2, 0], [-4., 3.j, 0], alpha=alpha)

    def testLargeOutputGrid(self):
        cx = [-1.j, 2.j, 0]
        cy = [1.j, -3.j, 0]        
        nx = 25
        ny = 15
        xGrid = np.linspace(-1, 1, nx)
        yGrid = np.linspace(-1, 1, ny)
        outputXGrid = np.linspace(-10, 10, 2*nx)
        outputYGrid = np.linspace(-15, 15, 2*ny)
        alpha = 2.0
        U = np.zeros((nx, ny), dtype=np.complex_)
        U += np.outer(
            cx[0] + cx[1] * xGrid + cx[2] * xGrid**2,
            cy[0] + cy[1] * yGrid + cy[2] * yGrid**2)
        analyticUPrime = (
            alpha**2 / 2.
            * np.outer(
                _quadraticDefiniteIntegral([np.min(xGrid), np.max(xGrid)], outputXGrid, alpha, cx),
                _quadraticDefiniteIntegral([np.min(yGrid), np.max(yGrid)], outputYGrid, alpha, cy)))
        numericUPrime = fdi.computeIntegral(U, xGrid, yGrid, outputXGrid, outputYGrid, alpha)
        self.assertLess(_maxErrorRelativeToTargetRMS(numericUPrime, analyticUPrime), 1e-6)        

    def testLargeInputGrid(self):
        cx = [-1.j, 2.j, 0]
        cy = [1.j, -3.j, 0]
        outputNx = 25
        outputNy = 15
        nx = 2 * outputNx
        ny = 2 * outputNy
        xGrid= np.linspace(-10, 10, nx)
        yGrid = np.linspace(-15, 15, ny)        
        outputXGrid = np.linspace(-1, 1, outputNx)
        outputYGrid = np.linspace(-1, 1, outputNy)
        alpha = 2.0
        U = np.zeros((nx, ny), dtype=np.complex_)
        U += np.outer(
            cx[0] + cx[1] * xGrid + cx[2] * xGrid**2,
            cy[0] + cy[1] * yGrid + cy[2] * yGrid**2)
        analyticUPrime = (
            alpha**2 / 2.
            * np.outer(
                _quadraticDefiniteIntegral([np.min(xGrid), np.max(xGrid)], outputXGrid, alpha, cx),
                _quadraticDefiniteIntegral([np.min(yGrid), np.max(yGrid)], outputYGrid, alpha, cy)))
        numericUPrime = fdi.computeIntegral(U, xGrid, yGrid, outputXGrid, outputYGrid, alpha)
        self.assertLess(_maxErrorRelativeToTargetRMS(numericUPrime, analyticUPrime), 1e-6) 

class TestConvergenceOnNonlinearFunctions(unittest.TestCase):

    def testConvergenceOnQuadratic(self):
        def getRelativeError(n):
            nx = n
            ny = n
            xGrid = np.linspace(-0.5, 0.5, nx)
            yGrid = np.linspace(-0.5, 0.5, ny)
            outputXGrid = xGrid
            outputYGrid = yGrid
            alpha = 1.
            cx = [-1., 0., 1.]
            cy = [1., 0., -2.]
            U = np.zeros((nx, ny), dtype=np.complex_)
            U += np.outer(
                cx[0] + cx[1] * xGrid + cx[2] * xGrid**2,
                cy[0] + cy[1] * yGrid + cy[2] * yGrid**2)
            analyticUPrime = (
                alpha**2 / 2.
                * np.outer(
                    _quadraticDefiniteIntegral([np.min(xGrid), np.max(xGrid)], outputXGrid, alpha, cx),
                    _quadraticDefiniteIntegral([np.min(yGrid), np.max(yGrid)], outputYGrid, alpha, cy)))
            numericUPrime = fdi.computeIntegral(U, xGrid, yGrid, outputXGrid, outputYGrid, alpha)
            #_makeComparisonPlot(analyticUPrime, "analytic", numericUPrime, "numeric")
            return _maxErrorRelativeToTargetRMS(numericUPrime, analyticUPrime)

        relativeErrors = [getRelativeError(n) for n in [10, 20, 50, 100, 500]]
        for i in range(0, len(relativeErrors) - 1):
            self.assertLess(relativeErrors[i+1], relativeErrors[i])
        self.assertLess(relativeErrors[-1], 1e-4)

    def testConvergenceOnExponential(self):
        def getRelativeError(n):
            xGrid = np.linspace(0, 1, n)
            yGrid = np.linspace(0, 1, n)
            outputXGrid = xGrid
            outputYGrid = yGrid
            cx = 1.
            cy = 2.
            U = np.outer(np.exp(1.j * np.pi * cx * xGrid), np.exp(1.j * np.pi * cy * yGrid))
            alpha = 2.0
            analyticUPrime = (
                alpha**2 / 2.
                * np.outer(
                    _exponentialDefiniteIntegral([np.min(xGrid), np.max(xGrid)], outputXGrid, alpha, cx),
                    _exponentialDefiniteIntegral([np.min(yGrid), np.max(yGrid)], outputYGrid, alpha, cy)))
            numericUPrime = fdi.computeIntegral(U, xGrid, yGrid, outputXGrid, outputYGrid, alpha)
            #_makeComparisonPlot(analyticUPrime, "analytic", numericUPrime, "numeric")
            return _maxErrorRelativeToTargetRMS(numericUPrime, analyticUPrime)

        relativeErrors = [getRelativeError(n) for n in [10, 20, 50, 100, 500]]
        for i in range(0, len(relativeErrors) - 1):
            self.assertLess(relativeErrors[i+1], relativeErrors[i])
        self.assertLess(relativeErrors[-1], 1e-4)

class TestCheckArrayStrictlyIncreasing(unittest.TestCase):

    def testCorrectFlatArray(self):
        fdi._checkArrayStrictlyIncreasing(np.array([5, 6, 7]), "testarray")

    def testCorrectColumnVector(self):
        fdi._checkArrayStrictlyIncreasing(np.array([5, 6, 7]).reshape((3,1)), "testarray")

    def testCorrectRowVector(self):
        fdi._checkArrayStrictlyIncreasing(np.array([5, 6, 7]).reshape((3,1)), "testarray")

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")
    def testStrictlyDecreasing(self):
        with self.assertRaisesRegex(ValueError, "is not strictly increasing"):
            fdi._checkArrayStrictlyIncreasing(np.arange(100, 0, -1), "testarray")

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")            
    def testMixedUpOrder(self):
        with self.assertRaisesRegex(ValueError, "is not strictly increasing"):
            fdi._checkArrayStrictlyIncreasing(np.array([7, 5, 6, 9]), "testarray")

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")
    def testDuplicates(self):
        with self.assertRaisesRegex(ValueError, "is not strictly increasing"):
            fdi._checkArrayStrictlyIncreasing(np.array([5, 5, 6, 6, 7, 7]), "testarray")

class TestCheckOneDimensionalArrayTypeAndLength(unittest.TestCase):

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")
    def testWrongDataType(self):
        with self.assertRaisesRegex(ValueError, "does not match expected dtype"):
            fdi._checkOneDimensionalArrayTypeAndLength(
                np.array([1.1, 2.2, 3.3]), 3, "testarray", expectedDtype=np.int_)

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")            
    def testWrongDimension(self):
        with self.assertRaisesRegex(ValueError, "wrong dimension"):
            fdi._checkOneDimensionalArrayTypeAndLength(np.zeros((3, 3, 3), dtype=np.float_), 2, "testarray")

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")
    def testWrongLength(self):
        with self.assertRaisesRegex(ValueError, "does not match expected length"):
            fdi._checkOneDimensionalArrayTypeAndLength(np.array([7.0, 8.0, 9.0]), 2, "testarray")

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")            
    def testWrongShape(self):
        with self.assertRaisesRegex(ValueError, "is not compatible"):
            fdi._checkOneDimensionalArrayTypeAndLength(np.zeros((3, 1),  dtype=np.float_), 2, "testarray")
        with self.assertRaisesRegex(ValueError, "is not compatible"):
            fdi._checkOneDimensionalArrayTypeAndLength(np.zeros((1, 3), dtype=np.float_), 2, "testarray")

    def testCorrectColumnVector(self):
        fdi._checkOneDimensionalArrayTypeAndLength(np.zeros((3, 1), dtype=np.float_), 3, "testarray")

    def testCorrectRowVector(self):
        fdi._checkOneDimensionalArrayTypeAndLength(np.zeros((1, 3), dtype=np.float_), 3, "testarray")

    def testCorrectFlatArray(self):
        fdi._checkOneDimensionalArrayTypeAndLength(np.array([7.0, 8.0 , 9.0]), 3, "testarray")

class TestCheckOneDimensionalArrayTypeAndShape(unittest.TestCase):

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")    
    def testWrongDataType(self):
        with self.assertRaisesRegex(ValueError, "does not match expected dtype"):
            fdi._checkOneDimensionalArrayTypeAndShape(
                np.array([1.1, 2.2, 3.3]), "testarray", expectedDtype=np.int_)

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")            
    def testWrongDimension(self):
        with self.assertRaisesRegex(ValueError, "wrong dimension"):
            fdi._checkOneDimensionalArrayTypeAndShape(np.zeros((3, 3, 3), dtype=np.float_), "testarray")

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")            
    def testWrongShape(self):
        with self.assertRaisesRegex(ValueError, "is not compatible"):
            fdi._checkOneDimensionalArrayTypeAndShape(np.zeros((3, 2),  dtype=np.float_), "testarray")
        with self.assertRaisesRegex(ValueError, "is not compatible"):
            fdi._checkOneDimensionalArrayTypeAndShape(np.zeros((2, 3), dtype=np.float_), "testarray")

    def testCorrectColumnVector(self):
        fdi._checkOneDimensionalArrayTypeAndShape(np.zeros((3, 1), dtype=np.float_), "testarray")

    def testCorrectRowVector(self):
        fdi._checkOneDimensionalArrayTypeAndShape(np.zeros((1, 3), dtype=np.float_), "testarray")

    def testCorrectFlatArray(self):
        fdi._checkOneDimensionalArrayTypeAndShape(np.array([7.0, 8.0 , 9.0]), "testarray")


class TestCheckTwoDimensionalArray(unittest.TestCase):

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")
    def testWrongDataType(self):
        with self.assertRaisesRegex(ValueError, "does not match expected dtype"):
            fdi._checkTwoDimensionalArray(
                np.zeros((3, 3), dtype=np.float_), 2, "testarray", expectedDtype=np.int_)

    @unittest.skipUnless(hasattr(unittest.TestCase, "assertRaisesRegex"), "assertRaisesRegex not supported")
    def testWrongDimension(self):
        with self.assertRaisesRegex(ValueError, "wrong dimension"):
            fdi._checkTwoDimensionalArray(np.zeros((3, 3, 3), dtype=np.float_), 2, "testarray")

    def testCorrectArray(self):
        fdi._checkTwoDimensionalArray(np.zeros((3, 3), dtype=np.float_), 2, "testarray")
