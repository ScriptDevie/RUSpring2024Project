# imports
import numpy as np
from numpy.random import multivariate_normal as np_multivariate_normal
from scipy.stats import multivariate_normal as sp_multivariate_normal
from numpy import linalg as np_linalg
from scipy import linalg as sp_linalg




class Circulant_Embedding_Generator:
    def __init__(self,xPoints,
                      yPoints,
                      epsilon,
                      indexVarianceScaling):
        """set all dimensional parameters and calculate the nondimensional parameters"""
    
        self.xPoints                    = xPoints       
        self.numberOfXPoints            = xPoints.size
        self.numberOfRows               = 2*self.numberOfXPoints-1 
        
        self.yPoints                    = yPoints
        self.numberOfYPoints            = yPoints.size
        self.numberOfElementsInRows     = 2*self.numberOfYPoints-1
        
        
#        self.correlationMatrix          = np.empty([self.numberOfXPoints,
#                                                    self.numberOfXPoints,
#                                                    self.numberOfYPoints,
#                                                    self.numberOfYPoints])
#    
#        self.correlationMatrix2d        = np.empty([self.numberOfXPoints*self.numberOfYPoints,
#                                                    self.numberOfXPoints*self.numberOfYPoints])
#        
#        self.circulantMatrix            = np.empty([self.numberOfRows,
#                                                    self.numberOfRows,
#                                                    self.numberOfElementsInRows,
#                                                    self.numberOfElementsInRows])
#    
#        self.circulantMatrix2d          = np.empty([self.numberOfRows*self.numberOfElementsInRows,
#                                                    self.numberOfRows*self.numberOfElementsInRows]) 

        
        self.epsilon              = epsilon
        self.indexVarianceScaling = indexVarianceScaling

        self.matrixOfRows  = np.zeros([self.numberOfRows,self.numberOfElementsInRows]) 
        
        self.xLocations = np.vstack( (np.hstack( (np.zeros(self.numberOfXPoints,dtype=np.int64),
                                                  np.arange(self.numberOfXPoints-1,0,-1))
                                               ),
                                      np.hstack( (np.arange(0,self.numberOfXPoints,1),
                                                  np.zeros(self.numberOfXPoints-1, dtype=np.int64))
                                               ),
                                      np.arange(self.numberOfRows))
                                   ).T
        
        self.yLocations = np.vstack( (np.hstack( (np.zeros(self.numberOfYPoints,dtype=np.int64),
                                                  np.arange(self.numberOfYPoints-1,0,-1))
                                               ),
                                      np.hstack( (np.arange(0,self.numberOfYPoints,1),
                                                  np.zeros(self.numberOfYPoints-1,dtype=np.int64))
                                               ),
                                      np.arange(self.numberOfElementsInRows))         
                                   ).T

        self.rowsOfBCCB = np.empty([self.numberOfRows,self.numberOfElementsInRows],dtype=np.float64)
        self.GetRowsOfBCCB()  
        self.GetEigenvalues()
        
        #self.CorrelationMatrix()
        #self.CirculantMatrix()
        
        
        #self.meshPointsX, self.meshPointsY  = np.meshgrid(self.pointsX,self.pointsY)


        #self.variance                   = variance
        #self.correlationFlag            = correlationFlag
        #self.numberOfRealizationBatches = 0
        #self.seeds                      = []


        #self.CORRELATION_MATRIX()
        #self.ROW_COLUMN_MATRIX()
        #self.EIGENS()
        return None
    
    def CorrelationFunction(self,x1,x2,y1,y2):
        radiusSquared = (x2-x1)**2.0 + (y2-y1)**2.0
        radius        = np.sqrt(radiusSquared)
        if radius < 1:
            correlationValue = 1.0 - ((self.epsilon/self.indexVarianceScaling)**(2/3)) * radiusSquared
        else:
            correlationValue = 1.0 - ((self.epsilon/self.indexVarianceScaling)**(2/3)) * radiusSquared**(1/3)

        return correlationValue
    
    def GetRowsOfBCCB(self):
        for xLocation in self.xLocations:
            for yLocation in self.yLocations:
                self.rowsOfBCCB[xLocation[2],
                                yLocation[2]] = self.CorrelationFunction(self.xPoints[xLocation[0]],
                                                                         self.xPoints[xLocation[1]],
                                                                         self.yPoints[yLocation[0]],
                                                                         self.yPoints[yLocation[1]])
        return None 
                
    def CorrelationMatrix(self):
        for indexX1 in range(self.numberOfXPoints):
            for indexX2 in range(self.numberOfXPoints):
                for indexY1 in range(self.numberOfYPoints):
                    for indexY2 in range(self.numberOfYPoints):
                        self.correlationMatrix[indexX1,
                                               indexX2,
                                               indexY1,
                                               indexY2]  = self.CorrelationFunction(self.xPoints[indexX1],
                                                                                    self.xPoints[indexX2],
                                                                                    self.yPoints[indexY1],
                                                                                    self.yPoints[indexY2])
        for indexX1 in range(self.numberOfXPoints):
            for indexX2 in range(self.numberOfXPoints):  
                self.correlationMatrix2d[indexX1*self.numberOfYPoints:((indexX1+1)*self.numberOfYPoints),
                                         indexX2*self.numberOfYPoints:((indexX2+1)*self.numberOfYPoints)] = self.correlationMatrix[indexX1,
                       indexX2,
                       :,
                       :]

        return None  

    def CirculantMatrix(self):
        #self.firstRowOfBlocks = np.empty((self.numberOfRows,
        #                                  self.numberOfElementsInRows,
        #                                  self.numberOfElementsInRows))
        self.firstRowOfBlocks = MakeCirculantMatrix(self.rowsOfBCCB[0,:])
        for indexX in range(1,self.numberOfRows):
            self.firstRowOfBlocks = np.hstack((self.firstRowOfBlocks,MakeCirculantMatrix(self.rowsOfBCCB[indexX,:])))
       
        self.BCCBMatrix = MakeBCCBMatrix(self.firstRowOfBlocks)

        return None                 
    

    
    
    def GetEigenvectors(self):
        fourierMatrixSizeOfBlocks = np.fft.fft(np.eye(self.numberOfElementsInRows))
        fourierMatrixNumberOfBlocks = np.fft.fft(np.eye(self.numberOfRows))
        self.eigenvectors = (1.0/np.sqrt(self.numberOfRows*self.numberOfElementsInRows)) *np.kron(fourierMatrixNumberOfBlocks,fourierMatrixSizeOfBlocks)

        
        
    def GetEigenvalues(self):
        self.eigenvaluesMatrix = np.real(np.fft.fft2(self.rowsOfBCCB))
        negativeEigenvalues = self.eigenvaluesMatrix[self.eigenvaluesMatrix < 0]
        if negativeEigenvalues.size is 0:
            print('Exact Matrix Used')
        else:
            self.largestNegativeEigenvalue = np.amax(np.abs(negativeEigenvalues))
            print('largest negative eigenvalue:',self.largestNegativeEigenvalue)

        #self.largestNegativeEigenvalue = np.amax(np.abs(negativeEigenvalues))
        self.eigenvaluesMatrix[self.eigenvaluesMatrix < 0] = 0.0
        self.sqrtOfEigenvalues = np.sqrt(self.eigenvaluesMatrix)
        #self.eigenvaluesVector = np.reshape(self.eigenvaluesMatrix,(self.numberOfElementsInRows*self.numberOfRows,1))
       
        return None     

    def GetEigenVectorsAndValues(self):
        self.evals, self.evecs = np_linalg.eig(self.BCCBMatrix)

        return None 
    
    def PlotBCCBMatrix(self):
        plt.imshow(self.BCCBMatrix,cmap='jet', interpolation='none')

        return None    
    

    def GetRealization(self):
        randomComplexArray = np.random.randn(self.numberOfRows, self.numberOfElementsInRows) \
                      + 1j * np.random.randn(self.numberOfRows, self.numberOfElementsInRows)
        fullRealization = np.fft.fft2(np.multiply(self.sqrtOfEigenvalues,randomComplexArray),norm="ortho")
        realization = fullRealization[0:self.numberOfXPoints,0:self.numberOfYPoints]; 
        return np.real(realization), np.imag(realization)     
    
    
def MakeBCCBMatrix(firstRowOfBlocks):
    (numberOfElementsInRows,numberOfElementsInRowsTimesNumberOfRows) = firstRowOfBlocks.shape
    numberOfRows = np.int64(numberOfElementsInRowsTimesNumberOfRows/numberOfElementsInRows)
    print(numberOfRows,numberOfElementsInRowsTimesNumberOfRows,numberOfElementsInRows)

    BCCBMatrix = firstRowOfBlocks
    for index in range(1,numberOfRows):
        BCCBMatrix = np.vstack((BCCBMatrix,np.roll(firstRowOfBlocks,index*numberOfElementsInRows,axis=1)))
       
    return BCCBMatrix
    
def MakeCirculantMatrix(row):
    sizeOfMatrix = row.size
    row = row.reshape((1,sizeOfMatrix))
    circulantMatrix = np.empty((sizeOfMatrix,sizeOfMatrix),dtype=np.float64)
    circulantMatrix[0,:] = row
    for index in range(1,sizeOfMatrix):
        circulantMatrix[index,:] = np.roll(circulantMatrix[index-1,:],1)
       
    return circulantMatrix
    

    
