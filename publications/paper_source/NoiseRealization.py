# imports
import numpy as np
from numpy.random import multivariate_normal as np_multivariate_normal
from scipy.stats import multivariate_normal as sp_multivariate_normal
from numpy import linalg as np_linalg
#from scipy import linalg as sp_linalg


import pdb




class Noise_Realization:
    def __init__(self, xPoints, yPoints, nZ, epsilon, indexVarianceScaling, indexStandardDev, correlationLength):
        self.xPoints = xPoints
        self.numberOfXPoints = xPoints.size
        self.dx = self.xPoints[1]-self.xPoints[0]
        
        self.yPoints = yPoints
        self.numberOfYPoints = yPoints.size
        self.dy = self.yPoints[1]-self.yPoints[0]
        
        self.nZ = nZ
        self.numberofrealizations = nZ//2
        
        self.epsilon  = epsilon
        self.indexvariancescaling = indexVarianceScaling
        
        self.indexStandardDev = indexStandardDev
        
        self.correlationlength = correlationLength
        
        self.meshX, self.meshY = np.meshgrid(self.xPoints, self.yPoints)
      
        self.checks = dict.fromkeys(['Large Correlation Matrix', 'Parameters', 'Modes', 'Small Correlation Matrix','Sqrt Correlation Matrix Eigen','Sqrt Correlation Matrix Chol'])

        self.checks['Large Correlation Matrix'] = self.GetLargeCorrelationMatrix()
        
        self.Mat = np.zeros((5,5))


        return None


    def Update(self,C,Wx,Wy,Tx,Ty,X,Y,Fx,Fy,P):

        self.checks['Parameters']               = self.SetParameters(C,Wx,Wy,Tx,Ty,X,Y,Fx,Fy,P)
        self.checks['Modes']                    = self.SetModes()
        self.checks['Small Correlation Matrix'] = self.GetSmallCorrelationMatrix()
        self.checks['Sqrt Correlation Matrix Chol']   = self.ComputeSqrtCorrelationMatrix_Chol()
        self.checks['Sqrt Correlation Matrix Eigen']  = self.ComputeSqrtCorrelationMatrix_Eigen()

        return self.checks

    def SetParameters(self,C,Wx,Wy,Tx,Ty,X,Y,Fx,Fy,P):
        success = True

        if (Wx < 0.0 or Wy < 0.0):
            success = False
        if (X < np.min(self.xPoints) or X > np.max(self.xPoints)):
            success = False
        if (Y < np.min(self.yPoints) or Y > np.max(self.yPoints)):
            success = False            

        self.C = C
        self.Wx = Wx
        self.Wy = Wy
        self.Tx = Tx
        self.Ty = Ty
        self.X = X
        self.Y = Y
        self.Fx = Fx
        self.Fy = Fy

        return success


    def SetModes(self):
        success = True
        self.ansatzSquared = np.zeros((self.numberOfXPoints,self.numberOfYPoints))
        self.M_Tx = np.zeros((self.numberOfXPoints,self.numberOfYPoints))
        self.M_Ty = np.zeros((self.numberOfXPoints,self.numberOfYPoints))
        self.M_Fx = np.zeros((self.numberOfXPoints,self.numberOfYPoints))
        self.M_Fy = np.zeros((self.numberOfXPoints,self.numberOfYPoints))
        self.M_P  = np.zeros((self.numberOfXPoints,self.numberOfYPoints))

        for indexX in range(self.numberOfXPoints):
            for indexY in range(self.numberOfYPoints):
                theta =  0.5 * ( (self.Wx * self.Wx * (self.xPoints[indexX] - self.X) * (self.xPoints[indexX] - self.X)) 
                               + (self.Wy * self.Wy * (self.yPoints[indexY] - self.Y) * (self.yPoints[indexY] - self.Y)) )

                self.ansatzSquared[indexX,indexY] = (self.Wx * self.Wy / np.pi) * np.exp(- 2.0 * theta)
                
                self.M_Tx[indexX,indexY]  = 4.0 * self.Wx * self.Wx * (self.xPoints[indexX] - self.X) * self.ansatzSquared[indexX,indexY]

                self.M_Ty[indexX,indexY]  = 4.0 * self.Wy * self.Wy * (self.yPoints[indexY] - self.Y) * self.ansatzSquared[indexX,indexY]
                
                self.M_Fx[indexX,indexY]  = self.Wx * self.Wx * ( 1.0 - 2.0 * self.Wx * self.Wx * (self.xPoints[indexX] - self.X) * (self.xPoints[indexX] - self.X) ) * self.ansatzSquared[indexX,indexY]

                self.M_Fy[indexX,indexY]  = self.Wy * self.Wy * ( 1.0 - 2.0 * self.Wy * self.Wy * (self.yPoints[indexY] - self.Y) * (self.yPoints[indexY] - self.Y) ) * self.ansatzSquared[indexX,indexY]

                self.M_P[indexX,indexY]   = (2.0 - self.Wx * self.Wx * (self.xPoints[indexX] - self.X) * (self.xPoints[indexX] - self.X) - self.Wy * self.Wy * (self.yPoints[indexY] - self.Y) * (self.yPoints[indexY] - self.Y) ) * self.ansatzSquared[indexX,indexY]
        return success


    def GetSmallCorrelationMatrix(self):
        #
        success = True
        #self.CorTxTx 
        self.Mat[0,0] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Tx,self.M_Tx), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        #self.CorTxTy 
        self.Mat[0,1] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Tx,self.M_Ty), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        self.Mat[1,0] = self.Mat[0,1]
        #self.CorTxFx 
        self.Mat[0,2] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Tx,self.M_Fx), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        self.Mat[2,0] = self.Mat[0,2]
        #self.CorTxFy 
        self.Mat[0,3] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Tx,self.M_Fy), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        self.Mat[3,0] = self.Mat[0,3]
        #self.CorTxP  
        self.Mat[0,4] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Tx,self.M_P ), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        self.Mat[4,0] = self.Mat[0,4]

        #self.CorTyTy 
        self.Mat[1,1] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Ty,self.M_Ty), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        #self.CorTyFx 
        self.Mat[1,2] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Ty,self.M_Fx), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        self.Mat[2,1] = self.Mat[1,2]        
        #self.CorTyFy 
        self.Mat[1,3] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Ty,self.M_Fy), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        self.Mat[3,1] = self.Mat[1,3]
        #self.CorTyP  
        self.Mat[1,4] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Ty,self.M_P ), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        self.Mat[4,1] = self.Mat[1,4]

        #self.CorFxFx 
        self.Mat[2,2] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Fx,self.M_Fx), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        #self.CorFxFy 
        self.Mat[2,3] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Fx,self.M_Fy), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        self.Mat[3,2] = self.Mat[2,3]
        #self.CorFxP  
        self.Mat[2,4] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Fx,self.M_P ), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        self.Mat[4,2] = self.Mat[2,4]

        #self.CorFyFy 
        self.Mat[3,3] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Fy,self.M_Fy), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        #self.CorFyP  
        self.Mat[3,4] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Fy,self.M_P ), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        self.Mat[4,3] = self.Mat[3,4]

        #self.CorPP    
        self.Mat[4,4] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_P,self.M_P  ), dx=self.dx), dx=self.dx), dx=self.dy ),dx=self.dy) 
        
        if (np.any(np.isnan(self.Mat))):
            success = False
        if (np.any(np.isinf(self.Mat))):
            success = False
                
        return success


    def ComputeSqrtCorrelationMatrix_Eigen(self):
        success = True
        
        eigenVal,eigenVec = np_linalg.eigh(self.Mat)
        
        eigenVal[eigenVal<0.0] = 0.0
        
        self.sqrtMat_Eigen = np.dot(np.sqrt(np.diag(eigenVal)),eigenVec)
        return success

    def ComputeSqrtCorrelationMatrix_Chol(self):
        success = True
        
        self.sqrtMat_Chol = np.linalg.cholesky(self.Mat)
        
        return success

    def PrintArray(self,A):
        print(np.matrix(A))

    def ReturnMatrix_Eigen(self):
        return self.sqrtMat_Eigen

    def ReturnMatrix_Chol(self):
        return self.sqrtMat_Chol


    def ReturnFullMatrix(self):
        return self.Mat        

    def CorrelationFunction(self,x1,x2,y1,y2):
        radiusSquared = (x2-x1)**2.0 + (y2-y1)**2.0
        radius        = np.sqrt(radiusSquared)
        if radius < 1:
            correlationValue = 1.0 - ((self.epsilon/self.indexVarianceScaling)**(2/3)) * radiusSquared
        else:
            correlationValue = 1.0 - ((self.epsilon/self.indexVarianceScaling)**(2/3)) * radiusSquared**(1/3)

        return correlationValue
    
    def correlation_funct(self,x1,x2,y1,y2): #, correlationlength, epsilon, indexvariancescaling):
        
        radiusSquared = (x2-x1)**2.0 + (y2-y1)**2.0
        radius        = np.sqrt(radiusSquared)

        if radius > 1:
            correlation = 1 - ((self.epsilon/self.indexvariancescaling)**(2/3))*(radius/self.correlationlength)**(2/3)
        else:
            correlation = 1 - ((self.epsilon/self.indexvariancescaling)**(2/3))*(radius/self.correlationlength)**2
            
        return correlation

    def GetIntegrand(self,A,B):
        integrand = np.zeros((self.numberOfXPoints,self.numberOfXPoints,self.numberOfYPoints,self.numberOfYPoints))
        for indexX1 in range(self.numberOfXPoints):
            for indexX2 in range(self.numberOfXPoints):
                for indexY1 in range(self.numberOfYPoints):
                    for indexY2 in range(self.numberOfYPoints):
                        integrand[indexX1,
                                  indexX2,
                                  indexY1,
                                  indexY2]  = A[indexX1,indexY1] * self.correlationMatrix[indexX1,indexX2,indexY1,indexY2] * B[indexX2,indexY2]

        return integrand  

    def GetLargeCorrelationMatrix(self):
        success = True
        self.correlationMatrix = np.empty((self.numberOfXPoints,self.numberOfXPoints,self.numberOfYPoints,self.numberOfYPoints))
        # pdb.set_trace()
        for indexX1 in range(self.numberOfXPoints):
            for indexX2 in range(self.numberOfXPoints):
                for indexY1 in range(self.numberOfYPoints):
                    for indexY2 in range(self.numberOfYPoints):
                        self.correlationMatrix[indexX1,
                                               indexX2,
                                               indexY1,
                                               indexY2]  = self.correlation_funct(self.xPoints[indexX1],
                                                                                 self.xPoints[indexX2],
                                                                                 self.yPoints[indexY1],
                                                                                 self.yPoints[indexY2])
        return success  


