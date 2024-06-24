# imports
import numpy as np
from numpy.random import multivariate_normal as np_multivariate_normal
from scipy.stats import multivariate_normal as sp_multivariate_normal
from numpy import linalg as np_linalg
from scipy import linalg as sp_linalg




class Noise_Realization:
    def __init__(self,xPoints,
                      yPoints,
                      correlationMatrix):
        """set all dimensional parameters and calculate the nondimensional parameters"""
    
        self.xPoints                    = xPoints       
        self.numberOfXPoints            = xPoints.size
        self.dx = xPoints[1] - xPoints[0]

        
        self.yPoints                    = yPoints
        self.numberOfYPoints            = yPoints.size
        self.dy = yPoints[1] - yPoints[0]
        
        self.correlationMatrix          = correlationMatrix

        self.Mat = np.zeros((5,5))
        
        return None


    def Update(C,Wx,Wy,Tx,Ty,X,Y,Fx,Fy,P):
        self.C = C
        self.Wx = Wx
        self.Wy = Wy
        self.Tx = Tx
        self.Ty = Ty
        self.X = X
        self.Y = Y
        self.Fx = Fx
        self.Fy = Fy

        self.SetModes()
        
        return None
  
    def SetModes():
        I =  self.C * np.sqrt(self.Wx * self.Wy / np.pi)

        for indexX in range(self.numberOfXPoints):
            for indexY in range(self.numberOfYPoints):
                theta =  0.5 * ( (self.Wx * self.Wx * (self.xPoints[indexX] - self.X) * (self.xPoints[indexX] - self.X)) 
                               + (self.Wy * self.Wy * (self.yPoints[indexY] - self.Y) * (self.yPoints[indexY] - self.Y)) )

                self.ansatzSquared[indexX,indexY] =  I * I * np.exp(- 2.0 * theta)
                
                self.M_Tx = 4.0 * self.Wx * self.Wx * (self.xPoints[indexX] - self.X) * self.ansatzSquared[indexX,indexY] / self.C / self.C

                self.M_Ty = 4.0 * self.Wy * self.Wy * (self.yPoints[indexX] - self.Y) * self.ansatzSquared[indexX,indexY] / self.C / self.C
                
                self.M_Fx = self.Wx * self.Wx * ( 1.0 - 2.0 * self.Wx * self.Wx * (self.xPoints[indexX] - self.X) * (self.xPoints[indexX] - self.X) ) * self.ansatzSquared[indexX,indexY] / self.C / self.C

                self.M_Fy = self.Wy * self.Wy * ( 1.0 - 2.0 * self.Wy * self.Wy * (self.yPoints[indexX] - self.Y) * (self.yPoints[indexX] - self.Y) ) * self.ansatzSquared[indexX,indexY] / self.C / self.C

                self.M_P = (2.0 - self.Wx * self.Wx * (self.xPoints[indexX] - self.X) * (self.xPoints[indexX] - self.X) - self.Wy * self.Wy * (self.yPoints[indexY] - self.Y) * (self.yPoints[indexY] - self.Y) ) * self.ansatzSquared[indexX,indexY] / self.C / self.C
        return None


    def SetCorrelationMatrix(self):
        
        #self.CorTxTx 
        self.Mat[0,0] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Tx,self.M_Tx))))) * self.dx * self.dx * self.dy * self.dy
        #self.CorTxTy 
        self.Mat[0,1] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Tx,self.M_Ty))))) * self.dx * self.dx * self.dy * self.dy
        self.Mat[1,0] = self.Mat[0,1]
        #self.CorTxFx 
        self.Mat[0,2] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Tx,self.M_Fx))))) * self.dx * self.dx * self.dy * self.dy
        self.Mat[2,0] = self.Mat[0,2]
        #self.CorTxFy 
        self.Mat[0,3] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Tx,self.M_Fy))))) * self.dx * self.dx * self.dy * self.dy
        self.Mat[3,0] = self.Mat[0,3]
        #self.CorTxP  
        self.Mat[0,4] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Tx,self.M_P ))))) * self.dx * self.dx * self.dy * self.dy
        self.Mat[4,0] = self.Mat[0,4]

        #self.CorTyTy 
        self.Mat[1,1] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Ty,self.M_Ty))))) * self.dx * self.dx * self.dy * self.dy
        #self.CorTyFx 
        self.Mat[1,2] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Ty,self.M_Fx))))) * self.dx * self.dx * self.dy * self.dy
        self.Mat[2,1] = self.Mat[1,2]        
        #self.CorTyFy 
        self.Mat[1,3] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Ty,self.M_Fy))))) * self.dx * self.dx * self.dy * self.dy
        self.Mat[3,1] = self.Mat[1,3]
        #self.CorTyP  
        self.Mat[1,4] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Ty,self.M_P ))))) * self.dx * self.dx * self.dy * self.dy
        self.Mat[4,1] = self.Mat[1,4]

        self.CorFxFx 
        self.Mat[2,2] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Fx,self.M_Fx))))) * self.dx * self.dx * self.dy * self.dy
    
        #self.CorFxFy 
        self.Mat[2,3] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Fx,self.M_Fy))))) * self.dx * self.dx * self.dy * self.dy
        self.Mat[3,2] = self.Mat[2,3]
        #self.CorFxP  
        self.Mat[2,4] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Fx,self.M_P ))))) * self.dx * self.dx * self.dy * self.dy
        self.Mat[4,2] = self.Mat[2,4]

        #self.CorFyFy 
        self.Mat[3,3]= np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Fy,self.M_Fy))))) * self.dx * self.dx * self.dy * self.dy
        #self.CorFyP  
        self.Mat[3,4] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_Fx,self.M_P ))))) * self.dx * self.dx * self.dy * self.dy
        self.Mat[4,3] = self.Mat[3,4]

        #self.CorPP    
        self.Mat[4,4] = np.trapz(np.trapz(np.trapz(np.trapz(self.GetIntegrand(self.M_P,self.M_P  ))))) * self.dx * self.dx * self.dy * self.dy

        self.L = np.linalg.cholesky(self.Mat)

        return None


    def CorrelationFunction(self,x1,x2,y1,y2):
        radiusSquared = (x2-x1)**2.0 + (y2-y1)**2.0
        radius        = np.sqrt(radiusSquared)
        if radius < 1:
            correlationValue = 1.0 - ((self.epsilon/self.indexVarianceScaling)**(2/3)) * radiusSquared
        else:
            correlationValue = 1.0 - ((self.epsilon/self.indexVarianceScaling)**(2/3)) * radiusSquared**(1/3)

        return correlationValue
    
    def GetIntegrand(self,A,B):
        for indexX1 in range(self.numberOfXPoints):
            for indexX2 in range(self.numberOfXPoints):
                for indexY1 in range(self.numberOfYPoints):
                    for indexY2 in range(self.numberOfYPoints):
                        self.integrand[indexX1,
                                       indexX2,
                                       indexY1,
                                       indexY2]  = A[indexX1,indexY1] * self.CorrelationFunction[indexX1,indexX2,indexY1,indexY2] * B[indexX2,indexY2]

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
        return None  

    def GetRealization(self):
        realization = np.zeros((1,5))
        for index1 in range(5):
            for index2 in range(5):
                realization[index1] += self.L[index1,index2] * np.random.randn()
        return realization     
   
