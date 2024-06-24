import numpy as np

#import matplotlib.pyplot as plt 
#from atm_prop_setup import atm_prop_setup


class ReducedNoiseRealization:
    def __init__(self, pathAndFilenameToData,constantFlag = True):
        
        self.constantFlag = constantFlag
        data = np.load(pathAndFilenameToData)
        self.SQRTMat = data['SQRTMat_Chol']
        
        if (self.constantFlag == 0):
           Wx = data['Wx']
           Wy = data['Wy']
           X = data['X']
           Y = data['Y']
           
           self.Wx = Wx
           self.Wy = Wy
           self.X  = X
           self.Y  = Y


        return None

    def GetRealization(self,WxTrue,WyTrue,XTrue,YTrue):
       
        indexWx = self.FindNearest(self.Wx,WxTrue)
        indexWy = self.FindNearest(self.Wy,WyTrue)
        indexX  = self.FindNearest(self.X,XTrue)
        indexY  = self.FindNearest(self.Y,YTrue)

        noise = np.randn((5,1))
        matrix = data[indexWx,indexWy,indexX,indexY,:,:]
        realization = np.dot(matrix,noise)
    
        return realization

    def FindNearest(self,array,value):
        array = np.asarray(array)
        index = (np.abs(array - value)).argmin()
        return index

    def GetRealizationConstant(self):
    
        return np.dot(self.SQRTMat,np.random.randn(5))
   