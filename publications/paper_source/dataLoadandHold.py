import numpy as np

#import matplotlib.pyplot as plt 
#from atm_prop_setup import atm_prop_setup


class Matrix_Data:
    def __init__(self, pathAndFilenameToData):
        
        np.load(pathAndFilenameToData)
        
        self.data = data
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

