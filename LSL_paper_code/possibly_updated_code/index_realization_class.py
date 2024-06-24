#random field generator using the circulant embedding method

import numpy as np
#import matplotlib.pyplot as plt

class index_realization_class:
    def __init__(self, X, Y, nZ, epsilon, indexvariancescaling, indexstandarddev, correlationlength):
        self.X = X
        self.nX = X.size
        self.dX = self.X[1]-self.X[0]
        
        self.Y = Y
        self.nY = Y.size
        self.dY = self.Y[1]-self.Y[0]
        
        self.nZ = nZ
        self.numberofrealizations = nZ//2
        
        self.epsilon  = epsilon
        self.indexvariancescaling = indexvariancescaling
        
        self.indexstandarddev = indexstandarddev
        
        self.correlationlength = correlationlength
        
        self.meshX, self.meshY = np.meshgrid(self.X, self.Y)
        
        self.rows = np.zeros([self.nX, self.nY])
        self.cols = np.zeros([self.nX, self.nY])
        
        # self.indexrealizations = np.zeros([self.nZ, self.nX, self.nY])
        # 
        return None
        
        #define the covariance function
    def correlation_funct(self,deltaX, deltaY): #, correlationlength, epsilon, indexvariancescaling):
        radius = np.sqrt(deltaX**2+deltaY**2)
        if radius > 1:
            correlation = 1 - ((self.epsilon/self.indexvariancescaling)**(2/3))*(radius/self.correlationlength)**(2/3)
        else:
            correlation = 1 - ((self.epsilon/self.indexvariancescaling)**(2/3))*(radius/self.correlationlength)**2
            
        return correlation
    
#     #sample the covariance function to build the BCCB mtx
#     def BlkCirc_row_build(self):
#         for i in np.arange(self.nX):
#             for j in np.arange(self.nY):
#                 self.rows[j,i] = self.correlation_funct(self.X[i]-self.X[0], self.Y[j]-self.Y[0]) #rows of block cov mtx
#                 self.cols[j,i] = self.correlation_funct(self.X[0]-self.X[i], self.Y[0]-self.Y[j]) #cols of block cov mtx
#         
#         self.BlkCirc_1 = np.append( self.rows, np.delete(np.flip(self.cols,1),-1,1),axis=1)
#         self.BlkCirc_2 = np.append(np.delete(np.flip(self.cols,0),-1,0), np.delete(np.delete(np.flip(np.flip(self.rows,0),1),-1,0),-1,1),axis=1)
#         
#         BlkCirc_row = np.append(self.BlkCirc_1,self.BlkCirc_2,axis = 0)
#         
#         return BlkCirc_row
#         
#     def compute_eigenvalues(self):
#         #compute eigenvalues
#         BlkCirc_row = self.BlkCirc_row_build(self)
#         self.lam = np.real(np.fft.fft2(BlkCirc_row))/(2.0*self.nX-1)/(2.0*self.nY-1)
#         
#         self.vectlam = self.lam.flatten()
#         self.vectlam[self.vectlam<0]=0  #this is just setting negative entries to 0 so we can take the square root
# 
#         evalz = np.sqrt(self.vectlam.reshape(2*self.nX-1,2*self.nY-1))
#         
#         return evalz
        
    def makeindexrealizations(self):
        indexrealizations = np.zeros([self.nZ, self.nX, self.nY])
        
        for i in np.arange(self.nX):
            for j in np.arange(self.nY):
                self.rows[j,i] = self.correlation_funct(self.X[i]-self.X[0], self.Y[j]-self.Y[0]) #rows of block cov mtx
                self.cols[j,i] = self.correlation_funct(self.X[0]-self.X[i], self.Y[0]-self.Y[j]) #cols of block cov mtx
        
        self.BlkCirc_1 = np.append( self.rows, np.delete(np.flip(self.cols,1),-1,1),axis=1)
        self.BlkCirc_2 = np.append(np.delete(np.flip(self.cols,0),-1,0), np.delete(np.delete(np.flip(np.flip(self.rows,0),1),-1,0),-1,1),axis=1)
        
        BlkCirc_row = np.append(self.BlkCirc_1,self.BlkCirc_2,axis = 0)
        
        self.lam = np.real(np.fft.fft2(BlkCirc_row))/(2.0*self.nX-1)/(2.0*self.nY-1)
        
        self.vectlam = self.lam.flatten()
        self.vectlam[self.vectlam<0]=0  #this is just setting negative entries to 0 so we can take the square root

        evalz = np.sqrt(self.vectlam.reshape(2*self.nX-1,2*self.nY-1))
        
        self.lam = np.real(np.fft.fft2(BlkCirc_row))/(2.0*self.nX-1)/(2.0*self.nY-1)
        
        self.vectlam = self.lam.flatten()
        self.vectlam[self.vectlam<0]=0  #this is just setting negative entries to 0 so we can take the square root

        evalz = np.sqrt(self.vectlam.reshape(2*self.nX-1,2*self.nY-1))
        
        for k in np.arange(self.numberofrealizations):
            kk = self.numberofrealizations+k
            
            # a = self.indexstandarddev*
            a = (np.random.randn(2*self.nX-1,2*self.nY-1) + np.complex(0,1.0)*np.random.randn(2*self.nX-1,2*self.nY-1))
            F = np.fft.fft2(evalz*a)
            F = F[0:self.nX:1,0:self.nY:1]
        
            field1 = np.real(F)
            field2 = np.imag(F)
            
            indexrealizations[k,:,:]=field1
            indexrealizations[kk,:,:]=field2
        
        return indexrealizations


    
