import numpy as np

import matplotlib.pyplot as plt 
#from atm_prop_setup import atm_prop_setup
from NoiseRealization import Noise_Realization



import pdb
# from concurrent.features import ProcessPoolExecutor 

#specify dimensional parameters 
wavelength =   1e-6
innerscale =   1e-2
outerscale =    1e1
indexstructureconstant = 1e-6 #1e-9# np.sqrt(5.0) * 1e-7 #1e-10   
indexvariancescaling =  1     
correlationlength = 1 
backgroundindex = 1+1e-6
aperturediameter  = 2e-1
lengthX =      1.0
nX =   50              #number of points taken in X direction 
lengthY =      1.0
nY =   50              #number of points taken in the Y direction 
propdist  =1e4
nZ =   2000              #number of points taken in Z direction (also same as number of random fields generated)

#indexstandarddeviation = 0 #set to no random effects, currently
indexstandarddev = (indexstructureconstant/np.sqrt(2.0))*\
                                ((indexvariancescaling * outerscale)**(1.0/3.0))  # (sigma_n)

# transverseCharacteristicLength  = aperturediameter
# propagationCharacteristicLength = propdist/100
# # innerscale = transverseCharacteristicLength
# # outerscale = propagationCharacteristicLength

wavenumber = (2.0)*np.pi/wavelength

#define dimensionless parameters

epsilon = innerscale/outerscale
alpha = ((innerscale**2)*wavenumber)/outerscale
gamma = innerscale*wavenumber*np.sqrt(indexstandarddev)
#gamma = 0    #for taking out the random effects
nu    = gamma * indexstandarddev
comppropdist = propdist/outerscale
compaperture = aperturediameter/innerscale

dZ           = comppropdist/nZ       #stepsize in Z direction
Z             = dZ*np.arange(0,nZ+1)

complengthX = lengthX/innerscale
#complengthX = 1
dX           = complengthX/nX      #stepsize in X direction
X             = -complengthX/2.0 + dX*np.arange(0,nX)

complengthY = lengthY/innerscale
#complengthY = 1
dY           = complengthY/nY     #stepsize in Y direction
Y             = -complengthY/2.0 + dY*np.arange(0,nY)

meshX, meshY = np.meshgrid(X, Y)

noiseRealization = Noise_Realization(X, Y, nZ, epsilon, indexvariancescaling, indexstandarddev, correlationlength)



N = 10
L = max(max(X),max(Y))

Wx = L/(2*N) * range(N)
Wy = L/(2*N) * range(N)
X = L/N * range(-N//2,N//2)
Y = L/N * range(-N//2,N//2)

compaperture = aperturediameter/innerscale
beamwaist = compaperture
beamwaistlocation = comppropdist/2.0
A0 = 10.0
widthX0 = np.sqrt(backgroundindex*alpha)*(beamwaist)/np.sqrt(beamwaist**4+(0-beamwaistlocation)**2)
widthY0 = np.sqrt(backgroundindex*alpha)*(beamwaist)/np.sqrt(beamwaist**4+(0-beamwaistlocation)**2)
focusX0 = -(0.5*backgroundindex*alpha)*(Z[0]-beamwaistlocation)/((0-beamwaistlocation)**2+beamwaist**4)
focusY0 = -(0.5*backgroundindex*alpha)*(Z[0]-beamwaistlocation)/((0-beamwaistlocation)**2+beamwaist**4)
phase0  = np.arctan((Z[0] - beamwaistlocation)/(beamwaist**2.0))

#data = np.empty((N,N,N,N,5,5))
#for indexWx, wx in enumerate(Wx):
#   for indexWy, wy in enumerate(Wy):
#       for indexX, x in enumerate(X):
#           for indexY,y in enumerate(Y):
#               print(indexWx,indexWy,indexX,indexY)
#               check = noiseRealization.Update(1.0,wx,wy,0.0,0.0,x,y,0.0,0.0,0.0)
#               print(check)
#               data[indexWx,indexWy,indexX,indexY,:,:] = noiseRealization.ReturnMatrix()

check = noiseRealization.Update(A0,widthX0,widthY0,0.0,0.0,0.0,0.0,focusX0,focusY0,phase0)
print(check)
SQRTMat_Chol = noiseRealization.ReturnMatrix_Chol()
SQRTMat_Eigen = noiseRealization.ReturnMatrix_Eigen()

fullMat = noiseRealization.ReturnFullMatrix()
#pdb.set_trace()

np.savez('data',SQRTMat_Chol=SQRTMat_Chol, SQRTMat_Eigen=SQRTMat_Eigen, fullMatrix=fullMat)

