#index realization- adapting the matlab code "stationary gaussian process" from: 
# Reference:
#  Kroese, D. P., & Botev, Z. I. (2015). Spatial Process Simulation.
#  In Stochastic Geometry, Spatial Statistics and Random Fields(pp. 369-404)
#  Springer International Publishing, DOI: 10.1007/978-3-319-10064-7_12

import numpy as np
import matplotlib.pyplot as plt
#import scipy.io.savemat
from scipy.io import savemat

#given nondim params
variance = 1.
correlationlength = 1.
indexstructureconstant = np.sqrt(5.0) * 1e-7  #1e-10
indexvariancescaling =      1.
backgroundindex = 1+1e-6

#dimensional params
wavelength =   1e-6
innerscale =   1e-2
outerscale =    1e2
aperturediameter  =      1
wavenumber = (2.0)*np.pi/wavelength

propdist = 1e3



#dimensionless params

indexstandarddev = (indexstructureconstant/np.sqrt(2.0))*\
                                      ((indexvariancescaling * outerscale)**(1.0/3.0))
        
# transverseCharacteristicLength  = aperturediameter
# propagationCharacteristicLength = propdist/100
# innerscale = transverseCharacteristicLength
# outerscale = propagationCharacteristicLength

epsilon = innerscale/outerscale
alpha = ((innerscale**2)*wavenumber)/outerscale
#indexstandarddev = 1e-6


gamma = innerscale*wavenumber*np.sqrt(indexstandarddev)

#compwidthX = 1.
lengthX = 2.
complengthX = lengthX/innerscale
nX = 200
dX = complengthX/nX
X = -complengthX/2. + dX*np.arange(0,nX)

#compwidthY = 1.
lengthY = 2.
complengthY = lengthY/innerscale
nY = 200
dY = complengthY/nY
Y = -complengthY/2. +dY*np.arange(0,nY)

meshX, meshY = np.meshgrid(X, Y)

nZ = 500       #determines the number of random phase screens that must be genenerated- needs to be even! bc we produce two 
                        #fields each run, only need to loop over the half of the number of z points!

correlationfunctchoice = 1   #determines which correlation function is used - gaussian(0) or kolomogorov(1)

#initialize rows and cols for sampling covariance function
rows = np.zeros([nX, nY])
cols = np.zeros([nX, nY])

#define the covariance function
def correlation_funct(deltaX, deltaY): #, correlationlength, epsilon, indexvariancescaling):
    radius = np.sqrt(deltaX**2+deltaY**2)
    if correlationfunctchoice == 0:
        correlation = np.exp(-((np.sqrt(deltaX**2.0+deltaY**2.0)/correlationlength)**2.0))
    else:
        if radius < 1:
            correlation = 1 - ((epsilon/indexvariancescaling)**(2/3))*(radius/correlationlength)**(2/3)
        else:
            correlation = 1 - ((epsilon/indexvariancescaling)**(2/3))*(radius/correlationlength)**2
    return correlation

#sample the covariance function to build the BCCB mtx
for i in np.arange(nX):
    for j in np.arange(nY):
        rows[j,i] = correlation_funct(X[i]-X[0], Y[j]-Y[0]) #rows of block cov mtx
        cols[j,i] = correlation_funct(X[0]-X[i], Y[0]-Y[j]) #cols of block cov mtx

BlkCirc_1 = np.append( rows, np.delete(np.flip(cols,1),-1,1),axis=1)
BlkCirc_2 = np.append(np.delete(np.flip(cols,0),-1,0), np.delete(np.delete(np.flip(np.flip(rows,0),1),-1,0),-1,1),axis=1)

BlkCirc_row = np.append(BlkCirc_1,BlkCirc_2,axis = 0)

#compute eigenvalues
lam = np.real(np.fft.fft2(BlkCirc_row))/(2.0*nX-1)/(2.0*nY-1)

vectlam = lam.flatten()

# print("negativeevals", vectlam[vectlam<0])
# print("sizevectlam", np.size(vectlam))
# print("sizenegevals",np.size(vectlam[vectlam<0]))
# print("sizezeroevals", np.size(vectlam[vectlam==0]))

vectlam[vectlam<0]=0  #this is just setting negative entries to 0 so we can take the square root

evalz = np.sqrt(vectlam.reshape(2*nX-1,2*nY-1))

#set it up to get a batch of realizations to use in the propagation solver
numberofrealizations = nZ//2
indexrealizations = np.zeros([nZ, nX, nY])

for k in np.arange(numberofrealizations):
    kk = numberofrealizations+k
    
    a = indexstandarddev*(np.random.randn(2*nX-1,2*nY-1) + np.complex(0,1.0)*np.random.randn(2*nX-1,2*nY-1))
    F = np.fft.fft2(evalz*a)
    F = F[0:nX:1,0:nY:1]

    field1 = np.real(F)
    field2 = np.imag(F)
    
    indexrealizations[k,:,:]=field1
    indexrealizations[kk,:,:]=field2

indexreals = {}
for ii in np.arange(nZ):
    indexreals[ii] = indexrealizations[ii,:,:]


np.save('indexrealizations2',indexrealizations)

#savemat('indexrealizations.mat', indexreals)

# plt.imshow(field1)
# plt.show()

fieldsample1 = indexrealizations[1,:,:]
fieldsample2 = indexrealizations[2,:,:]

fig = plt.figure(4)
ax = plt.gca()
cont = ax.contourf(meshX, meshY, fieldsample1, 50, linewidth=0, antianliased = False)
fig.colorbar(cont, shrink=0.5, aspect = 5)
plt.show()

fig = plt.figure(5)
ax = plt.gca()
cont = ax.contourf(meshX, meshY, fieldsample2 , 50, linewidth=0, antianliased = False)
fig.colorbar(cont, shrink=0.5, aspect = 5)
plt.show()

