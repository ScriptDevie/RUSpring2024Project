#attempting to transfer over one or two of the Paraxial_Equation propagators from D. Cargill's code
#transfered split-step FFT/exact propagator


import numpy as np
import scipy as sp
import scipy.integrate as int
import matplotlib.pyplot as plt
from scipy.stats import moment
from index_realization_class import index_realization_class

#import time

#start_time = time.clock()

## Define Dimensional Parameter Values
# codeName             # Value                   # (symbol) [char. value] <units>
#----------------------------------------------------------------------------------------------
waveLength             = 1e-6                    # (lambda) [10^{-6}]          <meters>
waveNumber             = (2.0*np.pi)/waveLength  # (lambda) [10^{6}]           <meters^{-1}>
innerScale             = 1e-2                    # (l_0)    [10^{-3}]          <meters>
outerScale             = 1e1                     # (L_0)    [10^{3}]           <meters>
indexStructureConstant = 1e-9#4data=1e-6 #np.sqrt(5.0) * 1e-7     # (C_n)  [10^{-10},10^{-6}] <meters^{-1/3}>
indexVarianceScaling   = 1                       # C_0
apertureDiameter       = 2e-1                    # (D_{i})  [1 - .1]           <meters>
propagationDistance    = 1e4                     # (L)      [10^{3} - 10^{4}]  <meters> 
lengthX                = 1.0                     # XLength x_max               <meters>
lengthY                = 1.0                     # YLength y_max               <meters>
backgroundIndex        = 1.0 + 1e-6              # n_0      [1+e-6]
correlationlength      = 1
transverseCharacteristicLength  = apertureDiameter
propagationCharacteristicLength = propagationDistance/100
# 
transverseCharacteristicLength = innerScale
propagationCharacteristicLength = outerScale

## Define Simulation Parameter Values
#----------------------------------------------------------------------------------------------
numberOfPointsX                   = 400
numberOfPointsY                   = 400
numberOfPointsZ                   = 2000

numbruns = 1

initialFieldIdentifier            = 'lagra'

myRank = 1
outputFilename = 'output.txt'

# Dimensionless Parameter Values
#indexStandardDeviation = (indexStructureConstant/np.sqrt(2.0))*\
#                              ((self.indexVarianceScaling * self.outerScale)**(1.0/3.0))  # (sigma_n)
#epsilon                = innerScale/self.outerScale 
#alpha                  = ((innerScale**2.0) * waveNumber)/outerScale
#beta                   = (innerScale * waveNumber) 
#gamma                  = innerScale * waveNumber * np.sqrt(indexStandardDeviation)
#nu                     = innerScale * waveNumber * indexStandardDeviation
#computationalPropagationDistance  = propagationDistance/outerScale
#computationalApertureDiameter     = apertureDiameter/innerScale

indexStandardDeviation = (indexStructureConstant/np.sqrt(2.0))*((indexVarianceScaling * outerScale)**(1.0/3.0))  # (sigma_n)

epsilon                = transverseCharacteristicLength/propagationCharacteristicLength 
alpha                  = ((transverseCharacteristicLength**2.0) * \
                                        waveNumber)/propagationCharacteristicLength

beta                   = (transverseCharacteristicLength * waveNumber) 
gamma                  = transverseCharacteristicLength * waveNumber * \
                                      np.sqrt(indexStandardDeviation)
#gamma = 0                                      
nu                     = transverseCharacteristicLength * waveNumber * indexStandardDeviation

computationalPropagationDistance  = propagationDistance/propagationCharacteristicLength
computationalApertureDiameter     = apertureDiameter/transverseCharacteristicLength


stepSizeZ           = computationalPropagationDistance/numberOfPointsZ
pointsZ             = stepSizeZ*np.arange(0,numberOfPointsZ+1)

computationalWidthY = lengthY/transverseCharacteristicLength
computationalWidthX = lengthX/transverseCharacteristicLength
        
stepSizeY           = computationalWidthY/numberOfPointsY
pointsY             = -computationalWidthY/2.0 + stepSizeY*np.arange(0,numberOfPointsY)
waveNumberStepSizeY = (2.0*np.pi)/computationalWidthY
waveNumbersY        = waveNumberStepSizeY * np.arange(-numberOfPointsY/2.0,numberOfPointsY/2.0)
          
stepSizeX           = computationalWidthX/numberOfPointsX
pointsX             = -computationalWidthX/2.0 + stepSizeX*np.arange(0,numberOfPointsX)
waveNumberStepSizeX = (2.0*np.pi)/computationalWidthX
waveNumbersX        = waveNumberStepSizeX * np.arange(-numberOfPointsX/2.0,numberOfPointsX/2.0)


meshPointsX, meshPointsY           = np.meshgrid(pointsX , pointsY)
meshWaveNumbersX, meshWaveNumbersY = np.meshgrid(waveNumbersX, waveNumbersY)

# np.save('pointsX', pointsX)
# np.save('pointsY', pointsY)
# np.save('pointsZ', pointsZ)
#------------------------------------------
#transfering over the things needed for the propagator below:

#need the initial field 
#-----------------------------------------------
beamWaist         = computationalApertureDiameter
beamWaistLocation = computationalPropagationDistance/2.0
                                  
                                  
width = beamWaist / np.sqrt( beamWaist**4.0 + (pointsZ[0] - beamWaistLocation)**2.0 ) 
# \frac{z-z_0}{(z-z_0)^2 + w_0^4}
focus = (pointsZ[0] - beamWaistLocation) / ( (pointsZ[0] - beamWaistLocation)**2.0 + (beamWaist**4.0) )
                
widthX = np.sqrt(backgroundIndex * alpha) * width
widthY = np.sqrt(backgroundIndex * alpha) * width
                                                                                  
focusX = - (backgroundIndex * alpha / 2.0) * focus                          
focusY = - (backgroundIndex * alpha / 2.0) * focus
                                                          
phase = np.arctan((pointsZ[0] - beamWaistLocation)/(beamWaist**2.0))  
        
amplitude = 10.0

positionX = 0.0
positionY = 0.0

tiltX     = 0.0
tiltY     = 0.0  
    
theta = 0.5* ((widthX**2.0) * ((meshPointsX-positionX)**2.0) + (widthY**2.0) * ((meshPointsY-positionY)**2.0))
phi   = phase + tiltX * (meshPointsX-positionX) + tiltY * (meshPointsY-positionY) \
                          + focusX * ((meshPointsX-positionX)**2.0) + focusY * ((meshPointsY-positionY)**2.0)
initialField  = (amplitude*np.sqrt(widthX*widthY)/np.sqrt(np.pi)) * np.exp(- (theta + 1.j*phi) )

#set up the functions used in the split-step calculation
#-----------------------------------------------------------
#set up function that calculates the energy
def Energy(stepSizeX,stepSizeY,field):
    integralOverY = int.simps(np.abs(field)**2.0, dx=stepSizeY)
    energy =  int.simps(integralOverY, dx=stepSizeX)

    return energy
    
def irradiance(field):
    irradiance = (np.abs(field))**2
    return irradiance 

#set up empty array to take in the split-step solution
#the first two-dim array is the initial field
peakfields = np.zeros([numberOfPointsX, numberOfPointsY, numbruns], dtype = np.complex64)
finalfields = np.zeros([numberOfPointsX, numberOfPointsY, numbruns], dtype = np.complex64)
centerirradiance = np.zeros([numberOfPointsZ+1, numbruns])
for jj in np.arange(numbruns):
    print(jj)
    recordedFieldFFTIntgrator = np.empty([numberOfPointsX,\
                                                    numberOfPointsY,\
                                                    numberOfPointsZ+1], dtype=np.complex128)
    recordedFieldFFTIntgrator[:,:,0] = initialField     #initial field
    
    # fieldEnergyFFTIntgrator  = np.empty(numberOfPointsZ+1, dtype=np.float64)
    # fieldEnergyFFTIntgrator[0] = Energy(stepSizeX, stepSizeY, recordedFieldFFTIntgrator[:,:,0]) #initial energy
    
    
    #set up the FFT integrator next:
    def FFTIntegratorExact(backgroundIndex, alpha, meshWaveNumbersX, meshWaveNumbersY,field,stepSize):
        fftIntegrator = np.exp(-(1.j/(2.0*backgroundIndex*alpha))*\
                                    (meshWaveNumbersX**2.0 + meshWaveNumbersY**2.0)*\
                                    (stepSize))
        
        returnField = np.fft.ifft2( np.fft.ifftshift( fftIntegrator * np.fft.fftshift(np.fft.fft2(field)) ) )
        return returnField
    
  
    indexrealization = np.load('indexrealization2.npy')
    # indexrealization_generator = index_realization_class(pointsX, pointsY, numberOfPointsZ, epsilon, indexVarianceScaling, indexStandardDeviation, correlationlength)
    # 
    # indexrealization = indexrealization_generator.makeindexrealizations()
    
    for indexZ in range(numberOfPointsZ):
        indexSheet = indexrealization[indexZ,:,:]
        temp = np.copy(recordedFieldFFTIntgrator[:,:,indexZ])
        temp *= np.exp(1.j * ((gamma**2.0)/alpha) *(stepSizeZ/2.0) * indexSheet)
        temp = FFTIntegratorExact(backgroundIndex, alpha, meshWaveNumbersX, meshWaveNumbersY,temp, stepSizeZ)
        temp *= np.exp(1.j * ((gamma**2.0)/alpha) * (stepSizeZ/2.0) * indexSheet)
    
        recordedFieldFFTIntgrator[:,:,indexZ+1] = np.copy(temp)  
        # fieldEnergyFFTIntgrator[indexZ+1] = Energy(stepSizeX,stepSizeY,recordedFieldFFTIntgrator[:,:,indexZ+1])
        # print("Finshed step ",indexZ+1," of ",numberOfPointsZ)
        # print("FFT Energy: ",fieldEnergyFFTIntgrator[indexZ+1])
        # return None
    
    peakfields[:,:,jj] = recordedFieldFFTIntgrator[:,:,numberOfPointsZ//2]
    finalfields[:,:,jj] = recordedFieldFFTIntgrator[:,:,numberOfPointsZ]
    
    for ii in np.arange(numberOfPointsZ+1):
        centerirradiance[ii,jj] = irradiance(recordedFieldFFTIntgrator[:,:,ii])[numberOfPointsX//2,numberOfPointsY//2]



fig = plt.figure(11)
plt.plot(pointsZ,centerirradiance)
plt.title('center irradiance')
plt.show()
# 
fig = plt.figure(14)
ax = fig.gca()
cont = ax.contourf(meshPointsX, meshPointsY, irradiance(recordedFieldFFTIntgrator[:,:,0]), 50, linewidth=0, antianliased=False)
fig.colorbar(cont, shrink=0.5, aspect=5)
plt.title('Initial Irradiance')
plt.show()

fig = plt.figure(15)
ax = fig.gca()
cont = ax.contourf(meshPointsX, meshPointsY, irradiance(recordedFieldFFTIntgrator[:,:,500]), 50, linewidth=0, antianliased=False)
fig.colorbar(cont, shrink=0.5, aspect=5)
plt.title('Peak Irradiance')
plt.show()

fig = plt.figure(16)
ax = fig.gca()
cont = ax.contourf(meshPointsX, meshPointsY, irradiance(recordedFieldFFTIntgrator[:,:,1000]), 50, linewidth=0, antianliased=False)
fig.colorbar(cont, shrink=0.5, aspect=5)
plt.title('Final Irradiance')
plt.show()

#print("--- %s seconds ---" % (time.clock() - start_time))

peak_yslice = irradiance(recordedFieldFFTIntgrator[:,:,numberOfPointsZ//2])[numberOfPointsX//2,:]

peak_xslice = irradiance(recordedFieldFFTIntgrator[:,:,numberOfPointsZ//2])[:,numberOfPointsY//2]

initial_yslice = irradiance(recordedFieldFFTIntgrator[:,:,0])[numberOfPointsX//2,:]

initial_xslice = irradiance(recordedFieldFFTIntgrator[:,:,0])[:,numberOfPointsY//2]

final_xslice = irradiance(recordedFieldFFTIntgrator[:,:,numberOfPointsZ].tolist())[:,numberOfPointsY//2]
final_yslice = irradiance(recordedFieldFFTIntgrator[:,:,numberOfPointsZ].tolist())[numberOfPointsX//2,:]
# 

np.save('paraxial_centerirr2', centerirradiance)
np.save('paraxial_fields2',recordedFieldFFTIntgrator)
np.save('paraxial_xslice_peak2', peak_xslice)
np.save('paraxial_yslice_peak2', peak_yslice)
np.save('paraxial_xslice_final2', final_xslice)
np.save('paraxial_yslice_final2', final_yslice)

# np.save('paraxrun_phasescreens',indexrealization)

fig = plt.figure(1)
plt.plot(pointsX, initial_xslice)
plt.title('initial irradiance x slice')
plt.show()

fig = plt.figure(2)
plt.plot(pointsY, initial_yslice)
plt.title('initial irradiance y slice')
plt.show()

fig = plt.figure(3)
plt.plot(pointsX, peak_xslice)
plt.title('peak irradiance x slice')
plt.show()

fig = plt.figure(4)
plt.plot(pointsY, peak_yslice)
plt.title('peak irradiance y slice')
plt.show()

fig = plt.figure(5)
plt.plot(pointsX, final_xslice)
plt.title('final irradiance x slice')
plt.show()

fig = plt.figure(6)
plt.plot(pointsY, final_yslice)
plt.title('final irradiance y slice')
plt.show()

fig = plt.figure(7)

# firstmomenthelm = np.zeros(numberOfPointsZ)
# secondmomenthelm = np.zeros(numberOfPointsZ)
# thirdmomenthelm = np.zeros(numberOfPointsZ)
# fourthmomenthelm = np.zeros(numberOfPointsZ)
# 
# for jjj in np.arange(numberOfPointsZ):
#     firstmomenthelm[jjj] = moment(np.ravel(irradiance(recordedFieldFFTIntgrator[:,:,jjj])), moment=1)
#     secondmomenthelm[jjj] = moment(np.ravel(irradiance(recordedFieldFFTIntgrator[:,:,jjj])),moment=2)
#     thirdmomenthelm[jjj] =  moment(np.ravel(irradiance(recordedFieldFFTIntgrator[:,:,jjj])), moment=3)
#     fourthmomenthelm[jjj] = moment(np.ravel(irradiance(recordedFieldFFTIntgrator[:,:,jjj])), moment =4)