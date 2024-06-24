#LSL and Paraxial Helmholts Comps in Parallel

import numpy as np
# import matplotlib.pyplot as plt 
#from atm_prop_setup import atm_prop_setup
from index_realization_class import index_realization_class
from joblib import Parallel, delayed

#need to modify the for-loop slightly to be able to use the parallel feature from joblib

# from concurrent.features import ProcessPoolExecutor 

wavelength =   1e-6
innerscale =   1e-2
outerscale =    1e2
indexstructureconstant = 1e-6 # 1e-6 #5*indexstructureconstantvec[0] #np.sqrt(5.0) * 1e-7 #1e-10   
indexvariancescaling =      1     
correlationlength = 1 
backgroundindex = 1+1e-6
aperturediameter  = 2e-1
lengthX =      1.0
nX =   100              #number of points taken in X direction 
lengthY =      1.0
nY =   100              #number of points taken in the Y direction 
propdist  =1e5
nZ =  2000              #number of points taken in Z direction (also same as number of random fields generated)
    

#specify the number of runs - each run will generate a new set of random fields 
numbruns = 3000

#indexstandarddeviation = 0 #set to no random effects, currently
indexstandarddev = (indexstructureconstant/np.sqrt(2.0))*\
                                    ((indexvariancescaling * outerscale)**(1.0/3.0))  # (sigma_n)

wavenumber = (2.0)*np.pi/wavelength
    
#define dimensionless parameters
epsilon = innerscale/outerscale
alpha = ((innerscale**2)*wavenumber)/outerscale
gamma = innerscale*wavenumber*np.sqrt(indexstandarddev)
nu    = gamma * indexstandarddev
comppropdist = propdist/outerscale
compaperture = aperturediameter/innerscale
    
dZ           = comppropdist/nZ       #stepsize in Z direction
Z             = dZ*np.arange(0,nZ+1)
    
complengthX = lengthX/innerscale
dX           = complengthX/nX      #stepsize in X direction
X             = -complengthX/2.0 + dX*np.arange(0,nX)
waveNumberStepSizeX = (2.0*np.pi)/complengthX
waveNumbersX        = waveNumberStepSizeX * np.arange(-nX/2.0,nX/2.0)

complengthY = lengthY/innerscale

dY           = complengthY/nY     #stepsize in Y direction
Y             = -complengthY/2.0 + dY*np.arange(0,nY)
waveNumberStepSizeY = (2.0*np.pi)/complengthY
waveNumbersY        = waveNumberStepSizeY * np.arange(-nY/2.0,nY/2.0)
  
meshX, meshY = np.meshgrid(X, Y)
meshWaveNumbersX, meshWaveNumbersY = np.meshgrid(waveNumbersX, waveNumbersY)


   
#Gaussian ansatz 
def field(p):
    A, widthX, widthY, tiltX, tiltY, positionX, positionY, focusX, focusY, phase = p
    theta = 0.5*((widthX**2)*(meshX - positionX)**2 + (widthY**2)*(meshY - positionY)**2)
    phi = phase + tiltX*(meshX - positionX) + tiltY*(meshY - positionY) + focusX*(meshX - positionX)**2 + focusY*(meshY - positionY)**2
    field = (A*np.sqrt(widthX*widthY)/np.sqrt(np.pi))*np.exp(-(theta +1j*phi))
    
    return field

def irradiance(p):
    field_data = field(p)
    irradiance = (abs(field_data))**2
    
    return irradiance

###########################################
####### SET UP ODE SYSTEM AND SOLVE #######
###########################################

#define parameters for use in the derivative function
params = [backgroundindex, alpha, gamma]

def irradiance_parax(field):
    irradiance = (np.abs(field))**2
    return irradiance 

#define a projection for the random terms
def projection(indexrealization,mode, dX, dY):
    #need indexrealization and mode to be vectors- not multidim arrays... 
    integrand = indexrealization*mode
    
    proj = dX*dY*(np.trapz(np.trapz(integrand)))
    return proj

#define RHS of ODEs 

def f(p,params, dX, dY, meshX, meshY, indexrealization, index):
    A, widthX, widthY, tiltX, tiltY, positionX, positionY, focusX, focusY, phase = p #unpack unknowns living in p
    backgroundindex, alpha, gamma = params #unpack parameters that go into the derivatives
    
    V = (widthX*widthY/np.pi)*np.exp(-((widthX**2)*(meshX-positionX)**2+(widthY**2)*(meshY-positionY)**2))
    
    V_P  = (2-(widthX**2)*(meshX - positionX)**2-(widthY**2)*(meshY-positionY)**2)*V
    V_Tx = 4*(widthX**2)*(meshX-positionX)*V
    V_Ty = 4*(widthY**2)*(meshY-positionY)*V
    V_Fx = (widthX**2)*(1-2*(widthX**2)*(meshX-positionX)**2)*V
    V_Fy = (widthY**2)*(1-2*(widthY**2)*(meshY-positionY)**2)*V
    
    derivs = [0,                        #amplitude
              (2/(alpha*backgroundindex))*focusX*widthX, #4*focusX*widthX,           #widthX 
              (2/(alpha*backgroundindex))*focusY*widthY, #4*focusY*widthY,                     #widthY  
            -backgroundindex*(gamma**2)*projection(indexrealization[index,:,:],V_Tx, dX, dY),  #tiltX
            -backgroundindex*(gamma**2)*projection(indexrealization[index,:,:],V_Ty, dX, dY),  #tiltY
            -2*tiltX,                   #positionX
            -2*tiltY,                   #positionY
            (-1/(2*backgroundindex*alpha))*widthX**4+(2/(backgroundindex*alpha))*focusX**2 + ((gamma**2)/alpha)*projection(indexrealization[index,:,:], V_Fx,dX, dY), #focusX
            (-1/(2*backgroundindex*alpha))*widthY**4+(2/(backgroundindex*alpha))*focusY**2 + ((gamma**2)/alpha)*projection(indexrealization[index,:,:], V_Fy,dX, dY), #focusY
            (1/(2*backgroundindex*alpha))*(widthX**2+widthY**2)-((gamma**2)/alpha)*projection(indexrealization[index,:,:],V_P,dX, dY)] #phase
    return derivs

lag_soln = np.zeros([10,nZ+1,numbruns])
centerirradiance_lag = np.zeros([nZ+1,numbruns])

 #paraxial split-step solutions

finalfields = np.zeros([nX, nY, numbruns], dtype = np.complex128)
centerirradiance = np.zeros([nZ+1, numbruns])
centerirradiancenoatm = np.zeros([nZ+1,numbruns])


        

beamwaist = compaperture
beamwaistlocation = comppropdist

widthX0 = np.sqrt(backgroundindex*alpha)*(beamwaist)/np.sqrt(beamwaist**4+(0-beamwaistlocation)**2)
widthY0 = np.sqrt(backgroundindex*alpha)*(beamwaist)/np.sqrt(beamwaist**4+(0-beamwaistlocation)**2)
focusX0 = -(0.5*backgroundindex*alpha)*(Z[0]-beamwaistlocation)/((0-beamwaistlocation)**2+beamwaist**4)
focusY0 = -(0.5*backgroundindex*alpha)*(Z[0]-beamwaistlocation)/((0-beamwaistlocation)**2+beamwaist**4)
phase0  = np.arctan((Z[0] - beamwaistlocation)/(beamwaist**2.0))
amplitude0 = 10.
tiltX0 = 0.
tiltY0 = 0.
positionX0 = 0.
positionY0 = 0.


theta = 0.5* ((widthX0**2.0) * ((meshX-positionX0)**2.0) + (widthY0**2.0) * ((meshY-positionY0)**2.0))
phi   = phase0 + tiltX0 * (meshX-positionX0) + tiltY0 * (meshY-positionY0) \
                          + focusX0 * ((meshX-positionX0)**2.0) + focusY0 * ((meshY-positionY0)**2.0)
initialField  = (amplitude0*np.sqrt(widthX0*widthY0)/np.sqrt(np.pi)) * np.exp(- (theta + 1.j*phi) )
# recordedFieldFFTIntgrator[:,:,0] = initialField     #initial field


def LSLsolve(X, Y, nZ, epsilon, indexvariancescaling, indexstandarddev, correlationlength):

    indexrealization_generator = index_realization_class(X, Y, nZ, epsilon, indexvariancescaling, indexstandarddev, correlationlength)
    
    indexrealization = indexrealization_generator.makeindexrealizations()

    psln = np.zeros([10,nZ+1])
    centerirradiance_lag = np.zeros(nZ+1)
    
    psln[0,0]     = amplitude0                  #A0
    psln[1,0]     = widthX0 #0.05(used in animation)  0.04                 #widthX0
    psln[2,0]     = widthY0 #0.03(used in animation)     0.02                #widthY0
    psln[3,0]     = tiltX0 #0.04(used in animation)                   #tiltX0
    psln[4,0]     = tiltY0 #0.1(used in animation)                   #tiltY0
    psln[5,0]     = positionX0                   #positionX0
    psln[6,0]     = positionY0                   #positionY0
    psln[7,0]     = focusX0  #0.005  used in animation      #focusX0
    psln[8,0]     = focusY0 #0.005(used in animation)              #focusY0
    psln[9,0]     = np.arctan((Z[0] - beamwaistlocation)/(beamwaist**2.0)) 
    
    centerirradiance_lag[0] = irradiance(psln[:,0].tolist())[nX//2,nY//2]
    
    for ii in np.arange(nZ):
        p_ii = psln[:,ii]                #storing the current z-position soln vector 
        
        k1     = f(p_ii, params, dX, dY, meshX, meshY, indexrealization, ii)
    
        pcalc  = p_ii + (0.5*dZ*np.ones(10))*k1  #a temporary calculation used to compute the steps in runge-kutta
        k2     = f(pcalc, params, dX, dY, meshX, meshY, indexrealization, ii)
        #twok2    = (2*np.ones(10))*k2
        
        pcalc  = p_ii + (0.5*dZ*np.ones(10))*k2
        k3     = f(pcalc, params, dX, dY, meshX, meshY, indexrealization, ii)
        #twok3    = (2*np.ones(10))*k3
        
        pcalc  = p_ii + (dZ*np.ones(10))*k3
        k4     = f(pcalc, params, dX, dY, meshX, meshY, indexrealization, ii)
        
        p_iinew = p_ii + (((1./6)*dZ)*np.ones(10))*(k1+(2*np.ones(10))*k2+(2*np.ones(10))*k3+k4)
        # print("pnew",p_iinew)
        # print("sizepnew", np.size(p_iinew))
        
        psln[:,ii+1] = p_iinew
        #field_lag[:,:,ii+1] = field(p_iinew.tolist())
        centerirradiance_lag[ii+1] = irradiance(p_iinew.tolist())[nX//2,nY//2]
    return psln, centerirradiance_lag
    
def FFTIntegratorExact(backgroundIndex, alpha, meshWaveNumbersX, meshWaveNumbersY,field,stepSize):
    fftIntegrator = np.exp(-(1.j/(2.0*backgroundIndex*alpha))*\
                                    (meshWaveNumbersX**2.0 + meshWaveNumbersY**2.0)*\
                                    (stepSize))
        
    returnField = np.fft.ifft2( np.fft.ifftshift( fftIntegrator * np.fft.fftshift(np.fft.fft2(field)) ) )
    return returnField

def paraxsplitsolve(X, Y, nZ, epsilon, indexvariancescaling, indexstandarddev, correlationlength):
    peakfields = np.zeros([nX, nY], dtype = np.complex128)
    recordedFieldFFTIntgrator = np.empty([nX, nY,nZ+1], dtype=np.complex128)
    centerirradiance = np.zeros(nZ+1)

    recordedFieldFFTIntgrator[:,:,0] = initialField 
    centerirradiance[0] = irradiance_parax(initialField)[nX//2,nY//2]
    
    indexrealization_generator = index_realization_class(X, Y, nZ, epsilon, indexvariancescaling, indexstandarddev, correlationlength)
    
    indexrealization = indexrealization_generator.makeindexrealizations()

    for indexZ in range(nZ):
        indexSheet = indexrealization[indexZ,:,:]
        temp1 = recordedFieldFFTIntgrator[:,:,indexZ]
        # temp1 = np.copy(recordedFieldFFTIntgrator[:,:,indexZ])
        temp2 = temp1*np.exp(1.j * ((gamma**2.0)/alpha) *(dZ/2.0) * indexSheet)
        temp3 = FFTIntegratorExact(backgroundindex, alpha, meshWaveNumbersX, meshWaveNumbersY,temp2, dZ)
        temp4 = temp3*np.exp(1.j * ((gamma**2.0)/alpha) * (dZ/2.0) * indexSheet)
        recordedFieldFFTIntgrator[:,:,indexZ+1] = temp4
        centerirradiance[indexZ] = irradiance_parax(recordedFieldFFTIntgrator[:,:,indexZ])[nX//2,nY//2]
        
    peakfields[:,:] = recordedFieldFFTIntgrator[:,:,nZ]

    return peakfields,centerirradiance



# LagSoln = Parallel(n_jobs=-1)(delayed(LSLsolve)(X, Y, nZ, epsilon, indexvariancescaling, indexstandarddev, correlationlength) for j in range(numbruns))

ParaxSoln = Parallel(n_jobs=-1)(delayed(paraxsplitsolve)(X, Y, nZ, epsilon, indexvariancescaling, indexstandarddev, correlationlength) for j in range(numbruns))


# np.save('LagSoln3000runsCn1eminus6',LagSoln)
np.save('ParaxSoln3000runsCn1eminus6',ParaxSoln)