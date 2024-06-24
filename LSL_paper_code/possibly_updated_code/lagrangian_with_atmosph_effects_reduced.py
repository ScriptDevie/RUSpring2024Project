import numpy as np

import matplotlib.pyplot as plt 
#from atm_prop_setup import atm_prop_setup
from ReducedNoiseRealization import ReducedNoiseRealization

# from concurrent.features import ProcessPoolExecutor 

#specify dimensional parameters 
wavelength =   1e-6
innerscale =   1e-2
outerscale =    1e1
indexstructureconstant = 1e-6 #1e-9# np.sqrt(5.0) * 1e-7 #1e-10   
indexvariancescaling =      1     
correlationlength = 1 
backgroundindex = 1+1e-6
aperturediameter  = 2e-1
lengthX =      1.0
nX =   400              #number of points taken in X direction 
lengthY =      1.0
nY =   400              #number of points taken in the Y direction 
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

#specify the number of runs - each run will generate a new set of random fields 
numbruns = 100

#noiseRealization = Noise_Realization(X, Y, nZ, epsilon, indexvariancescaling, indexstandarddev, correlationlength)

path = '/home/dcargill/AFRL/Projects/AFOSR/Atmospheric_Propagation/Lagrangian_for_Atmospheric_Propagation/lagrangianscalinglawproject/Sophia Code/data.npz'
reducedNoiseRealization = ReducedNoiseRealization(path,True)


# fig = plt.figure(4)
# ax = plt.gca()
# cont = ax.contourf(meshX, meshY, indexrealization[1,:,:], 50, linewidth=0, antianliased = False)
# fig.colorbar(cont, shrink=0.5, aspect = 5)
# plt.show()
        
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


def PrintParams(params):
    backgroundindex, alpha, gamma = params #unpack parameters that go into the derivatives
    print (backgroundindex,alpha,gamma)
    return None

PrintParams(params)



def GetNoiseRealization(p,params):
    A, widthX, widthY, tiltX, tiltY, positionX, positionY, focusX, focusY, phase = p #unpack unknowns living in p
    backgroundindex, alpha, gamma = params #unpack parameters that go into the derivatives
    
    [TxNoise,TyNoise,FxNoise,FyNoise,PNoise] = reducedNoiseRealization.GetRealizationConstant()
    print([TxNoise,TyNoise,FxNoise,FyNoise,PNoise])
    

    noise = np.array([0.0,                        #amplitude
                      0.0, #4*focusX*widthX,           #widthX 
                      0.0, #4*focusY*widthY,                     #widthY  
                     -backgroundindex * (gamma**2) * TxNoise, #tiltX
                     -backgroundindex * (gamma**2) * TyNoise, #tiltY
                      0.0,                   #positionX
                      0.0,                   #positionY
                      ((gamma**2)/alpha) * FxNoise, #focusX
                      ((gamma**2)/alpha) * FyNoise, #focusY
                     -((gamma**2)/alpha) * PNoise]) #phase
    
    return noise



def f(p,params, index):
    A, widthX, widthY, tiltX, tiltY, positionX, positionY, focusX, focusY, phase = p #unpack unknowns living in p
    backgroundindex, alpha, gamma = params #unpack parameters that go into the derivatives
    
    noise = GetNoiseRealization(p,params)


#    V = (widthX*widthY/np.pi)*np.exp(-((widthX**2)*(meshX-positionX)**2+(widthY**2)*(meshY-positionY)**2))
#    V_P  = (2-(widthX**2)*(meshX - positionX)**2-(widthY**2)*(meshY-positionY)**2)*V
#    V_Tx = 4*(widthX**2)*(meshX-positionX)*V
#    V_Ty = 4*(widthY**2)*(meshY-positionY)*V
#    V_Fx = (widthX**2)*(1-2*(widthX**2)*(meshX-positionX)**2)*V
#    V_Fy = (widthY**2)*(1-2*(widthY**2)*(meshY-positionY)**2)*V
    

    derivs = np.array([0.0,                        #amplitude
                       (2/(alpha*backgroundindex))*focusX*widthX, #4*focusX*widthX,           #widthX 
                       (2/(alpha*backgroundindex))*focusY*widthY, #4*focusY*widthY,                     #widthY  
                       0.0,  #tiltX
                       0.0,  #tiltY
                       -2*tiltX,                   #positionX
                       -2*tiltY,                   #positionY
                       (-1/(2*backgroundindex*alpha))*widthX**4+(2/(backgroundindex*alpha))*focusX**2 , #focusX
                       (-1/(2*backgroundindex*alpha))*widthY**4+(2/(backgroundindex*alpha))*focusY**2 , #focusY
                       ( 1/(2*backgroundindex*alpha))*(widthX**2+widthY**2) ]) #phase
    
    return derivs + noise

lag_soln = np.zeros([10,nZ+1,numbruns])
centerirradiance_lag = np.zeros([nZ+1,numbruns])

 #paraxial split-step solutions
peakfields = np.zeros([nX, nY, numbruns], dtype = np.complex64)
finalfields = np.zeros([nX, nY, numbruns], dtype = np.complex64)
centerirradiance = np.zeros([nZ+1, numbruns])
centerirradiancenoatm = np.zeros([nZ+1,numbruns])

for j in np.arange(numbruns):  
    print(j) 
    #make random fields for the index of refraction
    
    # indexrealization = np.load('paraxrun_phasescreens.npy')    
    #define initial conditions
    #define the focus and width ICs based off of exact gaussian beam soln 
    #beamwaist_physical = 0.01
    #beamwaist = (8/(alpha*backgroundindex))*beamwaist_physical #this is for trying to match Laurence's code
    beamwaist = compaperture
    beamwaistlocation = comppropdist/2.0
    #beamwaistlocation = 10
    
    A0 = 10.0
    widthX0 = np.sqrt(backgroundindex*alpha)*(beamwaist)/np.sqrt(beamwaist**4+(0-beamwaistlocation)**2)
    widthY0 = np.sqrt(backgroundindex*alpha)*(beamwaist)/np.sqrt(beamwaist**4+(0-beamwaistlocation)**2)
    focusX0 = -(0.5*backgroundindex*alpha)*(Z[0]-beamwaistlocation)/((0-beamwaistlocation)**2+beamwaist**4)
    focusY0 = -(0.5*backgroundindex*alpha)*(Z[0]-beamwaistlocation)/((0-beamwaistlocation)**2+beamwaist**4)
    phase0  = np.arctan((Z[0] - beamwaistlocation)/(beamwaist**2.0))
    
    #Define the array that the RK4 solns will be stored. The first column holds the IC 
    psln = np.zeros([10,nZ+1])
    
    psln[0,0]     = A0                  #A0
    psln[1,0]     = widthX0 #0.05(used in animation)  0.04                 #widthX0
    psln[2,0]     = widthY0 #0.03(used in animation)     0.02                #widthY0
    psln[3,0]     = 0 #0.04(used in animation)                   #tiltX0
    psln[4,0]     = 0 #0.1(used in animation)                   #tiltY0
    psln[5,0]     = 0.                   #positionX0
    psln[6,0]     = 0.                   #positionY0
    psln[7,0]     = focusX0  #0.005  used in animation      #focusX0
    psln[8,0]     = focusY0 #0.005(used in animation)              #focusY0
    psln[9,0]     = np.arctan((Z[0] - beamwaistlocation)/(beamwaist**2.0))                     #phase0
    # 
    fig = plt.figure(20)
    ax = fig.gca()
    cont = ax.contourf(meshX, meshY, irradiance(psln[:,0].tolist()), 50, linewidth=0, antianliased=False)
    fig.colorbar(cont, shrink=0.5, aspect=5)
    plt.title('Initial Irradiance')
    plt.show()
    

    centerirradiance_lag[0,j] = irradiance(psln[:,0].tolist())[nX//2,nY//2]
    # field_lag = np.empty([nX,nY,nZ+1],dtype=np.complex128)
    # field_lag[:,:,0] = field(psln[:,0].tolist())
    
    # Runge-Kutta 4 solver
    for ii in np.arange(nZ):
        p_ii = psln[:,ii]                #storing the current z-position soln vector 
        
        k1     = f(p_ii, params, ii)
    
        pcalc  = p_ii + (0.5*dZ*np.ones(10))*k1  #a temporary calculation used to compute the steps in runge-kutta
        k2     = f(pcalc, params, ii)
        #twok2    = (2*np.ones(10))*k2
        
        pcalc  = p_ii + (0.5*dZ*np.ones(10))*k2
        k3     = f(pcalc, params, ii)
        #twok3    = (2*np.ones(10))*k3
        
        pcalc  = p_ii + (dZ*np.ones(10))*k3
        k4     = f(pcalc, params, ii)
        
        p_iinew = p_ii + (((1./6)*dZ)*np.ones(10))*(k1+(2*np.ones(10))*k2+(2*np.ones(10))*k3+k4)
        # print("pnew",p_iinew)
        # print("sizepnew", np.size(p_iinew))
        
        psln[:,ii+1] = p_iinew
        print(p_iinew)
        input("PRESS ENTER TO CONTINUE.")
        #field_lag[:,:,ii+1] = field(p_iinew.tolist())
        centerirradiance_lag[ii+1,j] = irradiance(p_iinew.tolist())[nX//2,nY//2]
    lag_soln[:,:,j] = psln[:,:]

initial_yslice = irradiance(psln[:,0].tolist())[:,nY//2]
initial_xslice = irradiance(psln[:,0].tolist())[:,nY//2]
peak_yslice = irradiance(psln[:,nZ//2].tolist())[:,nY//2]
peak_xslice = irradiance(psln[:,nZ//2].tolist())[nX//2,:]
final_yslice = irradiance(psln[:,nZ].tolist())[:,nY//2]
final_xslice = irradiance(psln[:,nZ].tolist())[nX//2,:]

np.save('lagrangian_soln4',psln)
# np.save('lagrangian_field', field_lag)
np.save('lagrangian_centerirr4',centerirradiance_lag)
np.save('lagrangian_xslice_peak4', peak_xslice)
np.save('lagrangian_yslice_peak4', peak_yslice)
np.save('lagrangian_xslice_final4', final_xslice)
np.save('lagrangian_yslice_final4', final_yslice)
np.save('indexrealization4',indexrealization)


fig = plt.figure(21)
ax = fig.gca()
cont = ax.contourf(meshX, meshY, irradiance(psln[:,nZ//2].tolist()), 50, linewidth=0, antianliased=False)
fig.colorbar(cont, shrink=0.5, aspect=5)
plt.title('Peak Irradiance')
plt.show()

fig = plt.figure(22)
ax = fig.gca()
cont = ax.contourf(meshX, meshY, irradiance(psln[:,nZ].tolist()), 50, linewidth=0, antianliased=False)
fig.colorbar(cont, shrink=0.5, aspect=5)
plt.title('Final Irradiance')
plt.show()
    
    
fig = plt.figure(30)
plt.plot(Z,centerirradiance_lag[:,0],'-')
plt.show()

fig = plt.figure(31)
plt.plot(X,peak_yslice)
plt.title('y-slice peak irradiance')

fig = plt.figure(32)
plt.plot(Y,peak_xslice)
plt.title('x-slice peak irradiance')

fig = plt.figure(33)
plt.plot(X,final_yslice)
plt.title('y-slice final irradiance')

fig = plt.figure(34)
plt.plot(Y,final_xslice)
plt.title('x-slice final irradiance')

fig = plt.figure(35)
plt.plot(X,initial_yslice)
plt.title('y-slice Initial irradiance')

fig = plt.figure(36)
plt.plot(Y,initial_xslice)
plt.title('x-slice initial irradiance')

# np.save('lag_soln_100runs', lag_soln)
# np.save('centerirr_lag_100runs', centerirradiance_lag)