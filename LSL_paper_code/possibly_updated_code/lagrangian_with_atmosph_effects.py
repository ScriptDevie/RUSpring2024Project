#Scaling Laws with random effects of the atmosphere
#the randomness appears in the representation of the index of refraction 

import numpy as np
#from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import time


#####################################
####### DEFINE ALL PARAMETERS #######
#####################################

#define parameters with dimension- use these to define the dimensionless parameters 
wavelength =   1e-6
innerscale =   1e-2
outerscale =    1e2
indexstructureconstant =  np.sqrt(5.0) * 1e-7 #1e-10   
indexvariancescaling =      1      
backgroundindex = 1+1e-6
aperturediameter  = 2e-1
lengthX =      2.0
nX =   200              #number of points taken in X direction 
lengthY =      2.0
nY =   200              #number of points taken in the Y direction 
propdist  =2e4
nZ =   500              #number of points taken in Z direction (also same as number of random fields generated)

#indexstandarddeviation = 0 #set to no random effects, currently
indexstandarddeviation = (indexstructureconstant/np.sqrt(2.0))*\
                                ((indexvariancescaling * outerscale)**(1.0/3.0))  # (sigma_n)
#gamma = innerscale*wavenumber*np.sqrt(indexstandarddeviation)

transverseCharacteristicLength  = aperturediameter
propagationCharacteristicLength = propdist/100
# innerscale = transverseCharacteristicLength
# outerscale = propagationCharacteristicLength

wavenumber = (2.0)*np.pi/wavelength

#define dimensionless parameters

#indexstandarddeviation = 1e-6
epsilon = innerscale/outerscale
alpha = ((innerscale**2)*wavenumber)/outerscale
gamma = innerscale*wavenumber*np.sqrt(indexstandarddeviation)
#gamma = 0
nu    = gamma * indexstandarddeviation
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

###############################################################
####### GENERATE RANDOM FIELDS FOR INDEX OF REFRACTION ########
###############################################################
#FIRST RUN THE INDEX_REALIZATION_3.py FILES TO SAVE A SET OF REALIZATIONS
#make sure all parameters above match.... should come up with a better way to just be able to run with params defined here..
indexrealization = np.load('indexrealizations2.npy')

##DEFINE THE FIELD AND IRRADIENCE FUNCTS

def field(p):
    A, widthX, widthY, tiltX, tiltY, positionX, positionY, focusX, focusY, phase = p
    theta = 0.5*((widthX**2)*(meshX - positionX)**2 + (widthY**2)*(meshY - positionY)**2)
    phi = phase + tiltX*(meshX - positionX) + tiltY*(meshY - positionY) + focusX*(meshX - positionX)**2 + focusY*(meshY - positionY)**2
    field = (A*np.sqrt(widthX*widthY)/np.sqrt(np.pi))*np.exp(-(theta +1j*phi))
    
    return field

def irradience(p):
    field_data = field(p)
    irradience = (abs(field_data))**2
    
    return irradience

###########################################
####### SET UP ODE SYSTEM AND SOLVE #######
###########################################

#define parameters for ODEint
params = [backgroundindex, alpha, gamma]

#define a projection for the random terms
def projection(indexrealization,mode, dX, dY):
    #need indexrealization and mode to be vectors- not multidim arrays... 
    integrand = indexrealization*mode
    
    proj = dX*dY*(np.trapz(np.trapz(integrand)))
    return proj

#define RHS of ODEs-- THIS REQUIRES INDEX FIELD REALIZATIONS

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


#define initial conditions
#define the focus and width ICs based off of exact gaussian beam soln??
## this way will allow us to specify the beamwaist and the beamwaist location?!
#beamwaist_physical = 0.01
#beamwaist = (8/(alpha*backgroundindex))*beamwaist_physical #this is for trying to match Laurence's code
beamwaist = compaperture
beamwaistlocation = comppropdist/2.0
#beamwaistlocation = 10

#these new ICs did not really do anything to help with prediciting the location of the beamwaist and the size
#in fact, the beamwaist location was VERY off. when specified for 100, it occured at 1 meter..
widthX0 = np.sqrt(backgroundindex*alpha)*(beamwaist)/np.sqrt(beamwaist**4+(0-beamwaistlocation)**2)
widthY0 = np.sqrt(backgroundindex*alpha)*(beamwaist)/np.sqrt(beamwaist**4+(0-beamwaistlocation)**2)
focusX0 = -(0.5*backgroundindex*alpha)*(Z[0]-beamwaistlocation)/((0-beamwaistlocation)**2+beamwaist**4)
focusY0 = -(0.5*backgroundindex*alpha)*(Z[0]-beamwaistlocation)/((0-beamwaistlocation)**2+beamwaist**4)
phase0  = np.arctan((Z[0] - beamwaistlocation)/(beamwaist**2.0))

#Define the array that the RK4 solns will be stored. The first column holds the IC 
psln = np.zeros([10,nZ+1])

psln[0,0]     = 10.                  #A0
psln[1,0]     = widthX0 #0.05(used in animation)  0.04                 #widthX0
psln[2,0]     = widthY0 #0.03(used in animation)     0.02                #widthY0
psln[3,0]     = 0 #0.04(used in animation)                   #tiltX0
psln[4,0]     = -0 #0.1(used in animation)                   #tiltY0
psln[5,0]     = 0.                   #positionX0
psln[6,0]     = 0.                   #positionY0
psln[7,0]     = focusX0  #0.005  used in animation      #focusX0
psln[8,0]     = focusY0 #0.005(used in animation)              #focusY0
psln[9,0]     = 0                   #phase0

# fig = plt.figure(20)
# ax = fig.gca()
# cont = ax.contourf(meshX, meshY, irradience(psln[:,0].tolist()), 50, linewidth=0, antianliased=False)
# fig.colorbar(cont, shrink=0.5, aspect=5)
# plt.title('Initial Irradience')
# plt.show()

#start_time = time.clock()
#Write a Runge-Kutta 4 solver
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
    k4     = f(p_ii, params, dX, dY, meshX, meshY, indexrealization, ii)
    
    p_iinew = p_ii + (((1/6)*dZ)*np.ones(10))*(k1+(2*np.ones(10))*k2+(2*np.ones(10))*k3+k4)
   # print("pnew",p_iinew)
   # print("sizepnew", np.size(p_iinew))
    
    psln[:,ii+1] = p_iinew
    

# #print("--- %s seconds ---" % (time.clock() - start_time))
# 
fig = plt.figure(23)
plt.plot(Z, psln[5,:])
plt.plot(Z, psln[6,:])
plt.title('Beam center location')
plt.show()

#for beams without tilt parameters, plot center irradience
centerirradience = np.zeros(nZ)
for iii in np.arange(nZ):
    centerirradience[iii] = irradience(psln[:,iii].tolist())[nX//2, nY//2]

fig = plt.figure(25)
plt.plot(Z[0:-1], centerirradience)
plt.title('centerline irradience (at 0,0)')
plt.show()

initialenergy = np.sum(np.sum(irradience(psln[:,0])))

irradience_field = np.zeros(nZ, nX, nY)
energydist = np.zeros(nZ)
for iii in np.arange(nZ):
    energydist[iii] = np.sum(np.sum(irradience(psln[:,iii+1].tolist())))/initialenergy

fig = plt.figure(24)
plt.plot(Z[0:-1], energydist)
plt.title('Energy')
plt.show()

# 
# #set up writer for animation
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=20, bitrate=1800)
# 
# # set figure with empty contour plot
# fig = plt.figure(3)
# ax = plt.gca()
#     
# def animate(i):
#     irradience_data = irradience(psln[:,i].tolist())
#     cont = ax.contourf(meshX, meshY, irradience_data, 50, linewidth=0, antianliased = False)
#     #fig.colorbar(cont, shrink=0.5, aspect = 5)
#     return cont,
# 
# anim = animation.FuncAnimation(fig, animate, frames = nZ)
# 
# anim.save('lagrangianwithatm_asymetrical_focusing_tilt.mp4', writer=writer)

