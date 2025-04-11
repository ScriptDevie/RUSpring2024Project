import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import moment

#file for making nice plots

#load data for paraxial solver
paraxial_xslice_peak=np.load('paraxial_xslice_peak3.npy')
paraxial_yslice_peak=np.load('paraxial_yslice_peak3.npy')
paraxial_xslice_final=np.load('paraxial_xslice_final3.npy')
paraxial_yslice_final=np.load('paraxial_yslice_final3.npy')
paraxial_centerirr=np.load('paraxial_centerirr3.npy')
paraxial_recordedfield=np.load('paraxial_fields3.npy')

#load data for lagrangian solver
lagrangian_xslice_peak=np.load('lagrangian_xslice_peak3.npy')
lagrangian_yslice_peak=np.load('lagrangian_yslice_peak3.npy')
lagrangian_xslice_final=np.load('lagrangian_xslice_final3.npy')
lagrangian_yslice_final=np.load('lagrangian_yslice_final3.npy')
lagrangian_soln=np.load('lagrangian_soln3.npy')
# lagrangian_field=np.load('lagrangian_field.npy')
lagrangian_centerirr=np.load('lagrangian_centerirr3.npy')


#load spatial discretization
pointsX = np.load('pointsX.npy')
pointsY = np.load('pointsY.npy')
pointsZ = np.load('pointsZ.npy')

meshX, meshY = np.meshgrid(pointsX, pointsY)

def field(p):
    A, widthX, widthY, tiltX, tiltY, positionX, positionY, focusX, focusY, phase = p
    theta = 0.5*((widthX**2)*(meshX - positionX)**2 + (widthY**2)*(meshY - positionY)**2)
    phi = phase + tiltX*(meshX - positionX) + tiltY*(meshY - positionY) + focusX*(meshX - positionX)**2 + focusY*(meshY - positionY)**2
    field = (A*np.sqrt(widthX*widthY)/np.sqrt(np.pi))*np.exp(-(theta +1j*phi))
    
    return field

def irradiance(p):
    field_data = field(p)
    irradiance = (np.abs(field_data))**2
    
    return irradiance

#compute some errors
#center irradiance error
center_err_2norm = np.linalg.norm(paraxial_centerirr-lagrangian_centerirr)/np.linalg.norm(paraxial_centerirr)
print('center err 2norm',center_err_2norm)
center_err_infnorm = np.linalg.norm(paraxial_centerirr-lagrangian_centerirr, np.inf)/np.linalg.norm(paraxial_centerirr, np.inf)
print('center err inf', center_err_infnorm)

#irradiance slice errors
peakslicex_err_2norm = np.linalg.norm(paraxial_xslice_peak-lagrangian_xslice_peak)/np.linalg.norm(paraxial_xslice_peak)
print('peak x 2err',peakslicex_err_2norm)

peakslicey_err_2norm = np.linalg.norm(paraxial_yslice_peak-lagrangian_yslice_peak)/np.linalg.norm(paraxial_yslice_peak)
print('peak y 2err', peakslicey_err_2norm)

finalslicex_err_2norm = np.linalg.norm(paraxial_xslice_final-lagrangian_xslice_final)/np.linalg.norm(paraxial_xslice_final)
print('final x 2err',finalslicex_err_2norm)
finalslicey_err_2norm = np.linalg.norm(paraxial_yslice_final-lagrangian_yslice_final)/np.linalg.norm(paraxial_yslice_final)
print('final y 2err', finalslicey_err_2norm)

fieldnormfro= np.zeros([np.size(pointsZ),1])
fieldnorminf= np.zeros([np.size(pointsZ),1])
irrnormfro = np.zeros([np.size(pointsZ),1])
irrnorminf = np.zeros([np.size(pointsZ),1])

# #field and irradiance errors 
# for i in np.arange(np.size(pointsZ)):
#     lagfield = field(lagrangian_soln[:,i].tolist())
#     fieldnormfro[i] = np.linalg.norm(lagfield-paraxial_recordedfield[:,:,i])/np.linalg.norm(paraxial_recordedfield[:,:,i])
#     fieldnorminf[i] = np.linalg.norm(lagfield-paraxial_recordedfield[:,:,i],np.inf)/np.linalg.norm(paraxial_recordedfield[:,:,i],np.inf)
#     irrnormfro[i] = np.linalg.norm(np.abs(lagfield)**2-np.abs(paraxial_recordedfield[:,:,i])**2)/np.linalg.norm(np.abs(paraxial_recordedfield[:,:,i])**2)
#     irrnorminf[i] = np.linalg.norm(np.abs(lagfield)**2-np.abs(paraxial_recordedfield[:,:,i])**2,np.inf)/np.linalg.norm(np.abs(paraxial_recordedfield[:,:,i])**2,np.inf)


#make some plots

#irradiance plots
fig = plt.figure(20)
ax = fig.gca()
cont = ax.contourf(meshX, meshY, irradiance(lagrangian_soln[:,0].tolist()), 50, linewidth=0, antianliased=False)
fig.colorbar(cont, shrink=0.5, aspect=5)
plt.title('Initial Irradiance',fontsize=12)
plt.show()

fig = plt.figure(21)
ax = fig.gca()
cont = ax.contourf(meshX, meshY, irradiance(lagrangian_soln[:,500].tolist()), 50, linewidth=0, antianliased=False)
fig.colorbar(cont, shrink=0.5, aspect=5)
plt.title('Peak Lagrangian Irradiance',fontsize=12)
plt.show()


parax_peakirr = np.abs(paraxial_recordedfield[:,:,500])**2

fig = plt.figure(22)
ax = fig.gca()
cont = ax.contourf(meshX, meshY, parax_peakirr, 50, linewidth=0, antianliased=False)
fig.colorbar(cont, shrink=0.5, aspect=5)
plt.title('Peak paraxial Helmholtz Irradiance',fontsize=12)
plt.show()


#plot the irradiance slices 
fig = plt.figure(1)
plt.plot(pointsX,paraxial_xslice_peak,'-',pointsX, lagrangian_xslice_peak,'--')
plt.legend(('Paraxial Helmholtz Eqn', 'Lagrangian Scaling Law'),
            loc='best', shadow=True)
plt.xlabel('y',  fontsize=12)
plt.ylabel('Peak Irradiance x-Slice',  fontsize=12)
plt.grid(True)
plt.show()

fig = plt.figure(2)
plt.plot(pointsY,paraxial_yslice_peak,'-',pointsY, lagrangian_yslice_peak,'--')
plt.legend(('Paraxial Helmholtz Eqn', 'Lagrangian Scaling Law'),
            loc='best', shadow=True)
plt.xlabel('x',  fontsize=12)
plt.ylabel('Peak Irradiance y-Slice', fontsize=12)
plt.grid(True)
plt.show()

fig = plt.figure(3)
plt.plot(pointsX,paraxial_xslice_final,'-',pointsX, lagrangian_xslice_final,'--')
plt.legend(('Paraxial Helmholtz Eqn', 'Lagrangian Scaling Law'),
            loc='best', shadow=True)
plt.xlabel('y')
plt.ylabel('Final Irradiance x-Slice',  fontsize=12)
plt.grid(True)
plt.show()

fig = plt.figure(4)
plt.plot(pointsY,paraxial_yslice_final,'-',pointsY, lagrangian_yslice_final,'--')
plt.legend(('Paraxial Helmholtz Eqn', 'Lagrangian Scaling Law'),
            loc='best', shadow=True)
plt.xlabel('x',  fontsize=12)
plt.ylabel('Final Irradiance y-Slice',  fontsize=12)
plt.grid(True)
plt.show()

#plot center irradiance
fig = plt.figure(5)
plt.plot(pointsZ,lagrangian_centerirr,'-',pointsZ, paraxial_centerirr,'--')
plt.legend(('Paraxial Helmholtz Eqn', 'Lagrangian Scaling Law'),
            loc='best', shadow=True)
plt.xlabel('z',  fontsize=12)
plt.ylabel('Center Irradiance',  fontsize=12)
plt.grid(True)
plt.show()

# #plot the field and irradiance error as a function of z
# fig = plt.figure(6)
# plt.plot(pointsZ,fieldnormfro,'-',pointsZ,fieldnorminf,'--')
# plt.legend(('Relative Error in Frobenius Norm','Relative Error in Infinity Norm'),loc='best',shadow=True)
# plt.xlabel('z',fontsize =12)
# plt.ylabel('Relative Error of Field', fontsize=12)
# plt.grid(True)
# plt.show()
# 
# fig = plt.figure(7)
# plt.plot(pointsZ,irrnormfro,'-',pointsZ,irrnorminf,'--')
# plt.legend(('Relative Error in Frobenius Norm','Relative Error in Infinity Norm'),loc='best',shadow=True)
# plt.xlabel('z',fontsize =12)
# plt.ylabel('Relative Error of Irradiance', fontsize=12)
# plt.grid(True)
# plt.show()



# #compute moments
# ##compute some moments?
# firstmoment = np.zeros(1000)
# secondmoment = np.zeros(1000)
# #secondmoment2 = np.zeros(1000)
# thirdmoment = np.zeros(1000)
# fourthmoment = np.zeros(1000)
# 
# for jjj in np.arange(1000):
#     firstmoment[jjj]  = moment(np.ravel(np.abs((lagrangian_field[:,:,jjj].tolist()))**2), moment=1)
#     secondmoment[jjj] = moment(np.ravel(np.abs((lagrangian_field[:,:,jjj].tolist()))**2), moment=2)
#     #secondmoment2[jjj] = moment(np.ravel(lagrangian_field[:,:,jjj]),moment=2)
#     thirdmoment[jjj]  = moment(np.ravel(np.abs((lagrangian_field[:,:,jjj].tolist()))**2), moment=3)
#     fourthmoment[jjj] = moment(np.ravel(np.abs((lagrangian_field[:,:,jjj].tolist()))**2), moment=4)
# 
# firstmomenthelm = np.zeros(1001)
# secondmomenthelm = np.zeros(1001)
# #secondmomenthelm2 = np.zeros(1001)
# thirdmomenthelm = np.zeros(1001)
# fourthmomenthelm = np.zeros(1001)
# 
# for jjj in np.arange(1001):
#     firstmomenthelm[jjj] = moment(np.ravel(np.abs((paraxial_recordedfield[:,:,jjj]))**2), moment=1)
#     secondmomenthelm[jjj] = moment(np.ravel(np.abs((paraxial_recordedfield[:,:,jjj]))**2),moment=2)
#    # secondmomenthelm2[jjj] = moment(np.ravel(paraxial_recordedfield[:,:,jjj]),moment=2)
#     thirdmomenthelm[jjj] =  moment(np.ravel(np.abs((paraxial_recordedfield[:,:,jjj]))**2), moment=3)
#     fourthmomenthelm[jjj] = moment(np.ravel(np.abs((paraxial_recordedfield[:,:,jjj]))**2), moment =4)
# 
# fig = plt.figure(6)
# plt.plot(pointsZ[1:1001], np.abs(secondmomenthelm[1:1001]-secondmoment))
# plt.xlabel('z')
# plt.ylabel('Difference between Second Moments')
# plt.grid(True)
# plt.show()
# 
# fig = plt.figure(7)
# plt.plot(pointsZ[1:1001],secondmomenthelm[1:1001],'-', pointsZ[1:1001],secondmoment,'--')
# plt.legend(('Paraxial Helmholtz Second Moment', 'Lagrangian Scaling Law Second Moment'),loc='best', shadow =True)
# plt.xlabel('z')
# plt.ylabel('Second Moment')
# plt.grid(True)
# plt.show()