# imports
import numpy as np
import scipy as sp
#from numpy.random import multivariate_normal as np_multivariate_normal
#from scipy.stats import multivariate_normal as sp_multivariate_normal
#from numpy import linalg as LA

from Atmospheric_Propagation import Atmospheric_Propagation
from Circulant_Embedding_Generator import Circulant_Embedding_Generator


class Variational_Equations(Atmospheric_Propagation):

    def __init__(self,myRank,
                      outputFilename,
                      waveLength,
                      innerScale,
                      outerScale,
                      indexStructureConstant,
                      indexVarianceScaling,
                      backgroundIndex, 
                      apertureDiameter,
                      lengthX, 
                      numberOfPointsX,
                      lengthY, 
                      numberOfPointsY,
                      propagationDistance,
                      numberOfPointsZ,
                      transverseCharacteristicLength,
                      propagationCharacteristicLength,
                      initialFieldIdentifier
                      ):               
                
        super().__init__(myRank,
                         outputFilename,
                         waveLength,
                         innerScale,
                         outerScale,
                         indexStructureConstant,
                         indexVarianceScaling,
                         backgroundIndex, 
                         apertureDiameter,
                         lengthX, 
                         numberOfPointsX,
                         lengthY, 
                         numberOfPointsY,
                         propagationDistance,
                         numberOfPointsZ,
                         transverseCharacteristicLength,
                         propagationCharacteristicLength,
                         initialFieldIdentifier)       
        

        self.IndexRealizationGenerator = Circulant_Embedding_Generator(self.pointsX,\
                                                                       self.pointsY,\
                                                                       self.epsilon,\
                                                                       self.indexVarianceScaling)
        self.GetIndexOfRefraction()


        self.meshPointsX, self.meshPointsY           = np.meshgrid(self.pointsX , self.pointsY)
        self.meshWaveNumbersX, self.meshWaveNumbersY = np.meshgrid(self.waveNumbersX, self.waveNumbersY)
        
        self.simulatedField = np.empty([self.numberOfPointsY,self.numberOfPointsX,self.numberOfPointsZ+1], dtype=complex)
        self.exactField     = np.empty([self.numberOfPointsY,self.numberOfPointsX,self.numberOfPointsZ+1], dtype=complex)

        self.SetInitialParameters(self.computationalApertureDiameter,
                                  self.computationalPropagationDistance)

        
        return None

    def SetInitialParameters(self,
                             beamWaist,
                             beamWaistLocation):

        
        self.exactAmplitude  = np.ones(self.numberOfPointsZ+1)
        
        self.exactWidthX     = np.zeros(self.numberOfPointsZ+1)
        self.exactWidthY     = np.zeros(self.numberOfPointsZ+1)
        
        self.exactFocusX     = np.zeros(self.numberOfPointsZ+1)
        self.exactFocusY     = np.zeros(self.numberOfPointsZ+1)
        
        self.exactTiltX      = np.zeros(self.numberOfPointsZ+1)
        self.exactTiltY      = np.zeros(self.numberOfPointsZ+1)  
        
        self.exactPositionX  = np.zeros(self.numberOfPointsZ+1)
        self.exactPositionY  = np.zeros(self.numberOfPointsZ+1)         
        
        self.exactPhase      = np.zeros(self.numberOfPointsZ+1)
        
        # \frac{w_0}{\sqrt{w_0^4+\left(z-z_0\right)^2}}        
        width = beamWaist / np.sqrt( beamWaist**4.0 + (self.pointsZ - beamWaistLocation)**2.0   ) 
        # \frac{z-z_0}{(z-z_0)^2 + w_0^4}
        focus = (self.pointsZ - beamWaistLocation) / ( (self.pointsZ - beamWaistLocation)**2.0 + (beamWaist**4.0) )
            
        self.exactWidthX = np.sqrt(self.backgroundIndex * self.alpha) * width
        self.exactWidthY = np.sqrt(self.backgroundIndex * self.alpha) * width
                                                                              
        self.exactFocusX = - (self.backgroundIndex * self.alpha / 2.0) * focus                          
        self.exactFocusY = - (self.backgroundIndex * self.alpha / 2.0) * focus
                                                      
        self.exactPhase = np.arctan((self.pointsZ - beamWaistLocation)/(beamWaist**2.0))

        self.amplitude  = np.zeros(self.numberOfPointsZ+1)
        self.widthX     = np.zeros(self.numberOfPointsZ+1)        
        self.widthY     = np.zeros(self.numberOfPointsZ+1)
        self.positionX  = np.zeros(self.numberOfPointsZ+1)        
        self.positionY  = np.zeros(self.numberOfPointsZ+1)                 
        self.focusX     = np.zeros(self.numberOfPointsZ+1)
        self.focusY     = np.zeros(self.numberOfPointsZ+1)
        self.tiltX      = np.zeros(self.numberOfPointsZ+1)
        self.tiltY      = np.zeros(self.numberOfPointsZ+1)                 
        self.phase      = np.zeros(self.numberOfPointsZ+1)
                 
        self.amplitude[0]  = self.exactAmplitude[0]
        self.phase[0]      = self.exactPhase[0]

        self.widthX[0]     = self.exactWidthX[0]        
        self.widthY[0]     = self.exactWidthY[0] 

        self.positionX[0]  = self.exactPositionX[0]
        self.positionY[0]  = self.exactPositionY[0]

        self.focusX[0]     = self.exactFocusX[0]
        self.focusY[0]     = self.exactFocusY[0]

        self.tiltX[0]      = self.exactTiltX[0]
        self.tiltY[0]      = self.exactTiltY[0]                 
        

        
        for index in range(self.numberOfPointsZ+1):
            self.exactField[:,:,index] = self.GetField(self.exactAmplitude[index],
                                                       self.exactWidthX[index],        
                                                       self.exactWidthY[index],
                                                       self.exactPositionX[index],
                                                       self.exactPositionY[index],
                                                       self.exactFocusX[index],
                                                       self.exactFocusY[index],
                                                       self.exactTiltX[index],
                                                       self.exactTiltY[index],
                                                       self.exactPhase[index])
        
        
        self.simulatedField[:,:,0] = self.GetField(self.amplitude[0],
                                                   self.widthX[0],        
                                                   self.widthY[0],
                                                   self.positionX[0],
                                                   self.positionY[0],
                                                   self.focusX[0],
                                                   self.focusY[0],
                                                   self.tiltX[0],
                                                   self.tiltY[0],
                                                   self.phase[0])
        return None
        
    def GetField(self,
                 amplitude,
                 widthX,        
                 widthY,
                 positionX,
                 positionY,
                 focusX,
                 focusY,
                 tiltX,
                 tiltY,
                 phase):

        theta = 0.5* ((widthX**2.0) * ((self.meshPointsX-positionX)**2.0) + (widthY**2.0) * ((self.meshPointsY-positionY)**2.0))
        phi   = phase + tiltX * (self.meshPointsX-positionX) + tiltY * (self.meshPointsY-positionY) \
                      + focusX * ((self.meshPointsX-positionX)**2.0) + focusY * ((self.meshPointsY-positionY)**2.0)
        field  = (amplitude*np.sqrt(widthX*widthY)/np.sqrt(np.pi)) * np.exp(- (theta + 1j*phi) )                
        return field


    def RungeKutta4(self):
        parameters = np.zeros(10)
        parameters[0] = self.amplitude[0]
        parameters[1] = self.widthX[0]
        parameters[2] = self.widthY[0]
        parameters[3] = self.positionX[0]
        parameters[4] = self.positionY[0]
        parameters[5] = self.focusX[0]
        parameters[6] = self.focusY[0]
        parameters[7] = self.tiltX[0]
        parameters[8] = self.tiltY[0]
        parameters[9] = self.phase[0]
        
        for index1 in range(self.numberOfPointsZ):

            index2 = 2*index1
            #
            parameters1    = self.RKFunction(parameters,index2)
            #
            parametersTemp = parameters + (self.stepSizeZ/2.0) * parameters1
            parameters2    = self.RKFunction(parametersTemp,index2+1)
            #
            parametersTemp = parameters + (self.stepSizeZ/2.0) * parameters2
            parameters3    = self.RKFunction(parametersTemp,index2+1)
            #
            parametersTemp = parameters + self.stepSizeZ * parameters3
            parameters4    = self.RKFunction(parametersTemp,index2+2)
            parameters = parameters + (self.stepSizeZ/6.0) * (parameters1 + 2.0*parameters2 + 2.0*parameters3 + parameters4)
            #
            self.amplitude[index1+1] = parameters[0]
            self.widthX[index1+1]    = parameters[1]
            self.widthY[index1+1]    = parameters[2] 
            self.positionX[index1+1] = parameters[3]
            self.positionY[index1+1] = parameters[4]
            self.focusX[index1+1]    = parameters[5]
            self.focusY[index1+1]    = parameters[6]
            self.tiltX[index1+1]     = parameters[7]
            self.tiltY[index1+1]     = parameters[8]                 
            self.phase[index1+1]     = parameters[9]

            self.simulatedField[:,:,index1+1] = self.GetField(self.amplitude[index1+1],
                                                              self.widthX[index1+1],
                                                              self.widthY[index1+1],
                                                              self.positionX[index1+1],
                                                              self.positionY[index1+1],
                                                              self.focusX[index1+1],
                                                              self.focusY[index1+1],
                                                              self.tiltX[index1+1],
                                                              self.tiltY[index1+1],                 
                                                              self.phase[index1+1])
            print('Finshed step ',index1+1,' of ',self.numberOfPointsZ)
        return None

    def RKFunction(self,
                   parameters,
                   index):
                       
        amplitude = parameters[0]
        widthX    = parameters[1]       
        widthY    = parameters[2] 
        positionX = parameters[3]
        positionY = parameters[4] 
        focusX    = parameters[5] 
        focusY    = parameters[6]
        tiltX     = parameters[7]
        tiltY     = parameters[8]
        phase     = parameters[9]

        #field = self.GetField(amplitude,
        #                      widthX,
        #                      widthY,
        #                      positionX,
        #                      positionY,
        #                      focusX,
        #                      focusY,
        #                      tiltX,
        #                      tiltY,
        #                      phase)  
  
        
        shiftedXMesh = (self.meshPointsX-positionX) 
        shiftedXMeshSquared = shiftedXMesh**2.0
        shiftedYMesh = (self.meshPointsY-positionY)
        shiftedYMeshSquared = shiftedYMesh**2.0
       
        centralMode = ((widthX*widthY)/np.pi) * np.exp(-((widthX**2.0)*shiftedXMeshSquared + (widthY**2.0)*shiftedYMeshSquared))

        modeP  = (2.0 - (widthX**2.0) * shiftedXMeshSquared  - (widthY**2.0) * shiftedYMeshSquared) * centralMode
        modeTx = 4.0 * (widthX**2.0) * shiftedXMesh * centralMode
        modeTy = 4.0 * (widthY**2.0) * shiftedYMesh * centralMode
        modeFx = (widthX**2.0) * (1.0 - 2.0 * (widthX**2.0) * shiftedXMeshSquared) * centralMode
        modeFy = (widthY**2.0) * (1.0 - 2.0 * (widthY**2.0) * shiftedYMeshSquared) * centralMode

        #indexSheet = self.IndexOfRefractionFunction()
        indexSheet = self.indexRealizations[:,:,index]
        
        projectionA  = 0.0
        projectionWx = 0.0
        projectionWy = 0.0
        projectionPx = 0.0
        projectionPy = 0.0
        projectionFx =  ((self.gamma**2.0)/self.alpha) * self.Projection(indexSheet,modeFx)
        projectionFy =  ((self.gamma**2.0)/self.alpha) * self.Projection(indexSheet,modeFy)
        projectionTx = -(self.backgroundIndex * (self.gamma**2.0)) * self.Projection(indexSheet,modeTx)
        projectionTy = -(self.backgroundIndex * (self.gamma**2.0)) * self.Projection(indexSheet,modeTy)
        projectionP  = -((self.gamma**2.0)/self.alpha) * self.Projection(indexSheet,modeP)


        print(projectionTx,projectionTy)
        forcing = np.zeros(10)
        forcing[0] = (0.0)                                                                     + projectionA # amplitude

        forcing[1] =   (2.0/(self.backgroundIndex*self.alpha)) * focusX * widthX               + projectionWx # widthX
        forcing[2] =   (2.0/(self.backgroundIndex*self.alpha)) * focusY * widthY               + projectionWy # widthY

        forcing[3] = - 2.0 * tiltX                                                             + projectionPx # positionX
        forcing[4] = - 2.0 * tiltY                                                             + projectionPy # positionY

        forcing[5] = - (1.0/(2.0*self.backgroundIndex*self.alpha)) * widthX**4.0 \
                     + (2.0/(self.backgroundIndex*self.alpha)) * focusX**2.0                   + projectionFx # focusX

        forcing[6] = - (1.0/(2.0*self.backgroundIndex*self.alpha)) * widthY**4.0 \
                     + (2.0/(self.backgroundIndex*self.alpha)) * focusY**2.0                   + projectionFy # focusY

        forcing[7] = 0.0                                                                       + projectionTx # tiltX
        forcing[8] = 0.0                                                                       + projectionTy # tiltY

        forcing[9] = (1.0/(2.0*self.backgroundIndex*self.alpha)) * (widthX**2.0 + widthY**2.0) + projectionP # phase


        return forcing 
    
    
#    def Projection(self,
#                   vector1,
#                   vector2):
#        
#        integrand = vector1*vector2    
#        integral  = self.stepSizeX * self.stepSizeY * np.trapz(np.trapz(integrand))
#        
#        return integral
        
    def Projection(self,
                   vector1,
                   vector2):
        
        integrand = vector1*vector2
        integralOverY = sp.integrate.simps(integrand, dx=self.stepSizeY)
        projection =  sp.integrate.simps(integralOverY, dx=self.stepSizeX)

        return projection        

    def GetIndexOfRefraction(self):
  
        self.numberOfPointsZForIndex = 2*self.numberOfPointsZ + 1
        self.indexRealizations = np.zeros([self.numberOfPointsY,\
                                           self.numberOfPointsX,\
                                           self.numberOfPointsZForIndex+1], dtype=np.float64)

        for index1 in range(0,self.numberOfPointsZForIndex,2):
            [self.indexRealizations[:,:,index1], \
             self.indexRealizations[:,:,index1+1]] = self.IndexRealizationGenerator.GetRealization()

        #for index1 in range(0,self.numberOfPointsZForIndex,2):
        #    self.indexRealizations[:,:,index1] = np.exp(- 2.0*(self.meshPointsX**2.0 + self.meshPointsY**2.0) / \
        #                                                      (self.computationalApertureDiameter**2.0)
        #                                                  )
        #    self.indexRealizations[:,:,index1+1] = np.exp(- 2.0*(self.meshPointsX**2.0 + self.meshPointsY**2.0) / \
        #                                                      (self.computationalApertureDiameter**2.0)
        #                                                  )
        return None

        
    def IndexOfRefractionFunction(self):
        indexSheet =  np.exp(- 2.0*(self.meshPointsX**2.0 + self.meshPointsY**2.0) / \
                                   (self.computationalApertureDiameter**2.0)
                            )
        return indexSheet        


    
    
    
      
    
 
    
