import numpy as np
import scipy as sp
import scipy.special
import time
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
#from matplotlib import cm
#from matplotlib.colors import LogNorm
#from mpl_toolkits.mplot3d import axes3d

from Atmospheric_Propagation import Atmospheric_Propagation
from Circulant_Embedding_Generator import Circulant_Embedding_Generator
#import fresnelDiffractionIntegral as fdi

#import collections
#from contextlib import contextmanager



class Paraxial_Equation(Atmospheric_Propagation):
    """
    This is a subclass of Atmospheric_Simulation

    """

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
                      initialFieldIdentifier):

        """set all dimensional parameters and calculate the nondimensional parameters"""

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

        self.meshPointsX, self.meshPointsY           = np.meshgrid(self.pointsX , self.pointsY)
        self.meshWaveNumbersX, self.meshWaveNumbersY = np.meshgrid(self.waveNumbersX, self.waveNumbersY)


        self.IndexRealizationGenerator = Circulant_Embedding_Generator(self.pointsX,\
                                                                       self.pointsY,\
                                                                       self.epsilon,\
                                                                       self.indexVarianceScaling)
        self.GetIndexOfRefraction()
        

        self.recordedFieldFFTIntgrator = np.empty([self.numberOfPointsX,\
                                                   self.numberOfPointsY,\
                                                   self.numberOfPointsZ+1], dtype=np.complex128)
        self.recordedFieldFFTIntgrator[:,:,0] = self.SetInitialField()

        self.fieldEnergyFFTIntgrator  = np.empty(self.numberOfPointsZ+1, dtype=np.float64)
        self.fieldEnergyFFTIntgrator[0] = self.Energy(self.recordedFieldFFTIntgrator[:,:,0])



        self.recordedFieldDirectIntgrator = np.empty([self.numberOfPointsX,\
                                                      self.numberOfPointsY,\
                                                      self.numberOfPointsZ+1], dtype=np.complex128)
        self.recordedFieldDirectIntgrator[:,:,0] = self.SetInitialField()

        self.fieldEnergyDirectIntgrator  = np.empty(self.numberOfPointsZ+1, dtype=np.float64)
        self.fieldEnergyDirectIntgrator[0] = self.Energy(self.recordedFieldDirectIntgrator[:,:,0])
        self.madeStoreInOutMatrix = False


        self.recordedFieldRK4IF = np.empty([self.numberOfPointsX,\
                                                      self.numberOfPointsY,\
                                                      self.numberOfPointsZ+1], dtype=np.complex128)
        self.recordedFieldRK4IF[:,:,0] = self.SetInitialField()
        self.fieldEnergyRK4IF  = np.empty(self.numberOfPointsZ+1, dtype=np.float64)
        self.fieldEnergyRK4IF[0] = self.Energy(self.recordedFieldDirectIntgrator[:,:,0])
        self.wrappedFieldFFT = np.fft.fftshift(np.fft.fft2(self.recordedFieldRK4IF[:,:,0]))

        #self.timer = fdi.Timer([], [], [])

        return None
    
#*****************************************************************************************************************
# Functions to set the initial condition
#*****************************************************************************************************************
    def SetInitialField(self):
        computationalApertureRadius = self.computationalApertureDiameter/2.0
        computationalApertureRadiusSquared = computationalApertureRadius**2.0
        initialField = np.empty([self.numberOfPointsX,\
                                 self.numberOfPointsY], dtype=np.complex128)
        if (self.initialFieldIdentifier == "const"):
            amplitude = np.sqrt(1.0/(np.pi*computationalApertureRadiusSquared))
            for index2 in range(self.numberOfPointsY):
                for index1 in range(self.numberOfPointsX):
                    radiusSquared = pow(self.pointsX[index1],2.0) + pow(self.pointsY[index2],2.0)
                    if (radiusSquared < computationalApertureRadiusSquared):
                        initialField[index1,index2] = np.complex(amplitude,0.0)
                    else:
                        initialField[index1,index2] = np.complex(0.0,0.0)
                        
                    
        if (self.initialFieldIdentifier == "debug"):
            initialField = (1.0/(np.sqrt(np.pi)*self.computationalApertureDiameter)) * \
            np.exp(-(self.meshPointsX**2.0 + self.meshPointsY**2.0)/\
                    (2.0 * self.computationalApertureDiameter**2.0)
                  )

        if (self.initialFieldIdentifier == "tilt"):
            initialField = (1.0/(np.sqrt(np.pi)*computationalApertureRadius)) * \
            np.exp(-(self.meshPointsX**2.0 + self.meshPointsY**2.0)/(2.0 * computationalApertureRadius**2.0)
                   + 1j * 2*np.pi/self.computationalApertureDiameter * self.meshPointsX 
                  )     

           
        if (self.initialFieldIdentifier == "lagra"):

            # \frac{w_0}{\sqrt{w_0^4+\left(z-z_0\right)^2}}      
            
            beamWaist         = self.computationalApertureDiameter
            beamWaistLocation = self.computationalPropagationDistance/2.0
                                  
                                  
            width = beamWaist / np.sqrt( beamWaist**4.0 + (self.pointsZ[0] - beamWaistLocation)**2.0 ) 
            # \frac{z-z_0}{(z-z_0)^2 + w_0^4}
            focus = (self.pointsZ[0] - beamWaistLocation) / ( (self.pointsZ[0] - beamWaistLocation)**2.0 + (beamWaist**4.0) )
                
            widthX = np.sqrt(self.backgroundIndex * self.alpha) * width
            widthY = np.sqrt(self.backgroundIndex * self.alpha) * width
                                                                                  
            focusX = - (self.backgroundIndex * self.alpha / 2.0) * focus                          
            focusY = - (self.backgroundIndex * self.alpha / 2.0) * focus
                                                          
            phase = np.arctan((self.pointsZ[0] - beamWaistLocation)/(beamWaist**2.0))  
        
            amplitude = 1.0

            positionX = 0.0
            positionY = 0.0

            tiltX     = 0.0
            tiltY     = 0.0  
    
            theta = 0.5* ((widthX**2.0) * ((self.meshPointsX-positionX)**2.0) + (widthY**2.0) * ((self.meshPointsY-positionY)**2.0))
            phi   = phase + tiltX * (self.meshPointsX-positionX) + tiltY * (self.meshPointsY-positionY) \
                          + focusX * ((self.meshPointsX-positionX)**2.0) + focusY * ((self.meshPointsY-positionY)**2.0)
            initialField  = (amplitude*np.sqrt(widthX*widthY)/np.sqrt(np.pi)) * np.exp(- (theta + 1.j*phi) )

        return initialField
        



#*****************************************************************************************************************
# Auxiliary Functions
#*****************************************************************************************************************


    def Energy(self,field):

        integralOverY = sp.integrate.simps(np.abs(field)**2.0, dx=self.stepSizeY)
        energy =  sp.integrate.simps(integralOverY, dx=self.stepSizeX)

        return energy




#*****************************************************************************************************************
# Index of Refraction Functions
#*****************************************************************************************************************



    def GetIndexOfRefraction(self):
  
        #self.numberOfPointsZForIndex = 2*self.numberOfPointsZ + 1
        #self.indexRealizations = np.zeros([self.numberOfPointsY,\
        #                                   self.numberOfPointsX,\
        #                                   self.numberOfPointsZForIndex+1], dtype=np.float64)
        self.indexRealizations = np.zeros([self.numberOfPointsX,\
                                           self.numberOfPointsY,\
                                           self.numberOfPointsZ+1], dtype=np.float64)
        for index1 in range(self.numberOfPointsZ+1):
                self.indexRealizations[:,:,index1] = (1.0+0.5*np.cos(self.pointsZ[index1])) *\
                                                       np.exp(- 2.0*(self.meshPointsX**2.0 + self.meshPointsY**2.0) / \
                                                                    (self.computationalApertureDiameter**2.0)
                                                             )

        #for index1 in range(0,self.numberOfPointsZForIndex,2):
        #    [self.indexRealizations[:,:,index1], \
        #     self.indexRealizations[:,:,index1+1]] = self.IndexRealizationGenerator.GetRealization()


        #for index1 in range(0,self.numberOfPointsZForIndex,2):
        #    self.indexRealizations[:,:,index1] = np.exp(- 2.0*(self.meshPointsX**2.0 + self.meshPointsY**2.0) / \
        #                                                      (self.computationalApertureDiameter**2.0)
        #                                               )
        #    self.indexRealizations[:,:,index1+1] = np.exp(- 2.0*(self.meshPointsX**2.0 + self.meshPointsY**2.0) / \
        #                                                    (self.computationalApertureDiameter**2.0)
        #                                             )
        return None

    def IndexOfRefractionFunction(self):
        #indexSheet =  np.exp(- 2.0*(self.meshPointsX**2.0 + self.meshPointsY**2.0) / \
        #                           (self.computationalApertureDiameter**2.0)
        #                    )
        indexSheet =  np.zeros([self.numberOfPointsX,\
                                self.numberOfPointsY], dtype=np.float64)
        return indexSheet


#*****************************************************************************************************************
# Pseudo-Spectral functions
#*****************************************************************************************************************

  
    def RK4IF(self):
        
        for index1 in range(self.numberOfPointsZ):
            #print('Starting step ',index1,' of ',self.numberOfPointsZ)

            index2 = 2*index1
                       
            K1 = self.RK4IF_Function(self.wrappedFieldFFT,
                                     self.pointsZ[index1],
                                     index2)
            #print("The forcing at:", self.pointsZ[index1],'Max K1:',np.max(K1)) 
            K2 = self.RK4IF_Function(self.wrappedFieldFFT + (self.stepSizeZ/2.0) * K1,
                                     self.pointsZ[index1] + self.stepSizeZ/2.0,
                                     index2 + 1)
            #print("The forcing at:", self.pointsZ[index1] + self.stepSizeZ/2.0,'Max K2:',np.max(K2)) 
    
            
            K3 = self.RK4IF_Function(self.wrappedFieldFFT + (self.stepSizeZ/2.0) * K2,
                                     self.pointsZ[index1] + self.stepSizeZ/2.0,
                                     index2 + 1)
            #print("The forcing at:", self.pointsZ[index1] + self.stepSizeZ/2.0,'Max K3:',np.max(K3)) 
    
            K4 = self.RK4IF_Function(self.wrappedFieldFFT + (self.stepSizeZ) * K3 ,
                                     self.pointsZ[index1] + self.stepSizeZ,
                                     index2 + 2)
            #print("The forcing at:", self.pointsZ[index1] + self.stepSizeZ,'Max K4:',np.max(K4)) 
    
                           
            self.addition = (self.stepSizeZ/6.0)*(K1 + 2.0*K2 + 2.0*K3 + K4)
            #print("The max of addition is:", np.max(self.addition))        
             
            self.wrappedFieldFFT +=  self.addition
            
                                                   
            self.fieldFFT = np.exp(-(1.j/(2.0*self.backgroundIndex*self.alpha))*\
                                    (self.meshWaveNumbersX**2.0 + self.meshWaveNumbersY**2.0)*\
                                    (self.pointsZ[index1] + self.stepSizeZ)\
                                  )*self.wrappedFieldFFT
    
            self.recordedFieldRK4IF[:,:,index1+1] = np.fft.ifft2(np.fft.ifftshift(self.fieldFFT))

            self.fieldEnergyRK4IF[index1+1] = self.Energy(self.recordedFieldRK4IF[:,:,index1+1])

            print("Finshed step ",index1+1," of ",self.numberOfPointsZ)
            print("RK4 Energy: ",self.fieldEnergyRK4IF[index1+1])
                                                                                                            
        return None 
    
    def RK4IF_Function(self,
                       wfFFT,
                       zPoint,
                       realizationIndex):
        
        fFFT = np.exp(-(1.j/(2.0*self.backgroundIndex*self.alpha)) * \
                       (self.meshWaveNumbersX**2.0 + self.meshWaveNumbersY**2.0) * \
                       (zPoint) \
                     ) * wfFFT

        f = np.fft.ifft2(np.fft.ifftshift(fFFT))
        indexRealization = self.IndexOfRefractionFunction()
        fun = np.fft.fftshift(np.fft.fft2(indexRealization  * f))
            
        forcing = 1.j * (self.gamma**2.0/self.alpha) * \
                  np.exp( (1.j/(2.0*self.backgroundIndex*self.alpha))*\
                          (self.meshWaveNumbersX**2.0 + self.meshWaveNumbersY**2.0)*\
                          (zPoint)\
                        ) * fun
        #print(np.max(np.abs( forcing - np.complex(0.0,(self.gamma*self.gamma/self.alpha)) * wfFFT)) )                   
        return forcing
    



#*****************************************************************************************************************
# Direct Approach 
#*****************************************************************************************************************

    def DirectIntegratorStellar(self,field,stepSize):
        constant = np.sqrt((self.backgroundIndex * self.alpha)/(np.pi * stepSize))
        returnField = fdi.computeIntegral(field, self.pointsX, self.pointsY, self.pointsX, self.pointsY, constant)
        returnField *= -1.j * returnField
        return returnField

    def DirectIntegrator(self,field,stepSize):
        constant = np.sqrt((self.backgroundIndex * self.alpha)/(np.pi * stepSize))
        returnField = np.empty([self.numberOfPointsX,\
                                self.numberOfPointsY], dtype=np.complex128)
        for indexY in range(self.numberOfPointsY):
            firstDot = np.dot(field,self.inOutMatrixY[:,indexY])
            for indexX in range(self.numberOfPointsX):
                returnField[indexX,indexY] = np.dot(self.inOutMatrixX[:,indexX],firstDot) 
                
        returnField *= -1.j * (constant**2.0) / 2.0
        return returnField

    def DefiniteIntegral(self,evaluationPoint,\
                              locationInSecondPlane,\
                              locationInFirstPlane,\
                              stepSize):
        """  evaluationPoint = x_{j} or x_{j+1};
             locationInSecondPlane = x^hat, beta;  
             locationInFirstPlane = x_{j} or x_{j+1}, gamma 
        """
        constant = np.sqrt((self.backgroundIndex * self.alpha)/(np.pi * stepSize))
        argument = constant * (evaluationPoint - locationInSecondPlane)

        (S, C) = scipy.special.fresnel(argument)
        fresnelTerm = -((locationInFirstPlane - locationInSecondPlane) / constant) * (C + 1.j*S)
        
        expTerm = -(1.j / (np.pi * constant**2.0)) * np.exp(1.j * (np.pi/2.0) * (argument**2.0))

        result = expTerm + fresnelTerm
        if np.isnan(result):
            raise ValueError("DefiniteIntegral({}, {}, {}) is nan".format(evaluationPoint, \
                                                                          locationInSecondPlane, \
                                                                          locationInFirstPlane))
        return result

    def MakeInOutMatrix(self,inputPoints,\
                             outputPoints,\
                             stepSize):

        numberOfInputPoints = len(inputPoints)
        numberOfOutputPoints = len(outputPoints)
        inOutMatrix = np.zeros((numberOfInputPoints, numberOfOutputPoints), dtype=np.complex128)
            

        for indexOutput in range(numberOfOutputPoints):
            inOutMatrix[0,\
                        indexOutput] = (self.DefiniteIntegral(inputPoints[1],\
                                                              outputPoints[indexOutput],\
                                                              inputPoints[1],\
                                                              stepSize) - \
                                        self.DefiniteIntegral(inputPoints[0],\
                                                              outputPoints[indexOutput],\
                                                              inputPoints[1],\
                                                              stepSize)\
                                       )/(inputPoints[1]-inputPoints[0])



            for indexInput in range(1,numberOfInputPoints-1):
                inOutMatrix[indexInput,\
                            indexOutput] = ((self.DefiniteIntegral(inputPoints[indexInput+1],\
                                                                   outputPoints[indexOutput],\
                                                                   inputPoints[indexInput+1],\
                                                                   stepSize) - \
                                             self.DefiniteIntegral(inputPoints[indexInput],\
                                                                   outputPoints[indexOutput],\
                                                                   inputPoints[indexInput+1],\
                                                                   stepSize)\
                                            )/(inputPoints[indexInput+1]-inputPoints[indexInput])\
                                           )\
                                           -\
                                           ((self.DefiniteIntegral(inputPoints[indexInput],\
                                                                   outputPoints[indexOutput],\
                                                                   inputPoints[indexInput-1],\
                                                                   stepSize) - \
                                             self.DefiniteIntegral(inputPoints[indexInput-1],\
                                                                   outputPoints[indexOutput],\
                                                                   inputPoints[indexInput-1],\
                                                                   stepSize)\
                                            )/(inputPoints[indexInput]-inputPoints[indexInput-1])\
                                           )

            inOutMatrix[numberOfInputPoints-1,\
                        indexOutput] = -(self.DefiniteIntegral(inputPoints[numberOfInputPoints-1],\
                                                               outputPoints[indexOutput],\
                                                               inputPoints[numberOfInputPoints-2],\
                                                               stepSize) - \
                                         self.DefiniteIntegral(inputPoints[numberOfInputPoints-2],\
                                                               outputPoints[indexOutput],\
                                                               inputPoints[numberOfInputPoints-2],\
                                                               stepSize) \
                                        )/(inputPoints[numberOfInputPoints-1]-inputPoints[numberOfInputPoints-2])        
        return inOutMatrix

    def StoreInOutMatrix(self,stepSize):

        self.inOutMatrixX = self.MakeInOutMatrix(self.pointsX,\
                                                 self.pointsX,\
                                                 stepSize)
        self.inOutMatrixY = self.MakeInOutMatrix(self.pointsY,\
                                                 self.pointsY,\
                                                 stepSize)
        self.madeStoreInOutMatrix = True
        return None
    

#*****************************************************************************************************************
# FFT approach 
#*****************************************************************************************************************

    def FFTIntegratorExact(self,field,stepSize):
        fftIntegrator = np.exp(-(1.j/(2.0*self.backgroundIndex*self.alpha))*\
                                (self.meshWaveNumbersX**2.0 + self.meshWaveNumbersY**2.0)*\
                                (stepSize))
    
        returnField = np.fft.ifft2( np.fft.ifftshift( fftIntegrator * np.fft.fftshift(np.fft.fft2(field)) ) )
        return returnField


    def FFTIntegratorApprox(self,field,stepSize):
        fftIntegrator = 1.0 - (1.j/(2.0*self.backgroundIndex*self.alpha))*\
                                (self.meshWaveNumbersX**2.0 + self.meshWaveNumbersY**2.0)*\
                                (stepSize)
    
        returnField = np.fft.ifft2( np.fft.ifftshift( fftIntegrator * np.fft.fftshift(np.fft.fft2(field)) ) )
        return returnField


#*****************************************************************************************************************
# Split Step functions
#*****************************************************************************************************************

    def SplitStep1FullApprox(self,steps):
      
        indexSheet = self.IndexOfRefractionFunction()

        for indexZ in range(steps):
            temp = self.recordedFieldFFTIntgrator[:,:,indexZ]
            temp = self.FFTIntegratorApprox(temp, self.stepSizeZ)
            temp *= 1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * self.stepSizeZ * indexSheet )
            self.recordedFieldFFTIntgrator[:,:,indexZ+1] = temp  
            self.fieldEnergyFFTIntgrator[indexZ+1] = self.Energy(self.recordedFieldFFTIntgrator[:,:,indexZ+1])
            print("Finshed step ",indexZ+1," of ",self.numberOfPointsZ)
            print("FFT Energy: ",self.fieldEnergyFFTIntgrator[indexZ+1])
        return None

    def SplitStep1HalfApprox(self,steps):
      
        indexSheet = self.IndexOfRefractionFunction()

        for indexZ in range(steps):
            temp = self.recordedFieldFFTIntgrator[:,:,indexZ]
            temp = self.FFTIntegratorExact(temp, self.stepSizeZ)
            temp *= 1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * self.stepSizeZ * indexSheet )
            self.recordedFieldFFTIntgrator[:,:,indexZ+1] = temp  
            self.fieldEnergyFFTIntgrator[indexZ+1] = self.Energy(self.recordedFieldFFTIntgrator[:,:,indexZ+1])
            print("Finshed step ",indexZ+1," of ",self.numberOfPointsZ)
            print("FFT Energy: ",self.fieldEnergyFFTIntgrator[indexZ+1])
        return None

    def SplitStep1Exact(self,steps):
       
        indexSheet = self.IndexOfRefractionFunction()

        for indexZ in range(steps):
            temp = self.recordedFieldFFTIntgrator[:,:,indexZ]
            temp = self.FFTIntegratorExact(temp, self.stepSizeZ)
            temp *= np.exp(1.j*((self.gamma**2.0)/self.alpha)* (self.stepSizeZ) * indexSheet)
            self.recordedFieldFFTIntgrator[:,:,indexZ+1] = temp  
            self.fieldEnergyFFTIntgrator[indexZ+1] = self.Energy(self.recordedFieldFFTIntgrator[:,:,indexZ+1])
            print("Finshed step ",indexZ+1," of ",self.numberOfPointsZ)
            print("FFT Energy: ",self.fieldEnergyFFTIntgrator[indexZ+1])
        return None


    def SplitStep2FullApprox(self,steps):
  
        indexSheet = self.IndexOfRefractionFunction()

        for indexZ in range(steps):
         
            temp = self.recordedFieldFFTIntgrator[:,:,indexZ]
            temp *= 1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * (self.stepSizeZ/2.0) * indexSheet )
            temp = self.FFTIntegratorApprox(temp, self.stepSizeZ)
            temp *= 1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * (self.stepSizeZ/2.0) * indexSheet )

            self.recordedFieldFFTIntgrator[:,:,indexZ+1] = temp  
            self.fieldEnergyFFTIntgrator[indexZ+1] = self.Energy(self.recordedFieldFFTIntgrator[:,:,indexZ+1])

            print("Finshed step ",indexZ+1," of ",self.numberOfPointsZ)
            print("FFT Energy: ",self.fieldEnergyFFTIntgrator[indexZ+1])
        return None

    def SplitStep2HalfApprox(self,steps):
  
        indexSheet = self.IndexOfRefractionFunction()

        for indexZ in range(steps):
         
            temp = self.recordedFieldFFTIntgrator[:,:,indexZ]
            temp *= 1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * (self.stepSizeZ/2.0) * indexSheet )
            temp = self.FFTIntegratorExact(temp, self.stepSizeZ)
            temp *= 1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * (self.stepSizeZ/2.0) * indexSheet )

            self.recordedFieldFFTIntgrator[:,:,indexZ+1] = temp  
            self.fieldEnergyFFTIntgrator[indexZ+1] = self.Energy(self.recordedFieldFFTIntgrator[:,:,indexZ+1])

            print("Finshed step ",indexZ+1," of ",self.numberOfPointsZ)
            print("FFT Energy: ",self.fieldEnergyFFTIntgrator[indexZ+1])
        return None

    def SplitStep2FFTExact(self,steps):
      
        indexSheet = self.IndexOfRefractionFunction()

        for indexZ in range(steps):
            temp = np.copy(self.recordedFieldFFTIntgrator[:,:,indexZ])
            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) *(self.stepSizeZ/2.0) * indexSheet)
            temp = self.FFTIntegratorExact(temp, self.stepSizeZ)
            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) * (self.stepSizeZ/2.0) * indexSheet)

            self.recordedFieldFFTIntgrator[:,:,indexZ+1] = np.copy(temp)  
            self.fieldEnergyFFTIntgrator[indexZ+1] = self.Energy(self.recordedFieldFFTIntgrator[:,:,indexZ+1])
            print("Finshed step ",indexZ+1," of ",self.numberOfPointsZ)
            print("FFT Energy: ",self.fieldEnergyFFTIntgrator[indexZ+1])
        return None
    
    
    def SplitStep2DirectExact(self,steps):
      
        indexSheet = self.IndexOfRefractionFunction()
        self.StoreInOutMatrix(self.stepSizeZ)

        for indexZ in range(steps):
            
            ## Half step of Deterministic index
            temp = np.copy(self.recordedFieldDirectIntgrator[:,:,indexZ])
            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) *(self.stepSizeZ/2.0) * indexSheet)
            temp = self.DirectIntegrator(temp, self.stepSizeZ)
            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) * (self.stepSizeZ/2.0) * indexSheet)

            self.recordedFieldDirectIntgrator[:,:,indexZ+1] = np.copy(temp)  
            self.fieldEnergyDirectIntgrator[indexZ+1] = self.Energy(self.recordedFieldDirectIntgrator[:,:,indexZ+1])
            print("Finshed step ",indexZ+1," of ",self.numberOfPointsZ)
            print("FFT Energy: ",self.fieldEnergyDirectIntgrator[indexZ+1])
        return None

    def SplitStep2DirectStellarExact(self,steps):
      
        indexSheet = self.IndexOfRefractionFunction()
        self.StoreInOutMatrix(self.stepSizeZ)

        for indexZ in range(steps):
            
            ## Half step of Deterministic index
            temp = np.copy(self.recordedFieldDirectIntgrator[:,:,indexZ])
            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) *(self.stepSizeZ/2.0) * indexSheet)
            temp = self.DirectIntegratorStellar(temp, self.stepSizeZ)
            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) * (self.stepSizeZ/2.0) * indexSheet)

            self.recordedFieldDirectIntgrator[:,:,indexZ+1] = np.copy(temp)  
            self.fieldEnergyDirectIntgrator[indexZ+1] = self.Energy(self.recordedFieldDirectIntgrator[:,:,indexZ+1])
            print("Finshed step ",indexZ+1," of ",self.numberOfPointsZ)
            print("FFT Energy: ",self.fieldEnergyDirectIntgrator[indexZ+1])
        return None

    def SplitStep4Approx(self,numberOfSteps):
        """

        """        
        omega = (2.0 + 2.0**(1/3) + 2.0**(-1/3))/3.0
        indexSheet = self.IndexOfRefractionFunction()

        for indexZ in range(numberOfSteps):
            
            #indexZ2 = 2*indexZ
            
            # Half step of noise
            #self.recordedFieldFFTIntgrator[:,:,indexZ] *= np.exp(1.j * \
            #                                                     ((self.gamma**2.0)/self.alpha)*np.sqrt(self.stepSizeZ/2.0) *\
            #                                                     self.indexRealizations[:,:,indexZ2])

            
            ## Half step of Deterministic index
            temp = self.recordedFieldFFTIntgrator[:,:,indexZ]
           

            temp *=  1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * (omega)*(self.stepSizeZ/2.0) * indexSheet )
            print(self.Energy(temp))

            temp = self.FFTIntegratorExact(temp,omega*self.stepSizeZ)
            print(self.Energy(temp))

            temp *= 1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * (omega)*(self.stepSizeZ/2.0) * indexSheet)
            print(self.Energy(temp))



            temp *=  1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * ((1.0-2.0*omega)*(self.stepSizeZ/2.0)) * indexSheet)
            print(self.Energy(temp))

            temp = self.FFTIntegratorExact(temp,(1.0-2.0*omega)*self.stepSizeZ)
            print(self.Energy(temp))

            temp *=  1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * ((1.0-2.0*omega)*(self.stepSizeZ/2.0)) * indexSheet)
            print(self.Energy(temp))


            temp *=  1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * ((omega)*(self.stepSizeZ/2.0)) * indexSheet)
            print(self.Energy(temp))

            temp = self.FFTIntegratorExact(temp,omega*self.stepSizeZ)
            print(self.Energy(temp))

            temp *= 1.0 + 1.j * ( ((self.gamma**2.0)/self.alpha) * ((omega)*(self.stepSizeZ/2.0)) * indexSheet)
            print(self.Energy(temp))



            self.recordedFieldFFTIntgrator[:,:,indexZ+1] = temp

            self.fieldEnergyFFTIntgrator[indexZ+1] = self.Energy(self.recordedFieldFFTIntgrator[:,:,indexZ+1])

            print("Finshed step ",indexZ+1," of ",self.numberOfPointsZ)
            print("FFT Energy: ",self.fieldEnergyFFTIntgrator[indexZ+1])


        return None


    def SplitStep4Exact(self,numberOfSteps):
        """

        """        
        omega = (2.0 + 2.0**(1/3) + 2.0**(-1/3))/3.0
        indexSheet = self.IndexOfRefractionFunction()

        for indexZ in range(numberOfSteps):
            
            #indexZ2 = 2*indexZ
            
            # Half step of noise
            #self.recordedFieldFFTIntgrator[:,:,indexZ] *= np.exp(1.j * \
            #                                                     ((self.gamma**2.0)/self.alpha)*np.sqrt(self.stepSizeZ/2.0) *\
            #                                                     self.indexRealizations[:,:,indexZ2])

            
            ## Half step of Deterministic index
            temp = np.copy(self.recordedFieldFFTIntgrator[:,:,indexZ])
            
            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) *\
                                 (omega)*(self.stepSizeZ/2.0) *\
                                 indexSheet
                          )
            temp = self.FFTIntegratorExact(temp,omega*self.stepSizeZ)
            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) *\
                                 (omega)*(self.stepSizeZ/2.0) *\
                                 indexSheet
                          )

            print(self.Energy(temp))

            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) *\
                                 (1.0-2.0*omega)*(self.stepSizeZ/2.0) *\
                                 indexSheet
                          )
            temp = self.FFTIntegratorExact(temp,(1.0-2.0*omega)*self.stepSizeZ)
            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) *\
                                  (1.0-2.0*omega)*(self.stepSizeZ/2.0) *\
                                  indexSheet
                          )

            print(self.Energy(temp))

            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) *\
                                 (omega)*(self.stepSizeZ/2.0) *\
                                 indexSheet
                          )
            temp = self.FFTIntegratorExact(temp,omega*self.stepSizeZ)
            temp *= np.exp(1.j * ((self.gamma**2.0)/self.alpha) *\
                                  (omega)*(self.stepSizeZ/2.0) *\
                                  indexSheet
                          )



            self.recordedFieldFFTIntgrator[:,:,indexZ+1] = np.copy(temp)

            self.fieldEnergyFFTIntgrator[indexZ+1] = self.Energy(self.recordedFieldFFTIntgrator[:,:,indexZ+1])

            print("Finshed step ",indexZ+1," of ",self.numberOfPointsZ)
            print("FFT Energy: ",self.fieldEnergyFFTIntgrator[indexZ+1])


        return None


  
