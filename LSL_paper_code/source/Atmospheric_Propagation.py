import numpy as np

class Atmospheric_Propagation(object):
    """
    Base class defintion for atmospheric propagation 

    """

    def __init__(self,     myRank = 1,
                   outputFilename = 'output.txt',
                       waveLength =   1e-6,
                       innerScale =   1e-3,
                       outerScale =    1e3,
           indexStructureConstant =   1e-8,
             indexVarianceScaling =      1,
                  backgroundIndex = 1+1e-6, 
                apertureDiameter  =      1,
                          lengthX =      3, 
                  numberOfPointsX =   1024,
                          lengthY =      3, 
                  numberOfPointsY =   1024,
              propagationDistance =    1e4,
                  numberOfPointsZ =   1024,
   transverseCharacteristicLength =      1,
  propagationCharacteristicLength =    1e3,
           initialFieldIdentifier = 'debug'):


        """
            set all dimensional parameters and calculate the nondimensional parameters
        """

        self.EYE = np.complex(0.0,1.0)
        outputs_file =  open(outputFilename, 'w+')

        # Dimensional Parameter Values
        self.waveLength             = waveLength
        self.waveNumber             = (2.0*np.pi)/self.waveLength
        self.innerScale             = innerScale
        self.outerScale             = outerScale
        self.indexStructureConstant = indexStructureConstant
        self.indexVarianceScaling   = indexVarianceScaling # (C_0) [1-10]
        self.backgroundIndex        = backgroundIndex      # n_0
        self.apertureDiameter       = apertureDiameter
                
        self.lengthX                = lengthX 
        self.numberOfPointsX        = numberOfPointsX
        self.lengthY                = lengthY 
        self.numberOfPointsY        = numberOfPointsY
        self.propagationDistance    = propagationDistance
        self.numberOfPointsZ        = numberOfPointsZ

        self.transverseCharacteristicLength  = transverseCharacteristicLength
        self.propagationCharacteristicLength = propagationCharacteristicLength

        self.initialFieldIdentifier = initialFieldIdentifier
        

        # Dimensionless Parameter Values
        #self.indexStandardDeviation = (self.indexStructureConstant/np.sqrt(2.0))*\
        #                              ((self.indexVarianceScaling * self.outerScale)**(1.0/3.0))  # (sigma_n)
        #self.epsilon                = self.innerScale/self.outerScale 
        #self.alpha                  = ((self.innerScale**2.0) * self.waveNumber)/self.outerScale
        #self.beta                   = (self.innerScale * self.waveNumber) 
        #self.gamma                  = self.innerScale * self.waveNumber * np.sqrt(self.indexStandardDeviation)
        #self.nu                     = self.innerScale * self.waveNumber * self.indexStandardDeviation
        #self.computationalPropagationDistance  = self.propagationDistance/self.outerScale
        #self.computationalApertureDiameter     = self.apertureDiameter/self.innerScale

        self.indexStandardDeviation = (self.indexStructureConstant/np.sqrt(2.0))*\
                                      ((self.indexVarianceScaling * self.outerScale)**(1.0/3.0))  # (sigma_n)

        self.epsilon                = self.transverseCharacteristicLength/self.propagationCharacteristicLength 
        self.alpha                  = ((self.transverseCharacteristicLength**2.0) * \
                                        self.waveNumber)/self.propagationCharacteristicLength

        self.beta                   = (self.transverseCharacteristicLength * self.waveNumber) 
        self.gamma                  = self.transverseCharacteristicLength * self.waveNumber * \
                                      np.sqrt(self.indexStandardDeviation)
        self.nu                     = self.transverseCharacteristicLength * self.waveNumber * self.indexStandardDeviation

        self.computationalPropagationDistance  = self.propagationDistance/self.propagationCharacteristicLength
        self.computationalApertureDiameter     = self.apertureDiameter/self.transverseCharacteristicLength


        self.stepSizeZ           = self.computationalPropagationDistance/self.numberOfPointsZ
        self.pointsZ             = self.stepSizeZ*np.arange(0,self.numberOfPointsZ+1)

        self.computationalWidthY = self.lengthY/self.transverseCharacteristicLength
        self.computationalWidthX = self.lengthX/self.transverseCharacteristicLength
        
        self.stepSizeY           = self.computationalWidthY/self.numberOfPointsY
        self.pointsY             = -self.computationalWidthY/2.0 + self.stepSizeY*np.arange(0,self.numberOfPointsY)
        self.waveNumberStepSizeY = (2.0*np.pi)/self.computationalWidthY
        self.waveNumbersY        = self.waveNumberStepSizeY * np.arange(-self.numberOfPointsY/2.0,self.numberOfPointsY/2.0)
          
        self.stepSizeX           = self.computationalWidthX/self.numberOfPointsX
        self.pointsX             = -self.computationalWidthX/2.0 + self.stepSizeX*np.arange(0,self.numberOfPointsX)
        self.waveNumberStepSizeX = (2.0*np.pi)/self.computationalWidthX
        self.waveNumbersX        = self.waveNumberStepSizeX * np.arange(-self.numberOfPointsX/2.0,self.numberOfPointsX/2.0)




#        print("innerScale:",self.innerScale)
#        print("outerScale:",self.outerScale)
#        print("indexStructureConstant:",self.indexStructureConstant)
#        print("indexVarianceScaling:",self.indexVarianceScaling)
#        print("backgroundIndex:",self.backgroundIndex)
#        print("apertureDiameter:",self.apertureDiameter)
#
#
#        print("lengthX:",self.lengthX)
#        print("numberOfPointsX:",self.numberOfPointsX)
#        print("lengthY:",self.lengthY)
#        print("numberOfPointsY:",self.numberOfPointsY)
#        print("propagationDistance:",self.propagationDistance)
#        print("numberOfPointsZ:",self.numberOfPointsZ)
#        print("initialFieldIdentifier:",self.initialFieldIdentifier) 
#        print("indexStandardDeviation:",self.indexStandardDeviation)
#        print("epsilon:",self.epsilon)
#        print("alpha:",self.alpha)
#        print("beta:",self.beta)
#        print("gamma:",self.gamma)
#        print("nu:",self.nu)

#        print("computationalApertureDiameter:",self.computationalApertureDiameter)
#        print("computationalPropagationDistance:",self.computationalPropagationDistance)
#
#        print("computationalLengthInZ:",self.computationalPropagationDistance)
#        print("stepSizeZ:",self.stepSizeZ)
#
#        print("computationalWidthX:",self.computationalWidthX)
#        print("stepSizeX:",self.stepSizeX)
#
#        print("computationalWidthY:",self.computationalWidthY)
#        print("stepSizeY:",self.stepSizeY)



        print("innerScale:",self.innerScale, file=outputs_file)
        print("outerScale:",self.outerScale, file=outputs_file)
        print("indexStructureConstant:",self.indexStructureConstant, file=outputs_file)
        print("indexVarianceScaling:",self.indexVarianceScaling, file=outputs_file)
        print("backgroundIndex:",self.backgroundIndex, file=outputs_file)
        print("apertureDiameter:",self.apertureDiameter, file=outputs_file)


        print("lengthX:",self.lengthX, file=outputs_file)
        print("numberOfPointsX:",self.numberOfPointsX, file=outputs_file)
        print("lengthY:",self.lengthY, file=outputs_file)
        print("numberOfPointsY:",self.numberOfPointsY, file=outputs_file)
        print("propagationDistance:",self.propagationDistance, file=outputs_file)
        print("numberOfPointsZ:",self.numberOfPointsZ, file=outputs_file)
        print("initialFieldIdentifier:",self.initialFieldIdentifier, file=outputs_file)  

        print("indexStandardDeviation:",self.indexStandardDeviation, file=outputs_file)
        print("epsilon:",self.epsilon, file=outputs_file)
        print("alpha:",self.alpha, file=outputs_file)
        print("beta:",self.beta, file=outputs_file)
        print("gamma:",self.gamma, file=outputs_file)
        print("nu:",self.nu, file=outputs_file)
        print("computationalApertureDiameter:",self.computationalApertureDiameter, file=outputs_file)
        print("computationalPropagationDistance:",self.computationalPropagationDistance, file=outputs_file)
        print("computationalLengthInZ:",self.computationalPropagationDistance, file=outputs_file)
        print("stepSizeZ:",self.stepSizeZ, file=outputs_file)
        print("computationalWidthX:",self.computationalWidthX, file=outputs_file)
        print("stepSizeX:",self.stepSizeX, file=outputs_file)
        print("computationalWidthY:",self.computationalWidthY, file=outputs_file)
        print("stepSizeY:",self.stepSizeY, file=outputs_file)


        return None
    
