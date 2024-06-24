

import numpy as np
import scipy as sp
from scipy.special import j1
import time
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
#from matplotlib import cm
#from matplotlib.colors import LogNorm
#from mpl_toolkits.mplot3d import axes3d
from mpi4py import MPI

import sys
sys.path.insert(0,"./source/")


#from ../source/Atmospheric_Propagation import Atmospheric_Propagation
from Paraxial_Equation import Paraxial_Equation




## Define Dimensional Parameter Values
# codeName             # Value                   # (symbol) [char. value] <units>
#----------------------------------------------------------------------------------------------
waveLength             = 1e-6                    # (lambda) [10^{-6}]          <meters>
waveNumber             = (2.0*np.pi)/waveLength  # (lambda) [10^{6}]           <meters^{-1}>
innerScale             = 1e-2                    # (l_0)    [10^{-3}]          <meters>
outerScale             = 1e2                     # (L_0)    [10^{3}]           <meters>
indexStructureConstant = np.sqrt(5.0) * 1e-7     # (C_n)  [10^{-10},10^{-6}] <meters^{-1/3}>
indexVarianceScaling   = 1                       # C_0
apertureDiameter       = 2e-1                    # (D_{i})  [1 - .1]           <meters>
propagationDistance    = 1e3                     # (L)      [10^{3} - 10^{4}]  <meters> 
lengthX                = 2.0                     # XLength x_max               <meters>
lengthY                = 2.0                     # YLength y_max               <meters>
backgroundIndex        = 1.0 + 1e-6              # n_0      [1+e-6]

transverseCharacteristicLength  = apertureDiameter
propagationCharacteristicLength = propagationDistance/100

## Define Simulation Parameter Values
#----------------------------------------------------------------------------------------------
numberOfPointsX                   = 512
numberOfPointsY                   = 512
numberOfPointsZ                   = 1024

initialFieldIdentifier            = 'lagra'


myRank = MPI.COMM_WORLD.Get_rank()
numberOfProcesses = MPI.COMM_WORLD.Get_size()

outputFilename = "output_"+str(myRank)+".txt"
print(outputFilename)



mySimulation = Paraxial_Equation(myRank,
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



#
#
mySimulation.SplitStep2FFTExact(numberOfPointsZ)


#from Variational_Equations import Variational_Equations
#lagSim = Variational_Equations(waveLength,
#                      innerScale,
#                      outerScale,
#                      indexStructureConstant,
#                      indexVarianceScaling,
#                      backgroundIndex, 
#                      apertureDiameter,
#                      lengthX, 
#                      numberOfPointsX,
#                      lengthY, 
#                      numberOfPointsY,
#                      propagationDistance,
#                      numberOfPointsZ,
#                      initialFieldIdentifier)





