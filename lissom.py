from rate_model import *
from projections import * 
import imagen
import numpy
import pylab
import numbergen
from visualization import *
from analysis import *

# Sheets
retina = InputSheet('Retina',36,0)
lgn_on = Sheet('LGN_ON',33,0.0001)
lgn_off = Sheet('LGN_OFF',33,0.0001)
V1 = Sheet('V1',25,0.0005)




#Projections
lgn_kernel = imagen.Gaussian(xdensity=7,ydensity=7,aspect_ratio=1.0,size=0.15)() - 0.7*imagen.Gaussian(xdensity=7,ydensity=7,aspect_ratio=1.0,size=0.4)()
lgn_kernel_on = lgn_kernel-lgn_kernel.mean()
lgn_kernel_off = - lgn_kernel_on

retina_to_lgn_on = ConvolutionalProjection("RatinaToLgnOn",retina,lgn_on,1.0,0.0002,lgn_kernel_on)
retina_to_lgn_off = ConvolutionalProjection("RatinaToLgnOff",retina,lgn_off,1.0,0.0002,lgn_kernel_off)

lgn_to_V1_kernel = imagen.Gaussian(xdensity=15,ydensity=15,aspect_ratio=1.0,size=0.25*2.5)()
lgn_on_to_V1 = FastConnetcionFieldProjection("LGNOnToV1",lgn_on,V1,0.5,0.0002,lgn_to_V1_kernel)
lgn_off_to_V1 = FastConnetcionFieldProjection("LGNOffToV1",lgn_off,V1,0.5,0.0002,lgn_to_V1_kernel)

V1_lat_exc_kernel = imagen.Gaussian(xdensity=11,ydensity=11,aspect_ratio=1.0,size=0.4)()
V1_lat_inh_kernel = imagen.Gaussian(xdensity=11,ydensity=11,aspect_ratio=1.0,size=0.4*numpy.sqrt(2))()
V1_lat_exc_kernel = V1_lat_exc_kernel/V1_lat_exc_kernel.sum()
V1_lat_inh_kernel = V1_lat_inh_kernel/V1_lat_inh_kernel.sum()
V1_lat_exc = ConvolutionalProjection("LateralExc",V1,V1,2*3.91,0.0002,V1_lat_exc_kernel)
V1_lat_inh = ConvolutionalProjection("LateralInh",V1,V1,2.5*-3.45,0.0002,V1_lat_inh_kernel)

#Model initialization and execution
lissom = Model([retina,lgn_on,lgn_off,V1],0.0001)

g1 = imagen.Gaussian(xdensity=72,ydensity=72,x=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=42),
                     y=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=43),
                     orientation=numbergen.UniformRandom(lbound=-numpy.pi,ubound=numpy.pi,seed=42),
                     size=0.048388, aspect_ratio=4.66667, scale=1.0)

g2 = imagen.Gaussian(xdensity=72,ydensity=72,x=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=12),
		     y=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=13),
                     orientation=numbergen.UniformRandom(lbound=-numpy.pi,ubound=numpy.pi,seed=22),
                     size=0.048388, aspect_ratio=4.66667, scale=1.0)
        

for i in xrange(10000):
    retina.set_activity(numpy.maximum(g1(),g2()))
    lissom.run(0.03)
    print i
    applyHebianLearningStepOnAFastConnetcionFieldProjection(lgn_on_to_V1,0.01)
    applyHebianLearningStepOnAFastConnetcionFieldProjection(lgn_off_to_V1,0.01)
    
    pylab.figure()
    display_model_state(lissom,filename="model_response.png")

import pickle
f = open('lissom_short.pickle','wb')
pickle.dump(lissom,f)
f.close()

pylab.figure()
plot_projection(lgn_on_to_V1,filename="onProjection.png")

pylab.figure()
plot_projection(lgn_off_to_V1,filename="offProjection.png")

pylab.figure()
fullfieldSineGratingOrientationTuningProtocol(lissom,retina,sheets=[V1],num_orientation=8,num_phase=10,duration=0.03,filename="responses.pickle",plot=True,load=False)





    
    
