import matplotlib
#matplotlib.use('Agg')
from rate_model import *
from projections import * 
import imagen
import imagen.random
import numpy
import pylab
import numbergen
from analysis import *
from visualization import *
from analysis import *
import time
import sys

# parameters
#aff_ratio, inh_ratio, lr, threshold


# Sheets
retina = InputSheet('Retina',45,0)
lgn_on = Sheet('LGN_ON',35,0.001)
lgn_off = Sheet('LGN_OFF',35,0.001)
V1 = Sheet('V1',25,0.002,threshold=float(sys.argv[4]))

print sys.argv


#Projections
on = imagen.Gaussian(xdensity=17,ydensity=17,aspect_ratio=1.0,size=0.5/numpy.sqrt(2))()
off =  imagen.Gaussian(xdensity=17,ydensity=17,aspect_ratio=1.0,size=2.0/numpy.sqrt(2))()
lgn_mask = imagen.Disk(xdensity=17,ydensity=17,size=1.0,smoothing=0)()
on = numpy.multiply(on,lgn_mask)
off = numpy.multiply(off,lgn_mask)
on = on/on.sum()
off = off/off.sum()
lgn_kernel_on = on - off
lgn_kernel_on = lgn_kernel_on/ lgn_kernel_on.sum() 
lgn_kernel_off = - lgn_kernel_on

retina_to_lgn_on = ConvolutionalProjection("RatinaToLgnOn",retina,lgn_on,1.0,0.001,lgn_kernel_on)
retina_to_lgn_off = ConvolutionalProjection("RatinaToLgnOff",retina,lgn_off,1.0,0.001,lgn_kernel_off)

lgn_to_V1_kernel = imagen.random.GaussianCloud(xdensity=21,ydensity=21,aspect_ratio=1.0,size=1.0)
lgn_to_V1_mask = imagen.Disk(xdensity=21,ydensity=21,size=1.0,smoothing=0)()
lgn_on_to_V1 = FastConnetcionFieldProjection("LGNOnToV1",lgn_on,V1,0.5,0.001,lgn_to_V1_kernel,mask=lgn_to_V1_mask)
lgn_off_to_V1 = FastConnetcionFieldProjection("LGNOffToV1",lgn_off,V1,0.5,0.001,lgn_to_V1_kernel,mask=lgn_to_V1_mask)

V1_lat_exc_kernel = imagen.Gaussian(xdensity=17,ydensity=17,aspect_ratio=1.0,size=0.5/numpy.sqrt(2))()
V1_lat_inh_kernel = imagen.Gaussian(xdensity=17,ydensity=17,aspect_ratio=1.0,size=0.5)()
lat_mask = imagen.Disk(xdensity=17,ydensity=17,size=1.0,smoothing=0)()
V1_lat_exc_kernel = numpy.multiply(V1_lat_exc_kernel,lat_mask)
V1_lat_inh_kernel = numpy.multiply(V1_lat_inh_kernel,lat_mask)
V1_lat_exc_kernel = V1_lat_exc_kernel/V1_lat_exc_kernel.sum()
V1_lat_inh_kernel = V1_lat_inh_kernel/V1_lat_inh_kernel.sum()
V1_lat_exc = ConvolutionalProjection("LateralExc",V1,V1,0.5*float(sys.argv[1]),0.001,V1_lat_exc_kernel)
V1_lat_inh = ConvolutionalProjection("LateralInh",V1,V1,0.5*float(sys.argv[1])* float(sys.argv[2]),0.001,V1_lat_inh_kernel)

#1.0*-3.45

#Model initialization and execution
lissom = Model([retina,lgn_on,lgn_off,V1],0.001)

g1 = imagen.Gaussian(xdensity=retina.radius*2,ydensity=retina.radius*2,x=numbergen.UniformRandom(lbound=-0.4,ubound=0.4,seed=342),
                     y=numbergen.UniformRandom(lbound=-0.4,ubound=0.4,seed=343),
                     orientation=numbergen.UniformRandom(lbound=-numpy.pi,ubound=numpy.pi,seed=333),
                     size=0.7*0.048388, aspect_ratio=1.2*4.66667, scale=1.0)

g2 = imagen.Gaussian(xdensity=retina.radius*2,ydensity=retina.radius*2,x=numbergen.UniformRandom(lbound=-0.4,ubound=0.4,seed=312),
		     y=numbergen.UniformRandom(lbound=-0.4,ubound=0.4,seed=313),
                     orientation=numbergen.UniformRandom(lbound=-numpy.pi,ubound=numpy.pi,seed=322),
                     size=0.7*0.048388, aspect_ratio=1.2*4.66667, scale=1.0)
        
        
run_for =10000
t = time.time()
for i in xrange(run_for):
    retina.set_activity(numpy.maximum(g1(),g2()))
    lissom.run(0.15)
    applyHebianLearningStepOnAFastConnetcionFieldProjection(lgn_on_to_V1,float(sys.argv[3]))
    applyHebianLearningStepOnAFastConnetcionFieldProjection(lgn_off_to_V1,float(sys.argv[3]))
    print i , ":", "Expected time to run: " , ((time.time()-t)/(i+1)) * (run_for-i) , "s"
    if i == run_for-1:
       pylab.figure();display_model_state(lissom,filename="activity.png")
    lissom.reset()
    
   
if True:    
    import pickle
    f = open('lissom_long.pickle','wb')
    pickle.dump(lissom,f)
    f.close()

    

pylab.figure()
plot_projection(lgn_on_to_V1,filename="onProjection.png")

pylab.figure()
plot_projection(lgn_off_to_V1,filename="offProjection.png")

pylab.figure();fullfieldSineGratingOrientationTuningProtocol(lissom,retina,sheets=[V1],num_orientation=8,num_phase=10,duration=0.02,frequency=5.0,filename="freq=5",reset=True,plot=True,load=False)
pylab.figure();fullfieldSineGratingOrientationTuningProtocol(lissom,retina,sheets=[V1],num_orientation=8,num_phase=10,duration=0.02,frequency=10.0,filename="freq=10",reset=True,plot=True,load=False)


if False:
    pylab.figure();display_model_state(lissom)
    
    
    pylab.figure()
    plot_projection(lgn_on_to_V1)
    pylab.figure()
    plot_projection(lgn_off_to_V1)
    
    pylab.show()



    
    
