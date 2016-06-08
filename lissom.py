import matplotlib
#matplotlib.use('Agg')
from rate_model import *
from projections import * 
import imagen
import imagen.random
from imagen.transferfn import DivisiveNormalizeL1
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
retina = InputSheet('Retina',2.4,25,None)
lgn_on = NoTimeconstantSheet('LGN_ON',1.6,25,None)
lgn_off = NoTimeconstantSheet('LGN_OFF',1.6,25,None)
V1 = Sheet('V1',1.0,50,0.002,threshold=float(sys.argv[4]))

print sys.argv


#Projections
# DoG weights for the LGN
centerg   = imagen.Gaussian(size=0.07385,aspect_ratio=1.0,output_fns=[DivisiveNormalizeL1()])
surroundg = imagen.Gaussian(size=0.29540,aspect_ratio=1.0,output_fns=[DivisiveNormalizeL1()])
on_weights = imagen.Composite(generators=[centerg,surroundg],operator=numpy.subtract)
off_weights = imagen.Composite(generators=[surroundg,centerg],operator=numpy.subtract)


retina_to_lgn_on = FastConnetcionFieldProjection("RetinaToLgnOn",retina,lgn_on,1.0,0.001,0.375,on_weights)
retina_to_lgn_off = FastConnetcionFieldProjection("RetinaToLgnOff",retina,lgn_off,1.0,0.001,0.375,off_weights)

lgn_on_to_V1 = FastConnetcionFieldProjection("LGNOnToV1",lgn_on,V1,0.5,0.001,0.27083,imagen.random.GaussianCloud(gaussian_size=2*0.27083))
lgn_off_to_V1 = FastConnetcionFieldProjection("LGNOffToV1",lgn_off,V1,0.5,0.001,0.27083,imagen.random.GaussianCloud(gaussian_size=2*0.27083))

center_lat  = imagen.Gaussian(size=0.05,aspect_ratio=1.0,output_fns=[DivisiveNormalizeL1()],xdensity=V1.density+1,ydensity=V1.density+1,bounds = BoundingBox(radius=V1.size/2.0))()[14:-14,14:-14]
surround_lat = imagen.Gaussian(size=0.15,aspect_ratio=1.0,output_fns=[DivisiveNormalizeL1()],xdensity=V1.density+1,ydensity=V1.density+1,bounds = BoundingBox(radius=V1.size/2.0))()[14:-14,14:-14]
center_lat = center_lat / numpy.sum(center_lat)
surround_lat = surround_lat / numpy.sum(surround_lat)

V1_lat_exc = ConvolutionalProjection("LateralExc",V1,V1,0.5*float(sys.argv[1]),0.001,center_lat)
V1_lat_inh = ConvolutionalProjection("LateralInh",V1,V1,-0.5*float(sys.argv[1])*float(sys.argv[2]),0.001,surround_lat)
#V1_lat_exc = FastConnetcionFieldProjection("LateralExc",V1,V1,0.5*float(sys.argv[1]),0.001,0.104,imagen.Gaussian(aspect_ratio=1.0, size=0.05))
#V1_lat_inh = FastConnetcionFieldProjection("LateralInh",V1,V1,-0.5*float(sys.argv[1])*float(sys.argv[2]),0.001,0.22917,imagen.Gaussian(aspect_ratio=1.0, size=0.15))

#Model initialization and execution
lissom = Model([retina,lgn_on,lgn_off,V1],0.001)

g1 = imagen.Gaussian(xdensity=retina.unit_diameter,ydensity=retina.unit_diameter,x=numbergen.UniformRandom(lbound=-0.4,ubound=0.4,seed=342),
                     y=numbergen.UniformRandom(lbound=-0.4,ubound=0.4,seed=343),
                     orientation=numbergen.UniformRandom(lbound=-numpy.pi,ubound=numpy.pi,seed=333),
                     size=0.7*0.048388, aspect_ratio=1.2*4.66667, scale=1.0)

g2 = imagen.Gaussian(xdensity=retina.unit_diameter,ydensity=retina.unit_diameter,x=numbergen.UniformRandom(lbound=-0.4,ubound=0.4,seed=312),
		     y=numbergen.UniformRandom(lbound=-0.4,ubound=0.4,seed=313),
                     orientation=numbergen.UniformRandom(lbound=-numpy.pi,ubound=numpy.pi,seed=322),
                     size=0.7*0.048388, aspect_ratio=1.2*4.66667, scale=1.0)
        
        
#pylab.figure();plot_projection(lgn_on_to_V1,filename="onProjection.png");pylab.show()
        
        
run_for = 10000
t = time.time()
for i in xrange(run_for):
    retina.set_activity(numpy.maximum(g1(),g2()))
    lissom.run(0.15)
    #pylab.figure();display_model_state(lissom,filename="activity.png");pylab.show()
    #pylab.figure();display_model_state(lissom);pylab.show()
    #lissom.run(0.15)
    #pylab.figure();display_model_state(lissom);
    #pylab.figure();plot_projection(lgn_on_to_V1,downsample=0.5)
    #pylab.figure();plot_projection(retina_to_lgn_off,filename="RatinaToLgnOff.png")
    
    lgn_on_to_V1.applyHebianLearningStep(float(sys.argv[3]))
    lgn_off_to_V1.applyHebianLearningStep(float(sys.argv[3]))
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

pylab.show()

if False:
    pylab.figure()
    plot_projection(lgn_on_to_V1)
    pylab.figure()
    plot_projection(lgn_off_to_V1)
    pylab.show()



    
    
