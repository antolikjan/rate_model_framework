from rate_model import *
import imagen
import numpy
import pylab
from visualization import *

# Sheets
retina = InputSheet('Retina',25,0)
lgn_on = Sheet('LGN_ON',25,10)
lgn_off = Sheet('LGN_OFF',25,10)

#Projections

lgn_kernel = imagen.Gaussian(xdensity=6,ydensity=6,aspect_ratio=1.0,size=0.15)() - 0.7*imagen.Gaussian(xdensity=6,ydensity=6,aspect_ratio=1.0,size=0.4)()
lgn_kernel_on = lgn_kernel-lgn_kernel.mean()
lgn_kernel_off = - lgn_kernel_on

retina_to_lgn_on = ConvolutionalProjection("RatinaToLgn",retina,lgn_on,1.0,5.0,lgn_kernel_on)
retina_to_lgn_off = ConvolutionalProjection("RatinaToLgn",retina,lgn_off,1.0,5.0,lgn_kernel_off)

#Model
lissom = Model([retina,lgn_on,lgn_off],0.1)

stim = imagen.Gaussian(xdensity=50,ydensity=50,aspect_ratio=0.15,size=0.15)()

for i in xrange(1):
    retina.set_activity(stim,)
    lissom.run(1)
    


display_model_state(lissom)
pylab.show()



    
    