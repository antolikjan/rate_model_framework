from rate_model import *
import imagen

# Sheets
retina = InputSheet('Retina',25,0)
lgn = Sheet('LGN',25,10)

#Projections

lgn_kernel = imagen.Gaussian(xdensity=10,ydensity=10,aspect_ratio=0,size=2.0)() - imagen.Gaussian(xdensity=10,ydensity=10,aspect_ratio=0,size=5.0)()

retina_to_lgn = ConvolutionalProjection("RatinaToLgn",retina,lgn,1.0,0.1,lgn_kernel)

#Model
lissom = Model([retina,lgn],0.1)


for i in xrange(100):
    retina.set_activity()
    lissom.run(1000)
