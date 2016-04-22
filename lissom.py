from rate_model import *

# Sheets
retina = InputSheet('Retina',25,0)
lgn = Sheet('LGN',25,10)

#Projections
retina_to_lgn = ConvolutionalProjection(lgn_kernel,retina,lgn,1.0)

#Model
lissom = Model([retina,lgn],0.1)


for i in xrange(100):
    retina.set_activity()
    lissom.run(1000)