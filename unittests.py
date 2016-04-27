from rate_model import *
import imagen
import numpy
import numbergen


def test_convolutional_projection_correction_factors():
    s1 = InputSheet('S1',20,0)
    s2 = Sheet('S2',20,0.0005)
    s3 = Sheet('S3',20,0.0005)

    kernel = imagen.Gaussian(xdensity=7,ydensity=7,aspect_ratio=1.0,size=0.3)() - 0.7*imagen.Gaussian(xdensity=7,ydensity=7,aspect_ratio=1.0,size=0.6)()
    kernel = kernel-kernel.mean()
    
    p1 = ConvolutionalProjection("S1toS2",s1,s2,1.0,0.0002,kernel)
    p2 = ConnectionFieldProjection("S1toS3",s1,s3,1.0,0.0002,kernel)

    #Model initialization and execution
    model = Model([s1,s2,s3],0.0001)
    
    g1 = imagen.Gaussian(xdensity=72,ydensity=72,x=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=42),
                         y=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=43),
                      orientation=numbergen.UniformRandom(lbound=-numpy.pi,ubound=numpy.pi,seed=42),
                     size=0.048388, aspect_ratio=4.66667, scale=1.0)
    
    g2 = imagen.Gaussian(xdensity=72,ydensity=72,x=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=12),
                         y=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=13),
                      orientation=numbergen.UniformRandom(lbound=-numpy.pi,ubound=numpy.pi,seed=22),
                      size=0.048388, aspect_ratio=4.66667, scale=1.0)
            
    
    for i in xrange(10):
        s1.set_activity(numpy.maximum(g1(),g2()))
        model.run(0.005)
        assert numpy.sum(s2.get_activity(0)-s3.get_activity(0)) <= 0.0000001
    
