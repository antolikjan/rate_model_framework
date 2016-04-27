from rate_model import *
import imagen
import numpy
import numbergen
import unittest


class TestQConvolutionalProjection(unittest.TestCase):
    
    def test_correction_factors(self):
        s1 = InputSheet('S1',20,0)
        s2 = Sheet('S2',10,0.0005)
        s3 = Sheet('S3',10,0.0005)
        s4 = Sheet('S4',20,0.0005)
        s5 = Sheet('S5',20,0.0005)
        
        kernel = imagen.Gaussian(xdensity=7,ydensity=7,aspect_ratio=1.0,size=0.3)() - 0.7*imagen.Gaussian(xdensity=7,ydensity=7,aspect_ratio=1.0,size=0.6)()
        kernel = kernel-kernel.mean()
        
        p1 = ConvolutionalProjection("S1toS2",s1,s2,1.0,0.0002,kernel)
        p2 = ConnetcionFieldProjection("S1toS3",s1,s3,1.0,0.0002,kernel)
        p3 = ConvolutionalProjection("S1toS4",s1,s4,1.0,0.0002,kernel)
        p4 = ConnetcionFieldProjection("S1toS5",s1,s5,1.0,0.0002,kernel)

        #Model initialization and execution
        model = Model([s1,s2,s3,s4,s5],0.0001)
        
        g1 = imagen.Gaussian(xdensity=40,ydensity=40,x=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=42),
                             y=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=43),
                          orientation=numbergen.UniformRandom(lbound=-numpy.pi,ubound=numpy.pi,seed=42),
                         size=0.048388, aspect_ratio=4.66667, scale=1.0)
        
        g2 = imagen.Gaussian(xdensity=40,ydensity=40,x=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=12),
                             y=numbergen.UniformRandom(lbound=-0.5,ubound=0.5,seed=13),
                          orientation=numbergen.UniformRandom(lbound=-numpy.pi,ubound=numpy.pi,seed=22),
                          size=0.048388, aspect_ratio=4.66667, scale=1.0)
                
        
        for i in xrange(1):
            s1.set_activity(numpy.maximum(g1(),g2()))
            model.run(0.005)
            self.assertTrue(numpy.sum(s4.get_activity(0)-s5.get_activity(0)) <= 0.0000001)
       
            self.assertTrue(numpy.sum(s2.get_activity(0)-s3.get_activity(0)) <= 0.0000001)
            
    
if __name__ == '__main__':
    unittest.main()
