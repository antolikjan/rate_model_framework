from rate_model import *
from projections import *
import imagen
import numpy
import unittest


class TestConsistencyOfProjections(unittest.TestCase):
    
    def test_correction_factors(self):
        s1 = InputSheet('S1',20,0)
        s2 = Sheet('S2',10,0.0005)
        s3 = Sheet('S3',10,0.0005)
        s4 = Sheet('S4',10,0.0005)
        s5 = Sheet('S5',20,0.0005)
        s6 = Sheet('S4',20,0.0005)
        s7 = Sheet('S5',20,0.0005)
        
        kernel = imagen.Gaussian(xdensity=7,ydensity=7,aspect_ratio=1.0,size=0.3)() - 0.7*imagen.Gaussian(xdensity=7,ydensity=7,aspect_ratio=1.0,size=0.6)()
        kernel = kernel-kernel.mean()
        
        p1 = ConvolutionalProjection("S1toS2",s1,s2,1.0,0.0002,kernel)
        p2 = ConnetcionFieldProjection("S1toS3",s1,s3,1.0,0.0002,kernel)
        p3 = FastConnetcionFieldProjection("S1toS4",s1,s4,1.0,0.0002,kernel)
        
        p4 = ConnetcionFieldProjection("S1toS5",s1,s5,1.0,0.0002,kernel)
        p5 = ConvolutionalProjection("S1toS6",s1,s6,1.0,0.0002,kernel)
        p6 = FastConnetcionFieldProjection("S1toS7",s1,s7,1.0,0.0002,kernel)

        #Model initialization and execution
        model = Model([s1,s2,s3,s4,s5,s6,s7],0.0001)
        s1.set_activity(numpy.random.rand(40,40))
        model.run(0.005)
        
        self.assertTrue(numpy.sum(s2.get_activity(0)-s3.get_activity(0)) <= 0.0000001)
        self.assertTrue(numpy.sum(s2.get_activity(0)-s4.get_activity(0)) <= 0.0000001)
        self.assertTrue(numpy.sum(s5.get_activity(0)-s6.get_activity(0)) <= 0.0000001)
        self.assertTrue(numpy.sum(s5.get_activity(0)-s7.get_activity(0)) <= 0.0000001)
                   


    
if __name__ == '__main__':
    unittest.main()
