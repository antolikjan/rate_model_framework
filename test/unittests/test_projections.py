from rate_model import *
from projections import *
import imagen
import numpy
import unittest


class TestConsistencyOfIndexAndCoordinatesConversion(unittest.TestCase):
    
      def test_correct_boundries(self):  
        s = Sheet('Sheet',1.0,100,0.0005)
        self.assertTrue(s.index_to_coord(0,0) == (-0.495,-0.495))
        self.assertTrue(s.index_to_coord(99,99) == (0.495,0.495))
        
      def test_consistent_back_and_forth_converstions(self):
          s = Sheet('Sheet',1.0,100,0.0005)
          self.assertTrue(self.helper(s,0,0) == (0,0))
          self.assertTrue(self.helper(s,99,99) == (99,99))
          self.assertTrue(self.helper(s,51,50) == (51,50))
          self.assertTrue(self.helper(s,49,33) == (49,33))
          
          s = Sheet('Sheet',1.5,100,0.0005)
          self.assertTrue(self.helper(s,0,0) == (0,0))
          self.assertTrue(self.helper(s,149,149) == (149,149))
          self.assertTrue(self.helper(s,76,75) == (76,75))
          self.assertTrue(self.helper(s,74,33) == (74,33))
      
          s = Sheet('Sheet',1.0,34,0.0005)
          self.assertTrue(self.helper(s,0,0) == (0,0))
          self.assertTrue(self.helper(s,31,31) == (31,31))
          self.assertTrue(self.helper(s,17,16) == (17,16))
          self.assertTrue(self.helper(s,15,13) == (15,13))
          

      def helper(self,sheet,x,y):
          x,y = sheet.index_to_coord(x,y)
          x,y = sheet.coord_to_index(x,y)
          return  x,y

#class TestConsistencyOfProjections(unittest.TestCase):
#    
#    def test_correction_factors(self):
#        s1 = InputSheet('S1',20,0)
#        s2 = Sheet('S2',10,0.0005)
#        s3 = Sheet('S3',10,0.0005)
#        s4 = Sheet('S4',10,0.0005)
#        s5 = Sheet('S5',20,0.0005)
#        s6 = Sheet('S4',20,0.0005)
#        s7 = Sheet('S5',20,0.0005)
#        
#        kernel = imagen.Gaussian(xdensity=7,ydensity=7,aspect_ratio=1.0,size=0.3)() - 0.7*imagen.Gaussian(xdensity=7,ydensity=7,aspect_ratio=1.0,size=0.6)()
#        kernel = kernel-kernel.mean()
#        
#        p1 = ConvolutionalProjection("S1toS2",s1,s2,1.0,0.0002,kernel)
#        p2 = ConnetcionFieldProjection("S1toS3",s1,s3,1.0,0.0002,kernel)
#        p3 = FastConnetcionFieldProjection("S1toS4",s1,s4,1.0,0.0002,kernel)
#        
#        p4 = ConnetcionFieldProjection("S1toS5",s1,s5,1.0,0.0002,kernel)
#        p5 = ConvolutionalProjection("S1toS6",s1,s6,1.0,0.0002,kernel)
#        p6 = FastConnetcionFieldProjection("S1toS7",s1,s7,1.0,0.0002,kernel)
#
#        #Model initialization and execution
#        model = Model([s1,s2,s3,s4,s5,s6,s7],0.0001)
#        s1.set_activity(numpy.random.rand(40,40))
#        model.run(0.005)
#        
#        self.assertTrue(numpy.sum(s2.get_activity(0)-s3.get_activity(0)) <= 0.0000001)
#        self.assertTrue(numpy.sum(s2.get_activity(0)-s4.get_activity(0)) <= 0.0000001)
#        self.assertTrue(numpy.sum(s5.get_activity(0)-s6.get_activity(0)) <= 0.0000001)
#        self.assertTrue(numpy.sum(s5.get_activity(0)-s7.get_activity(0)) <= 0.0000001)
                   


    
if __name__ == '__main__':
    unittest.main()
