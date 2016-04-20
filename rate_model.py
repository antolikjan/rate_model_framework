#####################################################################################
# This is a simple framework for building rate-models with 2D geometry.             #
#####################################################################################
import numpy
import scipy


class Model(object):
      """
      The model class. Essential a container of sheets, that makes sure the all update their vm and activities correctly.
      """
      def __init__(self, sheets):
          """
          Parameters
          ----------
          sheets : list(Sheet)
                 A list of sheets that comprise the model.
          """
          self.sheets = sheets
      
      def run(self,number_of_step):
          for i in xrange(0,number_of_step):
              for s in sheets:
                  s.update_vm()
            for s in sheets:
                  s.update_activities()
                    
      
    

class Sheet(object):
    """
    This class represent a 2D matrix of units. 
    
    Note that:
        1. all spatial parameters of the framework are in these units. 
        2. all sheets are centered on each other
        3. a consequence of 1 is that you cannot have sheets of different 'density'
    """
    
    def __init__(self, radius,threshold=0):
        """
        Parameters
        ----------
        radius : int
                 The radious of the sheet in units. Thus the sheet will have radius*2 x radius*2 units.
        
        threshold : float
                  The threshold of the units in the sheet.
        """
        self.vm = numpy.zeros((2*radius,2*radius))
        self.activities = numpy.zeros((2*radius,2*radius))
        self.threshold = threshold
        self.in_projections = []
        
    def update_vm(self):
        self.vm = numpy.zeros((2*radius,2*radius))
        for p in self.in_projections:
            self.vm = self.vm + p.activate()    
        
    def update_activities(self):        
        self.activities = self.vm.clip(min=0)
        assert

class Projection(object):    
    """
    A set of connections from one sheet to another. This is an abstract class.
    """
    
    def __init__(self, source, target, strength):
        """
        Each Projection has a source sheet and target sheet.
        
        Parameters
        ----------
        source : Sheet
                 The sheet from which projection passes rates.

        target : Sheet
                 The sheet to which projection passes rates.
                 
        strength : float
                 The strength of the projection.
        """
        self.source = source
        self.target = target
        self.strength = strength
        self.target.in_projections.append(self)
        
    
    def activate(self):
        """
        This returns a matrix of the same size as the target sheet, which corresponds to the contributions from this projections to the individual neurons.
        """
        raise NotImplementedError
        
    

class ConvolutionalProjection(Projection):    
    """
    A projection which has only one set of connections (connection kernel) which are assumed to be the same for all target neurons, except being centered on their position.
    This means that the output of ConvolutionalProjection is the spatial convolution of this connection kernel with the activities of the source Sheet.
    """
    def __init__(self,connection_kernel,source, target, strength):
        """
        Each Projection has a source sheet and target sheet.
        
        Parameters
        ----------
        source : Sheet
                 The sheet from which projection passes rates.

        target : Sheet
                 The sheet to which projection passes rates.
                 
        strength : float
                 The strength of the projection.
        """
        Projection.__init__(self, source, target, strength)
        self.connection_kernel = connection_kernel
        
    
    def activate(self):
        """
        This returns a matrix of the same size as the target sheet, which corresponds to the contributions from this projections to the individual neurons.
        """
        size_diff = self.source.radius - self.target.radius
        assert size_diff >=0 , "ERROR: The radious of source sheet of ConvolutionalProjection has to be larger than target"
        
        resp = scipy.signal.convolve2d(self.source.activities,self.connection_kernel, mode='same')[size_diff:-size_diff,size_diff:-size_diff]
        assert size(resp) == size(self.target.activities)
        
        return resp * self.strength
        
