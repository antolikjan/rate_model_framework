#####################################################################################
# This is a simple framework for building rate-models with 2D geometry.             #
#####################################################################################
import numpy
import scipy


class Model(object):
      """
      The model class. Essential a container of sheets, that makes sure the all update their vm and activities correctly.
      
      
      """
      def __init__(self, sheets, dt):
          """
          Parameters
          ----------
          sheets : list(Sheet)
                 A list of sheets that comprise the model.
                 
          dt : float
                 The time step of the simulation
          """
          self.sheets = sheets
          
          # determine the buffer depth by finding the longest delay
          
      
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
    
    def __init__(self,name,radius,time_constant,threshold=0):
        """
        Parameters
        ----------
        radius : int
                 The radious of the sheet in units. Thus the sheet will have radius*2 x radius*2 units.
        
        threshold : float
                  The threshold of the units in the sheet.
        """
        self.vm = numpy.zeros((2*radius,2*radius))
        self.threshold = threshold
        self.in_projections = []
        self.activity_buffer_index
        self.time_constant
        self.name = name
        self.not_initialized = True
    
    def _initialize(self,steps,dt):
        """
        Initializes sheet. Certain information is available onle once sheets are registered in the model.
        This is done here.
        """
        assert self.not_initialized, "Sheet "+ self.name + " has already been initialized"
        self.dt = dt
        self.buffer_index = 0
        self.buffer_depth = steps
        self.activities = numpy.zeros((steps,2*radius,2*radius))
        self.not_initialized = False

    
    def get_activity(delay):
        """
        Return's sheet activity delay in the past (rounded to the nearest multiple of dt).
        """
        index = round(delay/self.dt)
        assert index < self.buffer_depth, "ERROR: Activity with delay longer thenn the depth of activity buffer requested."
        return self.activities[buffer_index-index]
    
    def _register_projection(self,projection):
        """
        Registers projection as one of the input projection to the sheet.
        """
        self.in_projections.append(projection)
        
    def update(self):
        assert self.buffer_index < self.buffer_depth
        self.activities[self.buffer_index]*=0
        
        # sum the activity comming from all projections
        for p in self.in_projections:
            self.activities[self.buffer_index] += p.activate()
        
        #make the dt step 
        self.activities[self.buffer_index] = self.activities[self.buffer_index-1] + self.dt*(-self.activities[self.buffer_index-1]+self.activities[self.buffer_index])/self.time_constant
        
        #applt the non-linearity    
        self.activities[self.buffer_index] = self.activities[self.buffer_index].clip(min=self.threshold)
        
        #once all done, advance our buffer depth
        self.buffer_index = (self.buffer_index + 1) % self.buffer_depth
        

class Projection(object):    
    """
    A set of connections from one sheet to another. This is an abstract class.
    """
    
    def __init__(self, name, source, target, strength):
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
        
        delay : float
                The delay of the projection
        """
        self.name = name
        self.source = source
        self.target = target
        self.strength = strength
        self.delay = delay
        self.target._register_projection(self)
        
    
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
        assert size_diff >=0 , "ERROR: The radios of source sheet of ConvolutionalProjection has to be larger than target"
        
        resp = scipy.signal.convolve2d(self.source.get_activity(self.delay),self.connection_kernel, mode='same')[size_diff:-size_diff,size_diff:-size_diff]
        assert size(resp) == (2*self.target.radius,2*self.target.radius)
        
        return resp * self.strength
        
