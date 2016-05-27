#####################################################################################
# This is a simple framework for building rate-models with 2D geometry.             #
#####################################################################################
import numpy
import scipy.signal
import pylab
from visualization import *



class Model(object):
      """
      The model class. Essentially a container of sheets, that makes sure the all update their vm and activities correctly.
      
      All time variables are in seconds!
      
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
          self.dt = dt
          self.time = 0

          # find maximum delay and initialize sheets
          for s in sheets:
              # find maximum delay from that sheet
              max_delay = max([p.delay for p in s.out_projections]+[self.dt])
              s._initialize(int(numpy.ceil(max_delay/dt))+1,self.dt)
      
      def run(self,time):
          # not exact but close enough
          assert abs(time - int(numpy.round(time / self.dt))*self.dt) < 0.000000001*self.dt, "You can only run the network for times that are multiples of dt"
          
          
          for i in xrange(0,int(numpy.round(time/self.dt))):
              
              for s in self.sheets:
                  for p in s.in_projections:
                      p.activate()
                
              for s in self.sheets:
                  s.update()
              
              self.time += self.dt
          
              
          print("Ran network for " + str(time) + " seconds.")
         
      def reset(self):
 		for s in self.sheets:
 		    s.reset()	                    
      
    

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
                 The radius of the sheet in units. Thus the sheet will have radius*2 x radius*2 units.
        
        threshold : float
                  The threshold of the units in the sheet.
        """
        self.radius = radius
        self.vm = numpy.zeros((2*radius,2*radius),dtype=numpy.float32)
        self.threshold = threshold
        self.in_projections = []
        self.out_projections = []
        self.time_constant = time_constant
        self.name = name
        self.not_initialized = True
    
    def _initialize(self,buffer_depth,dt):
        """
        Initializes sheet. Certain information is available onle once sheets are registered in the model.
        This is done here.
        """
        assert self.not_initialized, "Sheet "+ self.name + " has already been initialized"
        self.dt = dt
        self.buffer_index = 0
        self.buffer_depth = buffer_depth
        self.activities = numpy.zeros((buffer_depth,2*self.radius,2*self.radius),dtype=numpy.float32)
        self.not_initialized = False

    
    def get_activity(self,delay):
        """
        Return's sheet activity delay in the past (rounded to the nearest multiple of dt).
        """
        index = numpy.round(delay/self.dt)
        assert index < self.buffer_depth, "ERROR: Activity with delay longer then the depth of activity buffer requested. " + str(self.buffer_depth) + ' ' + str(index)
        return self.activities[(self.buffer_index-index-1) % self.buffer_depth]
    
    def _register_in_projection(self,projection):
        """
        Registers projection as one of the input projection to the sheet.
        """
        self.in_projections.append(projection)

    def _register_out_projection(self,projection):
        """
        Registers projection as one of the output projections to the sheet.
        """
        self.out_projections.append(projection)
    
        
    def update(self):
        assert self.buffer_index < self.buffer_depth
        assert not self.not_initialized
        
        #delete new activities    
        self.activities[self.buffer_index]*=0
        
        # sum the activity comming from all projections
        for p in self.in_projections:
            self.activities[self.buffer_index] += p.activity
        
        #make the dt step 
        self.vm = self.vm + self.dt*(-self.vm+self.activities[self.buffer_index])/self.time_constant
        
        #apply the non-linearity    
        self.activities[self.buffer_index] = self.vm.clip(min=self.threshold)
        
        #once all done, advance our buffer depth
        self.buffer_index = (self.buffer_index + 1) % self.buffer_depth
        
    def reset(self):
        """
        Resets the sheet to be in the same state as after initialization (including he call to _initialize).
        """
        self.activities *= 0
        self.buffer_index = 0
        self.vm *= 0
        
        
class InputSheet(Sheet):
    """
    Sheet for which you can set input. It cannot have any incomming connections.
    
    For now only time invariant input can be set.
    """
    
    def _register_projection(self,projection):
        """
        Registers projection as one of the input projection to the sheet.
        """
        raise Error, "Input sheet cannot accept incomming projections."
    
    def set_activity(self,activity):
        assert numpy.shape(activity) == numpy.shape(self.activities[0])
        for i in xrange(0,self.buffer_depth):
            self.activities[i] = activity
    
    def update(self):
        pass

def applyHebianLearningStepOnAFastConnetcionFieldProjection(projection,learning_rate):
    """
    This method when applied to a ConnetcionFieldProjection will perform a single step of hebbian learning with learning rate *learning_rate*.
    """
    sa = projection.source.get_activity(projection.delay).ravel()
    ta = projection.target.get_activity(0).ravel()
    projection.cfs += learning_rate * numpy.dot(ta[:,numpy.newaxis],sa[numpy.newaxis,:])
    projection.cfs = numpy.multiply(projection.cfs,projection.masks)
    projection.cfs = projection.cfs / numpy.sum(numpy.abs(projection.cfs),axis=1)[:,numpy.newaxis]
    
    
    
    
