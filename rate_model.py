#####################################################################################
# This is a simple framework for building rate-models with 2D geometry.             #
#####################################################################################
import numpy
import scipy.signal


class Model(object):
      """
      The model class. Essentially a container of sheets, that makes sure the all update their vm and activities correctly.
      
      All time variables are in miliseconds!
      
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
          # find maximum delay 
          
          # initialize sheets
          for s in sheets:
              # find maximum delay from that sheet
              max_delay = max([p.delay for p in s.out_projections]+[self.dt])
              s._initialize(numpy.ceil(max_delay/dt),self.dt)
      
      def run(self,time):
          # not exact but close enough
          assert abs(time - int(numpy.round(time / self.dt))*self.dt) < 0.000000001*self.dt, "You can only run the network for times that are multiples of dt"
          
          for i in xrange(0,int(numpy.round(time/self.dt))):
              for s in self.sheets:
                  for p in s.in_projections:
                      p.activate()
                
              for s in self.sheets:
                  s.update()
          
          print("Ran network for " + str(time) + "ms")
         
                    
      
    

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
        self.vm = numpy.zeros((2*radius,2*radius))
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
        self.activities = numpy.zeros((buffer_depth,2*self.radius,2*self.radius))
        self.not_initialized = False

    
    def get_activity(self,delay):
        """
        Return's sheet activity delay in the past (rounded to the nearest multiple of dt).
        """
        index = numpy.round(delay/self.dt)-1
        assert index < self.buffer_depth, "ERROR: Activity with delay longer thenn the depth of activity buffer requested."
        return self.activities[self.buffer_index-index]
    
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
        assert size(activity) == size(self.activities[0])
        for i in xrange(0,self.buffer_depth):
            self.activities[i] = activity
    
    def update(self):
        pass

class Projection(object):    
    """
    A set of connections from one sheet to another. This is an abstract class.
    """
    
    def __init__(self, name, source, target, strength,delay):
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
        assert delay > 0, "ERROR: zero delay projections are not allowed"
        self.target._register_in_projection(self)
        self.source._register_out_projection(self)
        
    
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
    def __init__(self,name,source, target, strength,delay,connection_kernel):
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
        Projection.__init__(self, name,source, target, strength,delay)
        self.connection_kernel = connection_kernel
        
        
    
    def activate(self):
        """
        This returns a matrix of the same size as the target sheet, which corresponds to the contributions from this projections to the individual neurons.
        """
        size_diff = self.source.radius - self.target.radius
        assert size_diff >=0 , "ERROR: The radios of source sheet of ConvolutionalProjection has to be larger than target"
        resp = scipy.signal.convolve2d(self.source.get_activity(self.delay),self.connection_kernel, mode='same')
        if size_diff != 0:
            resp = resp[size_diff:-size_diff,size_diff:-size_diff]
        assert numpy.shape(resp) == (2*self.target.radius,2*self.target.radius), "ERROR: The size of calculated projection respone is " + str(numpy.shape(resp)) + "units, while the size of target sheet is " + str((2*self.target.radius,2*self.target.radius)) + " units"
         
        self.activity = resp * self.strength
        
