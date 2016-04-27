#####################################################################################
# This is a simple framework for building rate-models with 2D geometry.             #
#####################################################################################
import numpy
import scipy.signal
import pylab
from visualization import *

#pylab.ion()

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
          
              
          print("Ran network for " + str(time) + "seconds.")
         
                    
      
    

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
        
        #pylab.figure()
        #pylab.imshow(self.vm,cmap='gray',interpolation=None)
        #pylab.title('sheet')
        #pylab.show()

        
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
        self.activity = None
        
    
    def activate(self):
        """
        This returns a matrix of the same size as the target sheet, which corresponds to the contributions from this projections to the individual neurons.
        """
        raise NotImplementedError
        
    

class ConvolutionalProjection(Projection):    
    """
    A projection which has only one set of connections (connection kernel) which are assumed to be the same for all target neurons, except being centered on their position.
    This means that the output of ConvolutionalProjection is the spatial convolution of this connection kernel with the activities of the source Sheet.
    
    Note that the connection_kernel will be normalized so that the sum of it's absolute values is 1.
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
        self.connection_kernel = connection_kernel / numpy.sum(numpy.abs(connection_kernel))
        
        assert numpy.shape(connection_kernel)[0] % 2 == 1, "ERROR: initial kernel for ConnetionFieldProjection has to have add radius"
        
        #lets calculate edge correction factors:
        self.rad = int((numpy.shape(connection_kernel)[0]-1)/2)
        target_dim = self.target.radius*2
        cfs = [[self.connection_kernel[max(0,self.rad-j):target_dim-max(0,(j+self.rad)-target_dim),max(0,self.rad-i):target_dim-max(0,(i+self.rad)-target_dim)] for i in xrange(target_dim)] for j in xrange(target_dim)]
        self.corr_factors = numpy.array([[1/numpy.sum(numpy.abs(b)) for b in a] for a in cfs])
        
    def activate(self):
        """
        This returns a matrix of the same size as the target sheet, which corresponds to the contributions from this projections to the individual neurons.
        """
        size_diff = self.source.radius - self.target.radius
        assert size_diff >=0 , "ERROR: The radios of source sheet of ConvolutionalProjection has to be larger than target"
        resp = scipy.signal.fftconvolve(self.source.get_activity(self.delay),self.connection_kernel, mode='same')
        if size_diff != 0:
            resp = resp[size_diff:-size_diff,size_diff:-size_diff]
        assert numpy.shape(resp) == (2*self.target.radius,2*self.target.radius), "ERROR: The size of calculated projection respone is " + str(numpy.shape(resp)) + "units, while the size of target sheet is " + str((2*self.target.radius,2*self.target.radius)) + " units"
        
        self.activity = numpy.multiply(resp,self.corr_factors) * self.strength



class ConnetcionFieldProjection(Projection):    
    """
    A projection which has only one set of connections (connection kernel) which are assumed to be the same for all target neurons, except being centered on their position.
    This means that the output of ConvolutionalProjection is the spatial convolution of this connection kernel with the activities of the source Sheet.
    
    Note that all connection fields, including the border ones, will be normalized so that the sum of their absolute values is 1.
    
    """
    def __init__(self,name,source, target, strength,delay,initial_connection_kernel):
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
        assert numpy.shape(initial_connection_kernel)[0] % 2 == 1, "ERROR: initial kernel for ConnetionFieldProjection has to have add radius"
        self.rad = int((numpy.shape(initial_connection_kernel)[0]-1)/2)
        target_dim = self.target.radius*2
        initial_connection_kernel = initial_connection_kernel/numpy.sum(numpy.abs(initial_connection_kernel))
        self.cfs = [[initial_connection_kernel.copy()[max(0,self.rad-j):target_dim-max(0,(j+self.rad)-target_dim),max(0,self.rad-i):target_dim-max(0,(i+self.rad)-target_dim)] for i in xrange(target_dim)] for j in xrange(target_dim)]
        
        # make sure we created correct sizes
        for i in xrange(self.target.radius*2):
            for j in xrange(self.target.radius*2):
                assert numpy.all(numpy.shape(self.cfs[i][j]) <= numpy.shape(initial_connection_kernel)) and numpy.all(numpy.shape(self.cfs[i][j]) > (self.rad,self.rad)) , "ERROR: Connection field created at location " + str(i) + "," + str(j) + " that has size: " + str(numpy.shape(self.cfs[i][j])) + " while template size is:" + str(numpy.shape(initial_connection_kernel))
                self.cfs[i][j] = self.cfs[i][j].flatten()    
        
        self.activity = numpy.zeros((target_dim,target_dim))
    
    def activate(self):
        """
        This returns a matrix of the same size as the target sheet, which corresponds to the contributions from this projections to the individual neurons.
        """
        size_diff = self.source.radius - self.target.radius
        assert size_diff >=0 , "ERROR: The radius of source sheet of ConvolutionalProjection has to be larger than target"
        
        sa = self.source.get_activity(self.delay)
        
        for i in xrange(self.target.radius*2):
            for j in xrange(self.target.radius*2):
                self.activity[i][j] = numpy.dot(self.cfs[i][j],sa[max(i-self.rad,0):i+self.rad+1,max(j-self.rad,0):j+self.rad+1].ravel())
                
        self.activity = self.activity * self.strength



def applyHebianLearningStepOnAConnetcionFieldProjection(projection,learning_rate):
    """
    This method when applied to a ConnetcionFieldProjection will perform a single step of hebbian learning with learning rate *learning_rate*.
    """
    
    sa = projection.source.get_activity(projection.delay)
    pa = projection.target.get_activity(delay)
    
    for i in xrange(projection.target.radius*2):
            for j in xrange(projection.target.radius*2):
                projection.cfs[i][j] = projection.cfs[i][j]+ pa[i][j]*sa[max(i-self.rad,0):i+self.rad+1,max(j-self.rad,0):j+self.rad+1].ravel()
    
                # we have to renormalize the connection field
                projection.cfs[i][j] = projection.cfs[i][j] / numpy.sum(numpy.abs(projection.cfs[i][j]))
    
    
    
    
    
    