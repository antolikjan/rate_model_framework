import numpy
import scipy
from imagen.boundingregion import BoundingBox
import imagen


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
        self.changed =True
        
    
    def activate(self):
        """
        This returns a matrix of the same size as the target sheet, which corresponds to the contributions from this projections to the individual neurons.
        """
        raise NotImplementedError   


class FastConnetcionFieldProjection(Projection):    
    """
    A projection which has can have different connection field for each target neuron, which are assumed to be centered on the target's neuron position.
    This means that the output of ConvolutionalProjection is the spatial convolution of this connection kernel with the activities of the source Sheet.
    
    Note that all connection fields, including the border ones, will be normalized so that the sum of their absolute values is 1.    
    """
    def __init__(self,name,source,target,strength,delay,radius,initial_connection_kernel):
        """
        Each Projection has a source sheet and target sheet.
        
        Parameters
        ----------
        radius : float
                 How big the connections fields will be in the source sheet coordinates!   
        source : Sheet
                 The sheet from which projection passes rates.

        target : Sheet
                 The sheet to which projection passes rates.
                 
        strength : float
                 The strength of the projection.
                 
        inititial_connection_kernel: ndarray
                 The initial connection kernel. 
        """
        Projection.__init__(self, name,source, target, strength,delay)
        self.radius = radius
        
        mask = imagen.Disk(xdensity = source.density,ydensity = source.density,bounds = BoundingBox(radius=source.size/2),size=self.radius*2,smoothing=0)
        initial_connection_kernel.xdensity = source.density
        initial_connection_kernel.ydensity = source.density
        initial_connection_kernel.bounds = BoundingBox(radius=source.size/2.0)
        
        source_dim = self.source.unit_diameter
        target_dim = self.target.unit_diameter
        self.cfs = numpy.zeros((target_dim*target_dim,source_dim*source_dim),dtype=numpy.float32)
        self.masks = numpy.zeros((target_dim*target_dim,source_dim*source_dim),dtype=numpy.float32)
        
        for i in xrange(target_dim):
            for j in xrange(target_dim):
                ci,cj = self.target.index_to_coord(i,j)
                self.cfs[i*target_dim+j,:] = initial_connection_kernel(x=cj,y=-ci).flatten()
                self.masks[i*target_dim+j,:] = mask(x=cj,y=-ci).flatten()
        
        #apply masks and normalize
        self.cfs = numpy.multiply(self.cfs,self.masks)
        self.cfs = self.cfs / numpy.sum(numpy.abs(self.cfs),axis=1)[:,numpy.newaxis]
  
        assert numpy.min(self.masks) == 0 and numpy.max(self.masks) == 1
        self.activity = numpy.zeros((target_dim,target_dim),dtype=numpy.float32)
    
    def activate(self):
        """
        This returns a matrix of the same size as the target sheet, which corresponds to the contributions from this projections to the individual neurons.
        """
        sa = self.source.get_activity(self.delay).ravel()
        self.activity = self.strength * numpy.reshape(numpy.dot(self.cfs,sa),(self.target.unit_diameter,self.target.unit_diameter))
        
    def get_cf(self,posx,posy):
        ci,cj = self.target.index_to_coord(posx,posy)
        xi,yi = self.source.coord_to_index(ci-self.radius,cj-self.radius,clipped=True)
        xa,ya = self.source.coord_to_index(ci+self.radius,cj+self.radius,clipped=True)
        return numpy.reshape(self.cfs[posx*self.target.unit_diameter+posy,:],(self.source.unit_diameter,self.source.unit_diameter))[xi:xa,yi:ya]
        
    def applyHebianLearningStep(self,learning_rate):
        """
        This method when applied to a ConnetcionFieldProjection will perform a single step of hebbian learning with learning rate *learning_rate*.
        """
        sa = self.source.get_activity(self.delay).ravel()
        ta = self.target.get_activity(self.target.dt).ravel()
        self.cfs += learning_rate * numpy.dot(ta[:,numpy.newaxis],sa[numpy.newaxis,:])
        self.cfs = numpy.multiply(self.cfs,self.masks)
        self.cfs = self.cfs / numpy.sum(numpy.abs(self.cfs),axis=1)[:,numpy.newaxis]

class ConvolutionalProjection(Projection):    
    """
    A projection which has only one set of connections (connection kernel) which are assumed to be the same for all target neurons, except being centered on their position.
    This means that the output of ConvolutionalProjection is the spatial convolution of this connection kernel with the activities of the source Sheet.
    
    Note that the connection_kernel will be normalized so that the sum of it's absolute values is 1.
    
    ConvolutionalProjection is only allowed between sheets of the same size and density.
    """
    def __init__(self,name,source,target,strength,delay,connection_kernel):
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
        self.connection_kernel = connection_kernel #/ numpy.sum(numpy.abs(connection_kernel))
        assert self.source.size == self.target.size , "ERROR: The size of source sheet of ConvolutionalProjection has to be same as that of target sheet"
        assert self.source.density == self.target.density , "ERROR: The density of source sheet of ConvolutionalProjection has to be same as that of target sheet"
        assert self.target.unit_diameter == self.source.unit_diameter
        assert numpy.shape(connection_kernel)[0] % 2 == 1, "ERROR: initial kernel for ConvolutionalProjection has to have odd radius"
        #lets calculate edge correction factors:
        self.rad = int((numpy.shape(connection_kernel)[0]-1)/2)
        dim = self.target.unit_diameter
        cfs = [[self.connection_kernel[max(0,self.rad-j):2*self.rad+1-max(0,(j+1+self.rad)-dim),max(0,self.rad-i):2*self.rad+1-max(0,(i+1+self.rad)-dim)] for i in xrange(dim)] for j in xrange(dim)]
        self.corr_factors = numpy.array([[numpy.sum(numpy.abs(connection_kernel))/numpy.sum(numpy.abs(b)) for b in a] for a in cfs])
        
    def activate(self):
        """
        This returns a matrix of the same size as the target sheet, which corresponds to the contributions from this projections to the individual neurons.
        """
        resp = scipy.signal.fftconvolve(self.source.get_activity(self.delay),self.connection_kernel, mode='same')
        assert numpy.shape(resp) == (self.target.unit_diameter,self.target.unit_diameter), "ERROR: The size of calculated projection respone is " + str(numpy.shape(resp)) + "units, while the size of target sheet is " + str((self.target.unit_diameter,self.target.unit_diameter)) + " units"
        self.activity = numpy.multiply(resp,self.corr_factors) * self.strength
