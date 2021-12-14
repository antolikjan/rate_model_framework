#####################################################################################
# This is a simple framework for building rate-models with 2D geometry.             #
#####################################################################################
from visualization import *


class Model(object):
    """
    The model class. Essentially a container of sheets
    that makes sure the all update their vm and activities correctly.
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
            max_delay = max([p.delay for p in s.out_projections] + [self.dt])
            s._initialize(int(numpy.ceil(max_delay / dt)) + 1, self.dt)

    def run(self, time):
        # not exact but close enough
        assert abs(time - int(numpy.round(time / self.dt)) * self.dt) < 0.000000001 * self.dt, \
            "You can only run the network for times that are multiples of dt"

        for i in range(0, int(numpy.round(time / self.dt))):
            for s in self.sheets:
                for p in s.in_projections:
                    if p.source.changed:
                        p.activate()

            for s in self.sheets:
                s.update()

            for s in self.sheets:
                s.changed = s.tmp_changed

            self.time += self.dt

        print("Ran network for " + str(time) + " seconds.")

    def reset(self):
        for s in self.sheets:
            s.reset()


class Sheet(object):
    """
    This class represent a 2D matrix of units. 
    
    Note that:
        All sheets are centered on each other.
    """

    def __init__(self, name, size, density, time_constant, threshold=0):
        """
        Parameters
        ----------
        size : float
                 The size of the sheet in arbitrary units. Note that units will have coordinates of -size/2, size/2
        
        density : float
                 The density of the sheet. Thus the sheet will have round(size*density) x round(size*density) units.
                 
        time_constant:
                 The time constant of the neurons in the sheet.
        
        threshold : float
                  The threshold of the units in the sheet.
        """
        self.buffer_index = 0
        self.size = size
        self.density = density
        assert density * (
                size / 2) % 1 == 0, \
            "Density of the sheet times radius (size/2) has to be integer, but %f * %f / 2 = %f" % (
                density, size, density * (size / 2)
            )
        self.unit_diameter = int(numpy.floor(self.size / 2 * self.density) * 2)
        self.changed = True
        self.tmp_changed = True
        self.vm = numpy.zeros((self.unit_diameter, self.unit_diameter), dtype=numpy.float32)
        self.threshold = numpy.ones(self.vm.shape) * threshold
        self.in_projections = []
        self.out_projections = []
        self.time_constant = time_constant
        self.name = name
        self.not_initialized = True

    def _initialize(self, buffer_depth, dt):
        """
        Initializes sheet. Certain information is available only once sheets are registered in the model.
        This is done here.
        """
        assert self.not_initialized, "Sheet " + self.name + " has already been initialized"
        self.dt = dt
        self.buffer_index = 0
        self.buffer_depth = buffer_depth
        self.activities = numpy.zeros((buffer_depth, self.unit_diameter, self.unit_diameter), dtype=numpy.float32)
        self.not_initialized = False

    def get_activity(self, delay):
        """
        Return's sheet activity delay in the past (rounded to the nearest multiple of dt).
        """
        assert delay > 0
        index = numpy.round(delay / self.dt)
        assert index < self.buffer_depth, \
            "ERROR: Activity with delay longer then the depth of activity buffer requested: " + \
            str(self.buffer_depth) + ' ' + str(index)
        assert index > 0, "ERROR: Activity with delay less than timestep of simulation requested: " + \
                          str(self.dt) + ' ' + str(delay)
        return self.activities[int((self.buffer_index - index) % self.buffer_depth)]

    def _register_in_projection(self, projection):
        """
        Registers projection as one of the input projection to the sheet.
        """
        self.in_projections.append(projection)

    def _register_out_projection(self, projection):
        """
        Registers projection as one of the output projections to the sheet.
        """
        self.out_projections.append(projection)

    def update(self):
        assert self.buffer_index < self.buffer_depth
        assert not self.not_initialized

        # delete new activities
        self.activities[self.buffer_index] *= 0

        # sum the activity coming from all projections
        for p in self.in_projections:
            self.activities[self.buffer_index] += p.activity

        # make the dt step
        self.vm = self.vm + self.dt * (-self.vm + self.activities[self.buffer_index]) / self.time_constant

        # apply the non-linearity
        self.activities[self.buffer_index] = (self.vm - self.threshold).clip(0)

        # once all done, advance our buffer depth
        self.buffer_index = (self.buffer_index + 1) % self.buffer_depth

    def reset(self):
        """
        Resets the sheet to be in the same state as after initialization (including he call to _initialize).
        """
        self.activities *= 0
        self.buffer_index = 0
        self.vm *= 0
        self.tmp_changed = True
        self.changed = True

    def index_to_coord(self, x, y):
        """
        Returns indexes of a unit at coordinates x and y.
        """

        assert self.unit_diameter > x >= 0, "Indexes out of bounds (%f). Asked for (%f,%f)" % (
            self.unit_diameter, x, y
        )
        assert self.unit_diameter > y >= 0, "Indexes out of bounds (%f). Asked for (%f,%f)" % (
            self.unit_diameter, x, y
        )
        coord_x = (x - self.unit_diameter / 2) / (self.density * 1.0) + 1.0 / (2.0 * self.density)
        coord_y = (y - self.unit_diameter / 2) / (self.density * 1.0) + 1.0 / (2.0 * self.density)
        return coord_x, coord_y

    def coord_to_index(self, x, y, clipped=False):
        """
        Returns coordinates of a unit at indexes x and y.
        
        If clipped is true, if the coordinates are out of the bounds of the sheet, it will return
        indexes that are clipped to the minimum or maximum allowed indexes. 
        """

        if not clipped:
            assert -self.size / 2 <= x <= self.size / 2, "Coordinates (%f,%f) out of bounds (%f,%f)" % (
                x, y, -self.size / 2, self.size / 2
            )
            assert -self.size / 2 <= y <= self.size / 2, "Coordinates (%f,%f) out of bounds (%f,%f)" % (
                x, y, -self.size / 2, self.size / 2
            )
        index_x = numpy.round(x * self.density - 1.0 / (2 * self.density)) + self.unit_diameter / 2
        index_y = numpy.round(y * self.density - 1.0 / (2 * self.density)) + self.unit_diameter / 2

        if clipped:
            if index_x < 0:
                index_x = 0
            elif index_y < 0:
                index_y = 0
            elif index_x > self.unit_diameter - 1:
                index_x = self.unit_diameter - 1
            elif index_y > self.unit_diameter - 1:
                index_y = self.unit_diameter - 1

        return index_x, index_y


class NoTimeConstantSheet(Sheet):
    """
    Same as Sheet, but activity is computed instantaneously without taking into consideration time-constant or delays.
    """

    def __init__(self, name, size, density, time_constant, threshold=0):
        """
        Parameters
        ----------
        size : float
                 The size of the sheet in arbitrary units. Note that units will have coordinates of -size/2, size/2
        
        density : float
                 The density of the sheet. Thus the sheet will have round(size*density) x round(size*density) units.
                 
        time_constant:
                 The time constant of the neurons in the sheet.
        
        threshold : float
                  The threshold of the units in the sheet.
        """
        super().__init__(name, size, density, time_constant, threshold)
        self.activities = 0
        self.size = size
        self.density = density
        assert density * (
                size / 2) % 1 == 0, \
            "Density of the sheet times radius (size/2) has to be integer, but %f * %f / 2 = %f" % (
            density, size, density * (size / 2)
            )
        self.unit_diameter = int(numpy.floor(self.size / 2 * self.density) * 2)
        self.changed = True
        self.tmp_changed = True
        self.threshold = threshold
        self.in_projections = []
        self.out_projections = []
        self.name = name
        self.not_initialized = True

    def _initialize(self, buffer_depth, dt):
        """
        Initializes sheet. Certain information is available only once sheets are registered in the model.
        This is done here.
        """
        assert self.not_initialized, "Sheet " + self.name + " has already been initialized"
        self.activities = numpy.zeros((self.unit_diameter, self.unit_diameter), dtype=numpy.float32)
        self.not_initialized = False

    def get_activity(self, delay):
        """
        Return's sheet activity delay in the past (rounded to the nearest multiple of dt).
        """
        return self.activities

    def update(self):
        assert not self.not_initialized

        # sum the activity coming from all projections
        for p in self.in_projections:
            self.activities += p.activity

        # apply the non-linearity
        self.activities = (self.activities - self.threshold).clip(0)

        self.tmp_changed = sum([p.source.changed for p in self.in_projections])

    def reset(self):
        """
        Resets the sheet to be in the same state as after initialization (including he call to _initialize).
        """
        self.activities *= 0
        self.tmp_changed = True
        self.changed = True


class InputSheet(NoTimeConstantSheet):
    """
    Sheet for which you can set input. It cannot have any incoming connections.
    
    For now only time invariant input can be set.
    """

    def __init__(self, *params):
        NoTimeConstantSheet.__init__(self, *params)

    def _register_projection(self, projection):
        """
        Registers projection as one of the input projection to the sheet.
        """
        raise Exception("Input sheet cannot accept incoming projections.")

    def set_activity(self, activity):
        assert numpy.shape(activity) == numpy.shape(self.activities)
        self.activities = activity
        self.tmp_changed = True
        self.changed = True

    def update(self):
        self.tmp_changed = False


class HomeostaticSheet(Sheet):
    """
    Sheet with homeostatic control of activity of individual neurons.
    """

    def __init__(self, name, size, density, time_constant, init_threshold=0, alpha=0.01, mu=0.1, smoothing=0.991):
        """
        Parameters
        ----------
        size : float
                 The size of the sheet in arbitrary units. Note that units will have coordinates of -size/2, size/2

        density : float
                 The density of the sheet. Thus the sheet will have round(size*density) x round(size*density) units.
                 
        time_constant:
                 The time constant of the neurons in the sheet.
        
        threshold : float
                  The threshold of the units in the sheet.
        """
        Sheet.__init__(self, name, size, density, time_constant, init_threshold)
        self.alpha = alpha
        self.smoothing = smoothing
        self.mu = mu
        self.y_avg = numpy.ones((self.unit_diameter, self.unit_diameter), dtype=numpy.float32) * self.mu

    def apply_homeostatic_thresh(self):
        # Using activity matrix and and smoothed activity from *previous* call.    
        self.y_avg = (1.0 - self.smoothing) * self.get_activity(self.dt) + self.smoothing * self.y_avg
        self.threshold = self.threshold + self.alpha * (self.y_avg - self.mu)
