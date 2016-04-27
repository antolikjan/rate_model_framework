"""
This file contains analysis protocols.
"""

def FullfieldSineGratingOrientationTuningProtocol(model,sheets,num_orientation=8,num_phase=10,duration=0.2):
    """
    This analysis will present fullfield sine grating orientation tuning protocol to the model *mode*.
    
    It will collect the final activation of neurons in sheets *sheets* and compute orientation tuning preference and
    selectivity for the neurons in these sheets.
    """
    responses = []
    
    for s in sheets:
        responses[s.name] = []
    
    for i in xrange(num_orientation):
        for j in xrange(num_phase):
            stim = SineGrating(orientation=numpy.pi/num_orientation*i,phase=numpy.pi*2/num_phase*j)()
            retina.set_activity(stim)
            lissom.run(duration)
            for s in sheets:
                s.get_activity(0)    

