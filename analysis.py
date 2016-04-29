"""
This file contains analysis protocols.
"""
import matplotlib.gridspec as gridspec
import numpy
from imagen import SineGrating
import pylab

def fullfieldSineGratingOrientationTuningProtocol(model,retina,sheets=None,num_orientation=8,num_phase=10,duration=0.04,filename=None,plot=False,load=False):
    """
    This analysis will present fullfield sine grating orientation tuning protocol to the model *model*.
    
    It will collect the final activation of neurons in sheets *sheets* (if None all sheets in the model are analyzed) and compute orientation tuning preference and
    selectivity for the neurons in these sheets.
    
    If filename is not None it will save the collected data to file, filename
    """
    if not load:
        responses = {}
        
        if sheets == None:
           sheets = model.sheets
        
        #initialize the 
        for s in sheets:
            responses[s.name] = numpy.zeros((num_orientation,num_phase,(s.radius*2)**2))
        
        #present the stimulation protocol and collect data
        for i in xrange(num_orientation):
            for j in xrange(num_phase):
                stim = SineGrating(orientation=numpy.pi/num_orientation*i,phase=numpy.pi*2/num_phase*j,xdensity=72,ydensity=72)()
                retina.set_activity(stim)
                model.run(duration)
                for s in sheets:
                    responses[s.name][i,j,:] = s.get_activity(0).copy().ravel()    
        if filename != None:
           import pickle
           f = open(filename,'wb')
           pickle.dump(responses,f)
           f.close()
    else:
           import pickle
           f = open(filename,'rb')
           responses = pickle.load(f)
           f.close()
    
    # lets calculate the orientation preference and selectivity
    angles = [numpy.pi/num_orientation*i for i in xrange(num_orientation)] 
    angles_as_complex_numbers = numpy.cos(angles) + numpy.sin(angles) * 1j
    
    orientation_preference_maps = {}
    orientation_selectivity_maps = {}
    
    for k in responses.keys():
        # first let's select the response to each orientation as the maximum response across phases
        resp = numpy.max(responses[k],axis=1)
        
        pylab.figure()
        pylab.subplot(1,len(angles),1)
        sheet_size = numpy.sqrt(len(resp[0]))
        print sheet_size
        for i in xrange(len(angles)):
                pylab.subplot(1,len(angles),i)
                im = pylab.imshow(numpy.resize(resp[i],(sheet_size,sheet_size)),interpolation='nearest',cmap='gray')
                pylab.axis('off')    
                pylab.colorbar(im,fraction=0.046, pad=0.04)
           
        
        pylab.show()
        # calculate selectivity and preference as the angle and magnitude of the mean of the responses projected into polar coordinates based on the orientation angle
        # to which they were collected
        polar_resp = resp * numpy.array(angles_as_complex_numbers)[:,numpy.newaxis]
        polar_mean_resp = numpy.mean(polar_resp,axis=0)
        
        orientation_preference_maps[k] = numpy.angle(polar_mean_resp)
        orientation_selectivity_maps[k] = numpy.absolute(polar_mean_resp)
    
    # make a plot of the maps if asked for
    if plot:
       gs = gridspec.GridSpec(2, len(responses.keys()))
       gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05,hspace=0.1,wspace=0.1)
       
       for i,s in enumerate(responses.keys()):
           sheet_size = numpy.sqrt(len(orientation_preference_maps[k]))
           pylab.subplot(gs[0,i])
           im=pylab.imshow(numpy.resize(orientation_preference_maps[k],(sheet_size,sheet_size)),interpolation='nearest')
           pylab.axis('off')    
           pylab.colorbar(im,fraction=0.046, pad=0.04)
           
           pylab.subplot(gs[1,i])
           im=pylab.imshow(numpy.resize(orientation_selectivity_maps[k],(sheet_size,sheet_size)),interpolation='nearest',cmap='gray')
           pylab.axis('off')
           pylab.colorbar(im,fraction=0.046, pad=0.04)
        
    
    return orientation_preference_maps,orientation_selectivity_maps