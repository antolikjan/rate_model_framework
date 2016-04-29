import rate_model
import pylab
import matplotlib.gridspec as gridspec
import numpy
    
def display_model_state(model,filename=None):
    # find the longest number of projections
    max_num_in_projections = 0
    for s in model.sheets:
        max_num_in_projections = max(max_num_in_projections,len(s.in_projections))
    print max_num_in_projections
    pylab.subplot(max_num_in_projections+1, len(model.sheets), 1)
    
    print max_num_in_projections+1
    
    for i in range(len(model.sheets)):
        print (max_num_in_projections+1)*i + 1
        pylab.subplot(max_num_in_projections+1, len(model.sheets),  i + 1 )
        pylab.title(model.sheets[i].name)
        im = pylab.imshow(model.sheets[i].get_activity(0),cmap='gray',interpolation='nearest')
        pylab.colorbar(im,fraction=0.046, pad=0.04)
        
        for j in range(len(model.sheets[i].in_projections)):
            pylab.subplot(max_num_in_projections+1, len(model.sheets),  len(model.sheets)*(j+1) + i+1)
            pylab.title(model.sheets[i].in_projections[j].name)
            im = pylab.imshow(model.sheets[i].in_projections[j].activity,cmap='gray',interpolation='nearest')
            pylab.colorbar(im,fraction=0.046, pad=0.04)
        
    
    pylab.tight_layout()
    if filename:
       pylab.savefig(filename,dpi=600)


def plot_projection(projection,downsample=0.2,filename=None):
    """
    Plots the connection fields in the projection. Only *downsample* fraction (evenly spaced) of the connection fields will be shown
    """
    
    size = int(numpy.floor(numpy.floor(2*projection.target.radius * downsample)))
    
    step = int(numpy.round(1/downsample))
    
    gs = gridspec.GridSpec(int(size), int(size))
    gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05,hspace=0.1,wspace=0.1)
    
    for i in range(size):
        for j in range(size):
            pylab.subplot(gs[int(i),int(j)])
            pylab.imshow(projection.get_cf(i*step,j*step),interpolation='none',cmap='gray')
            pylab.axis('off')
    
    if filename:
       pylab.savefig(filename,dpi=600)
            
        
