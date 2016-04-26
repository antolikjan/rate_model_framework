import rate_model
import pylab
    
def display_model_state(model):
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
