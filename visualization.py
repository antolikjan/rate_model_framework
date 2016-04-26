import rate_model
import pylab
    
def display_model_state(model):
    
    # find the longest number of projections
    max_num_in_projections = 0
    for s in model.sheets:
        max_num_in_projections = max(max_num_in_projections,len(s.in_projections))
    print max_num_in_projections
    pylab.subplot(max_num_in_projections+1, len(model.sheets), 1)
    
    for i in range(len(model.sheets)):
        pylab.subplot(max_num_in_projections, len(model.sheets),  max_num_in_projections*i + 1 )
        pylab.title(model.sheets[i].name)
        pylab.imshow(model.sheets[i].get_activity(0),cmap='gray',interpolation=None)
        pylab.colorbar()
        
        for j in range(len(model.sheets[i].in_projections)):
            print max_num_in_projections*i + j
            pylab.subplot(max_num_in_projections, len(model.sheets),  max_num_in_projections*i + j)
            pylab.title(model.sheets[i].in_projections[j].name)
            pylab.imshow(model.sheets[i].in_projections[j].activity,cmap='gray',interpolation=None)
            pylab.colorbar()
        
        
    