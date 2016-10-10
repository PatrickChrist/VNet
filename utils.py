import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion,generate_binary_structure



def generate_weightmap_from_label(label,class_weight=None,border_thickness=2,stepness=0.1,shape=(128,128,64)):
    ## Get Number of pixels for the current class
    if class_weight == None:
        class_weight = label.size/np.sum(label.ravel())
    
    ## Reshape for morphology operations
    label3d=label.reshape(shape)
    
    # Generate the weightmap
    weightmap= np.zeros_like(label3d)
    label3d_eroded=label3d
    label3d_dilate=label3d
    # Erosion Loop
    for step in range(border_thickness):
        label3d_eroded=label3d_eroded - 1.0/border_thickness*stepness * (binary_erosion(label3d_eroded)+label3d_eroded)
        label3d_dilate=label3d_dilate + 1.0/border_thickness * (binary_dilation(label3d_dilate)-label3d_dilate)
    weightmap = label3d_eroded + label3d_dilate - label3d
    
    weightmap = weightmap*class_weight
    return  weightmap.ravel()