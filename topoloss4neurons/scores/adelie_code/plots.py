#plots 

import gudhi
import matplotlib.pyplot as plt

def plot_proj(image):
    #plot the max value projection of a 3D image
    
    plt.imshow(image.max(0));
    plt.colorbar()
    plt.show()
    
def plot_layer(image, layer): 
    #plot a layer of a 3D image
    
    plt.imshow(image[layer]);
    plt.colorbar()
    plt.show()
    
def barcode(pd):
    #plot the gudhi persistence diagrams
    plt = gudhi.plot_persistence_diagram(persistence=pd, legend=True)
    plt.show()