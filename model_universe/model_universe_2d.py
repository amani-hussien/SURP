import plotly.graph_objects as plotly
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.ndimage as sci

# load universe parameters and build =======

def load_universe(file="universe_parameters.json"): 
    """
    Load a set of Gaussian spheres from a JSON file and generate a 3D
    density field. 

    Each sphere contributes a Gaussian blob defined by its center, amplitude
    and standard deviation. The resulting field is made into a 3D volume 
    and then reduced to a 2D slice in the middle.

    Parameters:
    file: str, optional
        path to JSON file containing sphere parameters.

    Returns:
    X: np.ndarray 
        3D meshgrid array of coordinates with shape (dim, dim, dim)
    Y: np.ndarray 
        3D meshgrid array of coordinates with shape (dim, dim, dim)
    Z: np.ndarray 
        3D meshgrid array of coordinates with shape (dim, dim, dim)

    """
    spheres = []

    with open("universe_parameters.json", "r") as f:
        spheres = json.load(f)

    dim = 32 # for now

    x = np.linspace(0, 1, dim)
    y = np.linspace(0, 1, dim)
    z = np.linspace(0, 1, dim)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    box = np.zeros((dim, dim, dim)) # empty volume of N^3 dimensions
    N = len(spheres)

    for s in spheres:

        cx = s["cx"]
        cy = s["cy"]
        cz = s["cz"]
        sigma = s["sigma"]
        amp = s["amp"]

        blob = amp * np.exp(
            -(
                (X - cx)**2 +
                (Y - cy)**2 +
                (Z - cz)**2
            ) / (2 * sigma**2)
        )

        box += blob

    # slice box 
    middle_box = box.shape[0]//2
    box = box[middle_box, :, :]
    return X, Y, Z, box


# plot of tracer density field ===================

def plot_universe(X, Y, Z, box):
    """
    Visualize a density fueld using 3D volumetric rendering. 

    This function uses an interactive plotly volume plot. 


    Parameters:
    X : np.ndarray
        3D array of x-coordinates defining spatial grid.

    Y : np.ndarray
        3D array of y-coordinates defining spatial grid.

    Z : np.ndarray
        3D array of z-coordinates defining spatial grid.

    box : np.ndarray
        3D scalar field representing the density values to
        visualize.

    Returns:
    None
        Displays interactive plotly volume rendering.

    """
    fig = plotly.Figure(data=plotly.Volume(
        x=X.flatten(), # flatten arrays into 1D lists
        y=Y.flatten(),
        z=Z.flatten(),
        value=box.flatten(),
        isomin=0,
        isomax=box.max(),
        opacity=0.08,  
        surface_count=60 
    ))

    fig.update_layout(template = "plotly_dark")
    fig.show()
    # fig.show(renderer="browser")


# overdensity field ==============
def overdensity_field(density_field):
    """
    Compute the overdensity field from a density field. 

    Parameters
    density_field : np.ndarray
        input density field 

    Returns
    np.ndarray
        overdensity field with mean value of approximately zero,
        positive values indicate overdense regions, while negative
        values indicate underdense regions
    """
    return (density_field/np.mean(density_field)) - 1

def gaussian_filter(box, R=3):
    """
    Apply a Gaussian smoothing filter to the density field.

    Parameters:
    box: np.ndarray
        input density field
    
    R: float, optional 
        standard deiation (sigma) of Gaussian kernel

    Returns:
    np.ndarray
        smoothed density field
    """
    return sci.gaussian_filter(box, sigma=R, mode='wrap')

# mark and marked density field =================
def marked_overdensity_field(tracer_field, smoothed_field,  delta_star=10, p=7):
    """
    Compute a marked overdensity field. A mark function is 
    applied to a smoothed overdensity field to reweight densities. 
    The marked field enhances/suppresses various structures. 

    Parameters:
    tracer_field: np.ndarray
        original overdensity field of tracers
    smoothed_field: np.ndarray
        smoothed overdensity field used to compute mark
    delta_star: float, optional
        dimensionless parameter
    p: float, optional
        dimensionless parameter 
    
    Returns:
    mark: np.ndarray
        the mark field
    marked_field: np.ndarray
        marked overdensity field
    """
    mark = ((1+delta_star)/(1+delta_star+smoothed_field))**p
    marked_field = (1+tracer_field)*mark-1
    return mark, marked_field

# plot heatmap ===================================
def plot_heatmap(density_field, title='density field'):
    """
    Display 2D heatmap of a density field.

    Parameters:
    density_field: np.ndarray
        2D array of field to visualize
    title: str, optional
        title of plot
    
    Returns: 
    None
        Displays the plot 
    """
    plt.imshow(density_field, cmap='plasma')
    plt.colorbar()
    plt.title(title)
    plt.show()


# code to generate random spheres
def load_random_universe(): 
    """
    Note: call this first, then call u.load_universe() in notebook.
    """
    spheres = []
    # generate spheres
    N=200 # fixed in all simulations
    for _ in range(N):
        sphere = {
            "cx": float(np.random.rand()),
            "cy": float(np.random.rand()),
            "cz": float(np.random.rand()),
            "sigma": float(np.random.uniform(0.01, 0.11)),
            "amp": float(np.random.uniform(0.003, 8))
        }
    
        spheres.append(sphere)
    
    # save to json
    with open("universe_parameters.json", "w") as f:
        json.dump(spheres, f, indent=4)


X, Y, Z, box = load_universe()
rho_x = box # tracer density field
rho_R = gaussian_filter(rho_x, R=3) # smoothed density tracer field

delta_x = overdensity_field(rho_x) # overdensity of tracer field
delta_R = overdensity_field(rho_R) # overdensity of downsampled tracer field
m, delta_M = marked_overdensity_field(delta_x, delta_R) # overdensity of marked field
