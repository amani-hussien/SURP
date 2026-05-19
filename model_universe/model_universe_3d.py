import plotly.graph_objects as plotly
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sci
import json
import plotly.graph_objects as go 



# load universe parameters and build =======
def load_universe(file="universe_parameters.json"): 
    """
    Load a set of Gaussian spheres from a JSON file and generate a 3D
    density field. 

    Each sphere contributes a Gaussian blob defined by its center, amplitude
    and standard deviation.

    Parameters:
    file: str, optional
        path to JSON file containing sphere parameters

    Returns:
    X: np.ndarray 
        3D meshgrid array of coordinates with shape (dim, dim, dim)
    Y: np.ndarray 
        3D meshgrid array of coordinates with shape (dim, dim, dim)
    Z: np.ndarray 
        3D meshgrid array of coordinates with shape (dim, dim, dim)

    """
    spheres = []

    with open(file, "r") as f:
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
        x=X.flatten(), 
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


# overdensity field ==========================
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


# mark and marked spectrum ====================
def marked_overdensity_field(tracer_field, downsampled_field, delta_star=4, p=10): # aviles
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
    mark = ((1+delta_star)/(1+delta_star+downsampled_field))**p
    marked_field = (1+tracer_field)*mark-1
    return mark, marked_field


# plot volume ===================================
def plot_volume(density_field, title='density field'):
    """
    Render 3D volumetric visualization of scalar field.

    This function creates an interactive plotly volume.

    Parameters:
    density_field: np.ndarray
        3D scalar field
    title: str, optional
        title for plot
    
    Returns:
    None
        displays 3D volume plot
    """
    fig = plotly.Figure(data=plotly.Volume
                    (x=X.flatten(), 
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=density_field.flatten(), 
                    isomin=density_field.min(), 
                    isomax=density_field.max(),
                    opacity=0.08,
                    surface_count=80, 
                    colorscale='plasma',
                    caps=dict(x_show=False, y_show=False, z_show=False)
                    ))


    fig.update_layout(title=dict(text=title), 
                    template='plotly_dark', width=600, height=600,
                    scene=dict(
                    xaxis_title='X Axis',
                    yaxis_title='Y Axis',
                    zaxis_title='Z Axis',
                    aspectmode='cube'),
                    margin=dict(l=50, r=50, b=50, t=50))
    # fig.show()
    fig.show()


# plot isosurface ===================================
def plot_isosurface(density_field, title='density field'):
    """
    Render a 3D isosurface visualization of a scalar field. 

    This function creates an interactive plotly isosurface rendering.

    Parameters:
    density_field: np.ndarray
        3D scalar field
    title: str, optional
        title for plot
    
    Returns:
    None
        displays 3D isosurface plot
    """
    fig = plotly.Figure(
        data=plotly.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),

            value=density_field.flatten(),
            isomin=density_field.min(),
            isomax=density_field.max(),

            surface_count=15,  
            opacity=0.12,

            colorscale='plasma',

            caps=dict(
                x_show=False,
                y_show=False,
                z_show=False
            )
        )
    )

    fig.update_layout(
        title=dict(text=title),
        template='plotly_dark',
        width=600,
        height=600,

        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            aspectmode='cube'
        ),

        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50
        )
    )

    fig.show()


X, Y, Z, box = load_universe()
rho_x = box # tracer density field
rho_R = gaussian_filter(rho_x, R=3) # smoothed density tracer field

delta_x = overdensity_field(rho_x) # overdensity of tracer field
delta_R = overdensity_field(rho_R) # overdensity of downsampled tracer field
m, delta_M = marked_overdensity_field(delta_x, delta_R) # overdensity of marked field

