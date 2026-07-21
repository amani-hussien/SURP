import plotly.graph_objects as plotly
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sci
import json
import plotly.graph_objects as go 


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

