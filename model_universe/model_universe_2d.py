import plotly.graph_objects as plotly
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.ndimage as sci


# load universe parameters and build =======
def load_universe(): # edit for different JSON files
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
    fig = plotly.Figure(data=plotly.Volume(
        x=X.flatten(), # flatten arrays into 1D lists
        y=Y.flatten(),
        z=Z.flatten(),
        value=box.flatten(),
        isomin=0,
        isomax=box.max(),
        opacity=0.08,  # lower = more transparent visually
        surface_count=60 # more rendering
    ))

    fig.update_layout(template = "plotly_dark")
    fig.show()
    # fig.show(renderer="browser")


# overdensity field ==============
def overdensity_field(density_field):
    return (density_field/np.mean(density_field)) - 1

def gaussian_filter(box, R=3):
    return sci.gaussian_filter(box, sigma=R, mode='wrap') # theres a radius= also

# mark and marked density field =================
def marked_overdensity_field(tracer_field, downsampled_field,  delta_star=10, p=7):
    mark = ((1+delta_star)/(1+delta_star+downsampled_field))**p
    marked_field = (1+tracer_field)*mark-1
    return mark, marked_field

# plot heatmap ===================================
def plot_heatmap(density_field, title='density field'):
    plt.imshow(density_field, cmap='plasma')
    plt.colorbar()
    plt.title(title)
    plt.show()


X, Y, Z, box = load_universe()
rho_x = box # tracer density field
rho_R = gaussian_filter(rho_x, R=3) # smoothed density tracer field

delta_x = overdensity_field(rho_x) # overdensity of tracer field
delta_R = overdensity_field(rho_R) # overdensity of downsampled tracer field
m, delta_M = marked_overdensity_field(delta_x, delta_R) # overdensity of marked field

