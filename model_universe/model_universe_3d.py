import plotly.graph_objects as plotly
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sci
import json
import plotly.graph_objects as go 



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


# overdensity field ==========================
def overdensity_field(density_field):
    return (density_field/np.mean(density_field)) - 1

def gaussian_filter(box, R=3):
    return sci.gaussian_filter(box, sigma=R, mode='wrap') # theres a radius= also


# mark and marked spectrum ====================
def marked_overdensity_field(tracer_field, downsampled_field, delta_star=4, p=10): # aviles
    mark = ((1+delta_star)/(1+delta_star+downsampled_field))**p
    marked_field = (1+tracer_field)*mark-1
    return mark, marked_field


# plot volume ===================================
def plot_volume(data_used, title='density field'):
    fig = plotly.Figure(data=plotly.Volume
                    (x=X.flatten(), 
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=data_used.flatten(), 
                    isomin=data_used.min(), 
                    isomax=data_used.max(),
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
def plot_isosurface(data_used, title='density field'):
    fig = plotly.Figure(
        data=plotly.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),

            value=data_used.flatten(),

            isomin=data_used.min(),
            isomax=data_used.max(),

            surface_count=15,   # fewer = cleaner shells
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

