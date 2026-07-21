from nbodykit.lab import * 
from nbodykit import setup_logging, style 
from nbodykit.lab import ArrayMesh
import matplotlib.pyplot as plt
import marked_power_spectrum as u 
import numpy as np
import pynbody

def load_COLA(filepath, Nmesh, boxsize=512):
    """
    Returns ArrayMesh (overdensity+1)

    """
    f = pynbody.load(filepath)

    # catalog elements
    pos = np.array(f['pos'])
    mass = np.array(f['mass'])
    vel = np.array(f['vel'])

    data = {'Position': pos, 'Mass': mass}
    cat = ArrayCatalog(data)
    cat.attrs['BoxSize'] = [boxsize, boxsize, boxsize]
    cat.attrs['Nmesh'] = [Nmesh, Nmesh, Nmesh]
    cat.attrs['Omega_cdm'] = 0.2657 # omegam: 0.315- omegab: 0.0493 
    cat.attrs['Omega_b'] = 0.0493
    cat.attrs['Omega_lambda'] = 0.685
    cat.attrs['h'] = 0.674

    mesh = cat.to_mesh(window = 'tsc', Nmesh=Nmesh, compensated=True, interlaced=True, position='Position')
    
    print(f.properties)
    return mesh

def load_nbody(filepath, Nmesh, boxsize=512):
    """
    Returns ArrayMesh (overdensity+1)

    """
    f = pynbody.load(filepath)
    
    
    # convert kpc/h -> Mpc/h
    pos = np.array(f['pos']) / 1000.0
    # catalog elements
    mass = np.array(f['mass'])
    vel = np.array(f['vel'])

    data = {'Position': pos, 'Mass': mass}
    cat = ArrayCatalog(data)
    cat.attrs['BoxSize'] = [boxsize, boxsize, boxsize]
    cat.attrs['Nmesh'] = [Nmesh, Nmesh, Nmesh]
    cat.attrs['Omega_cdm'] = 0.2657 # omegam: 0.315- omegab: 0.0493 
    cat.attrs['Omega_b'] = 0.0493
    cat.attrs['Omega_lambda'] = 0.685
    cat.attrs['h'] = 0.674

    mesh = cat.to_mesh(window = 'tsc', Nmesh=Nmesh, compensated=True, interlaced=True, position='Position')
    # NOTE: tsc
    print(f.properties)
    return mesh

def mesh_to_ndarray(mesh):
    """
    Overdensity where mean=0.
    """
    return np.array(mesh.to_real_field())-1

def compute_power_spectra(first, second=None, dk=None, kmin=None):
    """
    first - mesh 
    second - mesh
    returns Pk which has 'k' and 'power'
    """

    kwargs = {"mode": "1d"} 

    if dk is not None:
        kwargs["dk"] = dk

    if kmin is not None:
        kwargs["kmin"] = kmin

    Pk = FFTPower(first, **kwargs).power 

    if second is None:
        # compute 1d P(k)
        return Pk
    else:
        Pk_mark = FFTPower(first=second, **kwargs).power 
        Pk_cross = FFTPower(first=second,**kwargs, second=first).power
        return Pk, Pk_mark, Pk_cross
        
def gaussian_filter(field, filtersize=10, boxsize=1, meshsize=1):
    sigma = filtersize * meshsize / boxsize
    return u.gaussian_filter(field, sigma)

def plot_spectra(Pk, Pk_marked, Pk_cross):
    """
    returns: automatter, automark, cross
    """

    # plot
    plt.plot(Pk['k'], 
           (Pk['power'].real)*(Pk['k'])) 
    plt.plot(Pk_marked['k'], 
            (Pk_marked['power'].real)*(Pk_marked['k']))
    plt.plot(Pk_cross['k'], 
            (Pk_cross['power'].real)*(Pk_cross['k']))
    plt.legend([r'overdensity $P_{\delta \delta}(k)$', r'marked overdensity $P_{mm}(k)$', r'cross-spectrum $P_{m \delta}(k)$'])
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$k*P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Comparing auto- and cross-spectra")
    plt.tight_layout()
    plt.show()



