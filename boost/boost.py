import pynbody
import pylab 
import numpy as np
from nbodykit.lab import * 
from nbodykit import setup_logging, style 
from nbodykit.lab import ArrayMesh, ArrayCatalog
import matplotlib.pyplot as plt
import marked_power_spectrum as u 
import simulation_analysis as s 

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
SIMS_DIR = BASE_DIR.parent / "sims"
RESULTS_DIR = BASE_DIR.parent / "results"

# could simplify with regex
def compute_boost_COLA(GR_filepath, MG_filepath, z, delta_star=4, p=10, R=10):
    """
    for one simulation, at some redshift.

    returns: boost for Pk, Pk_mark, Pk_cross
    """

    # load simulation
    GRmesh = s.load_COLA(GR_filepath, Nmesh=512)
    MGmesh = s.load_COLA(MG_filepath, Nmesh=512)

    Pk_GR, Pk_marked_GR, Pk_cross_GR = power_spectra_helper(GRmesh, delta_star, p, R)
    Pk_MG, Pk_marked_MG, Pk_cross_MG = power_spectra_helper(MGmesh, delta_star, p, R)

    kMG = Pk_MG['k']
    kGR = Pk_GR['k']

    assert np.allclose(kMG, kGR)

    boost_Pk = (Pk_MG['power'].real-Pk_MG.attrs['shotnoise'])/(Pk_GR['power'].real-Pk_GR.attrs['shotnoise'])
    boost_Pk_marked = (Pk_marked_MG['power'].real-Pk_marked_MG.attrs['shotnoise'])/(Pk_marked_GR['power'].real-Pk_marked_MG.attrs['shotnoise'])
    boost_Pk_cross = (Pk_cross_MG['power'].real-Pk_cross_MG.attrs['shotnoise']) /(Pk_cross_GR['power'].real-Pk_cross_GR.attrs['shotnoise'])

    # save boosts
    COLA_path= SIMS_DIR / f"boostCOLAz{z}_p{p}_dstar{delta_star}_R{R}.npz"
    np.savez(COLA_path, k = kMG, boostPkmark=boost_Pk_marked, boostPk=boost_Pk, boostPkcross = boost_Pk_cross, p=p, delta_star=delta_star, R=R)


    return kMG, boost_Pk, boost_Pk_marked, boost_Pk_cross

def compute_boost_Nbody(GR_filepath, MG_filepath, z, delta_star=4, p=10, R=10):
    """
    For one simulation, at some redshift.

    returns: boost for Pk, Pk_mark, Pk_cross
    """

    # load simulation
    GRmesh = s.load_nbody(GR_filepath, Nmesh=512)
    MGmesh = s.load_nbody(MG_filepath, Nmesh=512)

    Pk_GR, Pk_marked_GR, Pk_cross_GR = power_spectra_helper(GRmesh, delta_star, p, R)
    Pk_MG, Pk_marked_MG, Pk_cross_MG = power_spectra_helper(MGmesh, delta_star, p, R)

    kMG = Pk_MG['k']
    kGR = Pk_GR['k']

    # k
    assert np.allclose(kMG, kGR)

    # GR
    assert np.allclose(Pk_GR['k'], Pk_marked_GR['k'])
    assert np.allclose(Pk_GR['k'], Pk_cross_GR['k'])

    # MG
    assert np.allclose(Pk_MG['k'], Pk_marked_MG['k'])
    assert np.allclose(Pk_MG['k'], Pk_cross_MG['k'])

    # GR vs MG
    assert np.allclose(Pk_GR['k'], Pk_MG['k'])

    boost_Pk = (Pk_MG['power'].real-Pk_MG.attrs['shotnoise'])/(Pk_GR['power'].real-Pk_GR.attrs['shotnoise'])
    boost_Pk_marked = (Pk_marked_MG['power'].real-Pk_marked_MG.attrs['shotnoise'])/(Pk_marked_GR['power'].real-Pk_marked_MG.attrs['shotnoise'])
    boost_Pk_cross = (Pk_cross_MG['power'].real-Pk_cross_MG.attrs['shotnoise']) /(Pk_cross_GR['power'].real-Pk_cross_GR.attrs['shotnoise'])

    # save boosts
    Nbody_path= SIMS_DIR / f"boostNbodyz{z}_p{p}_dstar{delta_star}_R{R}.npz"
    np.savez(Nbody_path, k = kMG, boostPkmark=boost_Pk_marked, boostPk=boost_Pk, boostPkcross = boost_Pk_cross, p=p, delta_star=delta_star, R=R)

    return kMG, boost_Pk, boost_Pk_marked, boost_Pk_cross

def power_spectra_helper(mesh, delta_star, p, R):
    # compute marked field
    density_field = s.mesh_to_ndarray(mesh)
    smoothed_field = s.gaussian_filter(density_field, boxsize=512, meshsize=512, filtersize=R)
    _, marked_overdensity_field = u.marked_overdensity_field(density_field, smoothed_field, delta_star, p)

    # compute power spectra
    marked_mesh = ArrayMesh(marked_overdensity_field+1, BoxSize=512)
    Pk, Pk_marked, Pk_cross = s.compute_power_spectra(mesh, marked_mesh)
    return Pk, Pk_marked, Pk_cross

def boost_ratio(k, boost_cola, boost_nbody, interpolate=False, tolerance=1):
    ratio = boost_cola / boost_nbody

    if interpolate:
        good_mask = (ratio <= 1.01) & (ratio >= 0.99)
        ratio = np.interp(k, k[good_mask], ratio[good_mask])

    return k, ratio


def boost_accuracy(k, boost_cola, boost_nbody):
    percent_diff = 100 * np.abs(boost_cola / boost_nbody - 1)

    return k, percent_diff

def plot_boost_accuracy(k, percent_Pk, z, linestyle='solid'):
    fig, ax = plt.subplots()

    ax.semilogx(k, percent_Pk, ls = linestyle, label=r"$P_{\delta\delta}$")

    ax.axhline(1.0, color='k', ls='--', alpha=0.5, label='1%')

    ax.set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
    ax.set_ylabel("Percent difference (%)")
    ax.set_title(f"COLA vs N-body boost accuracy at z={z}")
    ax.legend()

    return fig, ax