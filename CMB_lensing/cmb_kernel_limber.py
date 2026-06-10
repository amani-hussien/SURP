import numpy as np
import matplotlib.pyplot as plt 
import camb 
import astropy.constants as constants
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad

# cosmology, interpolator and other global constants
params=camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, ns=0.965)
results=camb.get_results(params)
cosmology = FlatLambdaCDM(H0=params.H0, Om0=params.omegam)
h = params.H0/100
c = constants.c.to('km/s').value
z_star = results.get_derived_params()['zstar'] 
comoving_zstar = cosmology.comoving_distance(z_star).value
power_spec_calculator = camb.get_matter_power_interpolator(
    params,
    nonlinear=True,
    k_hunit=True, 
    kmax=10.0,
    zmax=z_star
) 

def cmb_radial_kernel(z, mu, eta):
    """
    CMB radial kernel as defined in Sankar's paper.
    """
    comoving_z = cosmology.comoving_distance(z).value
    constant_term = (3/2)*(params.omegam)*((params.H0)/(c))**2
    comoving_terms = (1+z)*comoving_z*(comoving_zstar-comoving_z)/(comoving_zstar)
    return (constant_term*comoving_terms*sigma(mu, eta))

def cmb_radial_kernel2(z, mu, eta):
    """
    CMB radial kernel as defined in Karim's paper, modified to include MG.
    """
    constant_term = (3/2)*(params.omegam)/c*(params.H0)**2/cosmology.H(z).value
    comoving_term = (1+z)*cosmology.comoving_distance(z).value*(1-cosmology.comoving_distance(z).value/comoving_zstar)
    return constant_term*comoving_term*sigma(mu, eta)


def limber_integral(ell, mu, eta):
    """
    As defined in Sankar.
    """
    def integrand(z):
        comoving_dist = cosmology.comoving_distance(z).value
        k = (ell+0.5)/comoving_dist*h
        W_CMB = cmb_radial_kernel2(z, mu, eta)
        Hz = cosmology.H(z).value
        P_mm = power_spec_calculator.P(z, k) 
        return (W_CMB**2)/(Hz*comoving_dist**2)*P_mm

    return c*quad(integrand, 0, z_star)[0]



def limber_integral2(ell, mu, eta):
    """
    As defined in Karim.
    """
    def integrand2(z):
        comoving_dist = cosmology.comoving_distance(z).value
        k = (ell+0.5)/comoving_dist*h
        W_CMB = cmb_radial_kernel2(z, mu, eta)
        Hz = cosmology.H(z).value
        P_mm = power_spec_calculator.P(z, k) 
        return (W_CMB**2*Hz)/(comoving_dist**2)*P_mm

    return (1/c)*quad(integrand2, 0, z_star)[0]


def sigma(mu, eta):
    return 0.5*mu*(1+eta)
