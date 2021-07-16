'''
Example: load the two horizontal components of a record and computes the
directionality factor, RotD100/RotD50 response spectra and ratios, and the
directionality reponse spectrum.

luis.montejo@upr.edu
alan.rivera@upr.edu

Rivera-Figueroa, A., & Montejo, L.A. (2021). Spectral Matching RotD100 Target 
Spectra: Effect on records characteristics and seismic response. 
Earthquake Spectra (submitted for publication).

'''

import numpy as np
import matplotlib.pyplot as plt
from DirectionalitySpectra import dfactor,DFSpectra, load_PEERNGA_record

plt.close('all')

'''
use "load_PEERNGA_record" to load the 2 horizontal components:
s: acceleration series, dt: time step, n: number of data points, name: name of record
'''

comp1     = 'RSN730_SPITAK_GUK000.at2'   # horizontal comp1 [g]
comp2     = 'RSN730_SPITAK_GUK090.at2'   # horizontal comp2 [g]

s1,dt,n,name1 = load_PEERNGA_record(comp1)   
s2,dt,n,name2 = load_PEERNGA_record(comp2)

'''
use "dfactor" to calculate the directionality factor for the acceleration series
hr, htheta :  polar coordinates of the envelope (convex hull)
rmax       :  max response (radius)
thetamax   :  angle for rmax (deg)
df         :  directionality factor
'''

hr, htheta, rmax, thetamax, df = dfactor(s1,s2, plot=1)


'''
use "DFSpectra" to calculate
RotD100, RotD50, ratios RotD100/RotD50 and 
Directionality spectrum of accelerarion (DSA)
'''

dampratio = 0.05                         # damping ratio for spectra
theta = np.arange(0,180,1)               # vector with angles to evaluate rotated spectra
freqs = np.geomspace(0.1,1/(2*dt),100)   # frequencies vector
T = 1/freqs                              # periods vector

RotD100, RotD50, ratios, DSA = DFSpectra(T,s1,s2,dampratio,dt,theta, plot=1)







