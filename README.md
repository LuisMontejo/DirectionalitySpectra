# DirectionalitySpectra

This module contains python functions that allow the computation of response  directionality spectrum as described in Rivera-Figueroa and Montejo (2021).  

The functions included can also be used to load acceleration records in PEER/NGA format and compute RotD100/RotD50 response spectra.

*Rivera-Figueroa, A., & Montejo, L.A. (2021). Spectral Matching RotD100 Target Spectra: Effect on records characteristics and seismic response. Earthquake Spectra. https://doi.org/10.1177/87552930211049259

Functions:

    dfactor :               computes a single directionality factor 
    DFSpectra :             computes rotated response spectra and directionality response spectrum
    load_PEERNGA_record :   load record in .at2 format (PEER NGA Databases)   
