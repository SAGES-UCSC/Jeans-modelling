# Jeans-modelling

This python scipts performs dynamical modelling of galaxies using the spherical Jeans equations with constant orbital anisotropy. The method and the equations are given in `Mamon & Lokas (2005) <http://adsabs.harvard.edu/abs/2005MNRAS.363..705M>`_.

In order to use this script you need :

- the `emcee <https://github.com/dfm/emcee>`_ python module . 
- the file phot.py which defines some useful functions to be used in jeans.py

In order to run the script:

1) open jeans.py

2) input the text file containing the velocity dispersion data you want to fit

3) edit the photometric quantities of your tracer

4) the output is a text file that you can rename at the end of the file jeans.py

5) run the script with : python jeans.py
