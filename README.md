eeg_microstates.py
Python 2.7.x module containing functions to perform EEG microstate decomposition and information-theoretic analysis.

The package contains:
- basic '.edf' file format reader (http://www.edfplus.info/specs/edf.html)
- basic EEG filtering (6-th order Butterworth band-pass digital filter)
- microstate clustering
  - global field power and local maxima
  - modified K-means clustering
  - microstate sequence computation (competitive back-fitting)
- microstate analysis
  - symbol distribution and transition matrix
  - entropy
  - Markov tests for order 0, 1, 2 (cf. Kullback, Technometrics, 1962)
  - stationarity tests
  - symmetry test for the transition matrix
  - the autoinformation function (time-lagged mutual information)
  - Markov surrogates and confidence intervals

Author: Frederic von Wegner, 05/2017, fvwegner*at*gmail.com
