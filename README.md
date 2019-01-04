Python 2.7.x module containing functions to perform EEG microstate decomposition and information-theoretic analysis.
Based on the publications:  

[1] von Wegner F, Tagliazucchi E, Laufs H. Information-theoretical analysis of resting state EEG microstate sequences - non-Markovianity, non-stationarity and periodicities. NeuroImage 158 (2017) 99–111.

[2] von Wegner F, Laufs H. Information-theoretical analysis of EEG microstate sequences in Python. Front Neuroinformatics (2018) doi: 10.3389/fninf.2018.00030.

[3] von Wegner F, Knaut P, Laufs H. EEG microstate sequences from different clustering algorithms are information-theoretically invariant. Front Comp Neurosci (2018) doi: 10.3389/fncom.2018.00070.

The package contains:
- basic '.edf' file format reader (http://www.edfplus.info/specs/edf.html)
- basic EEG filtering (6-th order Butterworth band-pass digital filter)
- microstate clustering
  - global field power and local maxima
  - clustering algorithms: AAHC, modified k-means, k-medoids, PCA, Fast-ICA
  - microstate sequence computation (competitive back-fitting)
- microstate analysis
  - symbol distribution and transition matrix
  - Shannon entropy, entropy rate
  - Markov tests of order 0, 1, 2 (cf. Kullback, Technometrics, 1962)
  - transition matrix stationarity test
  - transition matrix symmetry test
  - the autoinformation function (time-lagged mutual information) of the microstate sequence
  - Markov surrogates and confidence intervals
- exemplary use of functions is contained in jupyter notebook format

Author: Frederic von Wegner, 05/2017, fvwegner*at*gmail.com
