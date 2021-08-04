import numpy as np
import os

import OpenMORe.model_order_reduction as model_order_reduction
import OpenMORe.utilities as utilities_
from OpenMORe.utilities import *
import matplotlib.pyplot as plt

############################################################################
# In this example it's shown how to perform dimensionality reduction and 
# feature extraction on a matrix X (turbo2D.csv) via Local Principal Component 
# Analysis (LPCA).
############################################################################

# Dictionary to load the input matrix, found in .csv format
file_options = {
    "path_to_file"                  :   os.path.abspath(os.path.join(__file__ ,"/hpcwork/itv/jh238254/Databases/WIP")),
    "input_file_name"               :   "turbo2D.csv",
}

# Dictionary with the instruction for the LPCA algorithm:
settings ={
    #centering and scaling options
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    #set the final dimensionality
    "number_of_eigenvectors"    : 7,
    
    #enable to plot the cumulative explained variance
    "enable_plot_variance"      : True,
    
    #set the number of the variable whose reconstruction must be plotted
    "variable_to_plot"          : 1,
}

# Load the input matrix 
X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
idx = np.genfromtxt('/Users/bastiancocco/Documents/Thesis/OpenMORe/data/reactive_flow/idx.txt', delimiter= ',')
X = utilities_.get_cluster(X, idx, index = 1, write = True)



model = model_order_reduction.PCA(X, settings)


# Perform the dimensionality reduction via Principal Component Analysis,
# and return the eigenvectors of the reduced manifold 
PCs = model.fit()


# Compute the projection of the original points on the reduced
# PCA manifold, obtaining the scores matrix Z
Z = model.get_scores()


# Assess the percentage of explained variance if the number of PCs has not
# been set automatically, and plot the result
model.get_explained()


# Reconstruct the matrix from the reduced PCA manifold
X_recovered = model.recover()


# Compare the reconstructed chosen variable "set_num_to_plot" with the
# original one, by means of a parity plot
model.plot_parity()
model.plot_PCs()