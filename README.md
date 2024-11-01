# structure_landscapes
Investigating RNA Structural Landscapes using Long Read, Nanopore, Technology: A probabilistic, stacked, ensemble learning method for noisy processes.

###### Run Order
# Preprocess and Predictions
0. Server Fx include normalization for non-stationarity,  additional server code like GUPPY from ONT not included here
1. run_predict.py runs all M0-M4 sequentially
2. Predict.py combines predictions over mean, calculates probabilites, and incorporates base pairing probabilities

# RESULTS
Same process on unmodified data
0. Preprocess RNAeval.py on native structures from library, extracts native mfes
1. dendrogram removes poor reads, extracts clusters, sizes, centroids
1A. for control data run dendrogram_native to get cluster distances from native structure
1B. for control data run figures_tables to generate plots and tables.
2. RNAfold runs RNAfold on generated centroids and reactivities, extracts mfes

Cluster Images: generated by Cluster_Images.py

Additional empty directories for output data, BpProbabilities holds code for extracting max base pairing probabilities from intra/inter molecular pairings
