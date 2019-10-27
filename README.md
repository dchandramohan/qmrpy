# qmrpy
A package for quantitative MRI in python

##Intro
Quantitative MRI analysis involves fitting models of the MRI signal from experiments that systematically vary image acquisition parameters to estimate intrinsic physical magnetic properties of the object being imaged that serve as quantitative physiological biomarkers.

##Methods
Python scripts implented multiple MR signal models including T2, T2 star, and T1 signal equations. Funcitons were written to include required inputs for each individual model including parameters such as: array of TE values, TR, and flip angle. The least squares fitting function from the numpy library was used to fit for unknown values including T2* frequency shift, phase shift, T1 and signal magnitude.

##Results and Discusion
Successfuly implented the T1-weighted model and the T2* complex model in pythona and generate results on a sample data set of phantom data. Future directions include using healthy volunteer datasets, distributing and distributing the toolbox to the wider MRI research community.


