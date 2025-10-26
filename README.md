# XG-RVFL: Towards Robust and Inversion-Free Randomized Neural Networks

This folder contains the MATLAB implementation of the paper:

Reference: Mushir Akhtar, A. Kumari, M. Sajid, A. Quadir, Mohd. Arshad, P. N. Suganthan, M. Tanveer (2025). “Towards Robust and Inversion-Free Randomized Neural Networks: The XG-RVFL Framework.” Revision submitted to Pattern Recognition, Elsevier.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
XG-RVFL introduces a robust, bounded, and inversion-free learning scheme for Randomized Neural Networks (RdNNs).
It replaces the standard least-squares objective with the fleXi-Guardian (XG) loss and employs a Nesterov Accelerated Gradient (NAG)-based solver, eliminating matrix inversion and improving robustness to noise and outliers.


_____________________________________________________________________

REPOSITORY STRUCTURE

Main.m    -> main experiment script (hyperparameter search + cross-validation)
XG_RVFL_Function.m -> model training and evaluation routine (implements Nesterov Accelerated Gradient optimization)
compute_dLdu_XG  -> helper function within XG_RVFL_Function.m that computes the derivative of the proposed XG loss
Evaluate.m     -> computes overall accuracy (%) for binary and multiclass tasks


________________________________________________________________________

EXPERIMENTAL SETUP

Software: MATLAB R2023a
Hardware: Intel Core i7-6700 CPU, 16 GB RAM, Windows 10
Evaluation: 5-fold cross-validation (80/20 split per fold)

Hyperparameter Grid:
C ∈ {10^-5, ..., 10^5}
N ∈ {3, 23, 43, ..., 203}
Activation ∈ {sigmoid, sine, tribas, radbas, tansig, ReLU}
a ∈ {-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2}
η ∈ {0.5, 1.0, 1.5, 2.0}

________________________________________________________________________

CONTACT

For questions , please contact:
Mushir Akhtar
Department of Mathematics, IIT Indore
Email: phd2101241004@iiti.ac.in
