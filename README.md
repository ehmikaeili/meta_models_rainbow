## Meta Modeling: Python Interface Between scikit-learn and FEniCS for Efficient Meta-Modelling
This program is written for predicting the position of brain tumour in brain surgery. In the surgery, as surgeon makes incision and opens the skull, the brain tumour relocates under new boundary conditions. Predicting the position of tumor under different incision sizes is invaluable data for surgeons, which can be predicted by constructing metamodels. In this program, we construct a meta-model with two parameters of Young modulus and incision radius. We employ FEM to calculated the sample points and then use Gaussian Process Regression technique to interpolate the space between them. From mechanical perspectives, we consider the brain media as hyperelastic and the tumor is presented by elasticity coefficients. 

### Required Libraries/platforms

* FEniCS 

* scikit-learn

* numpy

* scipy

* matplotlib
