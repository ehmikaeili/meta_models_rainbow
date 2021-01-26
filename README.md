## Meta Modeling: Python Interface Between scikit-learn and FEniCS for Efficient Meta-Modelling
This program is written by Pierre Kerfriden and Ehsan Mikaeili for predicting the position of brain tumour in brain surgery. In the surgery, as surgeon makes incision and opens the skull, the brain tumour relocates under new boundary conditions. Predicting the position of tumour under different incision sizes is invaluable data for surgeons, which can be predicted by constructing metamodels.

In this program, a meta-model is constructed with two parameters of Young modulus and incision radius. The finite element simulations are carried out in FEniCS platform and the data training for the prediction is done by using scikit-learn machine learning library. From mechanical perspectives, the brain media is considered as hyperelastic and the tumour is presented by elasticity coefficients. 

### Required Libraries/platforms

* FEniCS 

* scikit-learn

* numpy

* scipy

* matplotlib
