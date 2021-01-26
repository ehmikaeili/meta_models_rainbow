# Meta-modelling program by Pierre Kerfriden and Ehsan Mikaeili - Mines ParisTech and Cardiff University
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random

from scipy.optimize import fmin_bfgs, fmin
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

from brain_shift import brain_shift


class dataset_class:

    def __init__(self,FileName):
        self.FileName = FileName

    def SaveData(self):
        np.save(self.FileName + '_x.npy', self.x)
        np.save(self.FileName + '_y.npy', self.y)
        np.save(self.FileName + '_scales_x.npy', self.y)

    def ReadData(self):  
        print('./'+self.FileName + '_x.npy')  
        self.x = np.load(self.FileName + '_x.npy',allow_pickle=True)
        self.y = np.load(self.FileName + '_y.npy',allow_pickle=True)
        self.scales_x = np.load(self.FileName + '_scales_x.npy',allow_pickle=True)

    def AppendData(self,x,y):
        self.x = np.hstack((self.x,x))
        self.y = np.hstack((self.y,y))
        self.SaveData()

class sampler_class:

    def __init__(self,IO,dataset):
        self.dataset = dataset
        self.IO = IO
    
    def sample(self,type,N):
        
        if type == 'uniform':

            self.dataset.x = np.zeros((N,self.dataset.scales_x.shape[0]))

            for i in range(N):

                self.dataset.x[i,:] = np.random.uniform(0,1,self.dataset.scales_x.shape[0])
                x_scaled = np.copy(self.dataset.x[i,:])
                for j in range(self.dataset.scales_x.shape[0]):
                    x_scaled[j] = self.dataset.x[i,j] * (self.dataset.scales_x[j,1]-self.dataset.scales_x[j,0]) + self.dataset.scales_x[j,0]

                y = np.asarray(IO.eval(x_scaled,self.dataset.scales_x)).T
                if i==0:
                    self.dataset.y = y
                else:
                    self.dataset.y = np.vstack((self.dataset.y,y))


# non-parametric meta-model
class GPR_class:

    def __init__(self,x,y):

        self.x = x
        self.y = y

        self.gpr = np.ndarray((self.y.shape[1],),dtype=np.object)
        for i in range(self.y.shape[1]):

            dy = 0.
            Kernel = ConstantKernel() * RBF( length_scale = 0.1 , length_scale_bounds=[0.01 , 1] ) + WhiteKernel( noise_level=1.e-3 , noise_level_bounds=[1.e-8 , 1.0] )
            
            self.gpr[i] = GaussianProcessRegressor(kernel=Kernel, alpha=dy ** 2, random_state=100, n_restarts_optimizer=10).fit(self.x, self.y[:,i])
            print( 'self.gpr[i].kernel_ : ', self.gpr[i].kernel_ )

    def eval(self,x_star):

        vals = np.zeros((self.y.shape[1],1))
        vals_std = np.zeros((self.y.shape[1],1))
        for i in range(self.y.shape[1]):
            vals[i], vals_std[i] = self.gpr[i].predict(x_star.reshape(1, -1),return_std=True)
        return vals, vals_std
    
    def plot(self,NPoint,Output=0):

        if self.x.shape[1]==2:

            fig = plt.figure()
            xPlot = np.linspace(0,1, NPoint[0])
            yPlot = np.linspace(0,1, NPoint[1])
            X, Y = np.meshgrid(xPlot, yPlot)
            Z = np.copy(X)
            Z_plus = np.copy(X)
            Z_minus = np.copy(X)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    vals, vals_std = self.eval(np.array([X[i,j],Y[i,j]]))
                    Z[i,j] = vals[Output]
                    Z_plus[i,j] = Z[i,j] + 1.96*vals_std[Output]
                    Z_minus[i,j] = Z[i,j] - 1.96*vals_std[Output]

            ax = fig.gca(projection='3d')
            ax.plot_wireframe(X, Y, Z,color='black')
            ax.plot_wireframe(X, Y, Z_plus,color='red')
            ax.plot_wireframe(X, Y, Z_minus,color='blue')
            ax.scatter(self.x[:,0],self.x[:,1],self.y[:,Output],marker='o',c='black')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()


class IO_tumor:

    def __init__(self):
        # Requested outputs
        self.output_FEM = {'displacement_tumor_x': 0.0, 'displacement_tumor_y': 0.0, 'displacement_tumor_z': 0.0}

    def eval( self , parameters , bound):

        if (parameters.shape[0])>0 :
            self.elastic_modulus_brain = parameters[0]
            self.radius_incision = parameters[1]
            # Inputs from sample
            self.input_FEM = {'Elastic Modulus': parameters[0],'Incision Radius': parameters[1]}
            print(self.input_FEM)
            
            brain_shift( self )
            output = self.output_FEM
            print(output)

        return output.values()


print('----- metamodelling demo by P Kerfriden and E Mikaeili -----')

generate_new_data = 0

dataset = dataset_class('data')
IO = IO_tumor()

if generate_new_data:
    dataset.scales_x = np.array([[0. , 1.0],[0.5 , 1.5]])
    sampler = sampler_class(IO,dataset)
    sampler.sample('uniform',N=30)
else:
    dataset.ReadData()

print('dataset.x',dataset.x)
print('dataset.y',dataset.y)

GPR = GPR_class(dataset.x,dataset.y)
GPR.plot([15,15],1)
