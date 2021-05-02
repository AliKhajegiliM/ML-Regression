# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 09:00:52 2021

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Modules needed for 3d surface plots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import from plot_functions
import plot_functions as pf

univariate_path = "weights.txt"
multivariate_path = "insurance.csv"
# loading the data
def load_data(file_path):
    X, y = None, None
    ########################################################################
    if (file_path==univariate_path):
        df = pd.read_csv(file_path, sep='\s+', engine='python')
        df.columns = ['index','brain','body']
        X1=df.drop('index',1)
        y=X1.drop('brain',1)
        X=X1.drop('body',1)
    elif file_path==multivariate_path:
        df = pd.read_csv(file_path)
        y=df['charges']
        X=df.drop('charges',1)
        X.smoker=(X.smoker=='yes')*1.0
        X.sex=2*(X.sex=='male')-1
        X.region=(X.region=='northeast')*1.0+(X.region=='southeast')*2.0+(X.region=='northwest')*3.0+(X.region=='southwest')*4.0
    ########################################################################
    return X,y

#X_uni, y_uni = load_data(univariate_path)
#X_mult, y_mult = load_data(multivariate_path)
#
class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.1, n_iter=15):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        ''' 
        n,d = X.shape
        self.JHist = []
        for i in range(self.n_iter):
            self.JHist.append((self.computeCost(X, y, theta), theta))
            ########################################################################1
            hyp=np.dot(X,theta)
            loss=y-hyp
            grad=np.dot(X.T,loss)/n
            theta=theta+self.alpha*grad
            
            print("Iteration: ", i+1, " Cost: ", float(self.JHist[i][0]), " Theta: ", theta.T)
            ########################################################################
        return theta

    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
        '''
        ########################################################################
        n,d=X.shape
        cost=np.dot((y-np.dot(X,theta)).T,(y-np.dot(X,theta)))/(2*n)
        return cost
        ########################################################################

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        ########################################################################
        self.theta=self.gradientDescent(X, y, self.theta)
        print('Theta Gradient Descent:', self.theta)
        ########################################################################

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        ########################################################################
        prediction=np.dot(X,self.theta)
        return prediction
        ########################################################################
#       
def linreg_test_univariate(file_path, alpha=0.1):
    # load the data
    X, y = load_data(file_path)
    n, d = X.shape
    print(n,d)
    X=X.to_numpy()
    X=(X-np.mean(X))/np.std(X)
    y=y.to_numpy()
    X.reshape(n,d)
    y.reshape(n,1)
    ########################################################################
    #Add a row of ones for the bias term to X 
    X=np.append(np.ones((n,1)),X,axis=1)
    ########################################################################
    
    # initialize the model
    init_theta = np.matrix(np.ones((d+1,1)))*10  # note that we really should be initializing this to be near zero, but starting it near [10,10] works better to visualize gradient descent for this particular problem
    n_iter = 1500

    # Instantiate objects
    lr_model = LinearRegression(init_theta=init_theta, alpha=alpha, n_iter=n_iter)
    pf.plotData1D(X[:,1],y)
    lr_model.fit(X,y)
    pf.plotRegLine1D(lr_model, X, y)

    # Visualize the objective function convex shape
    theta1_vals = np.linspace(-200, 800, 100)
    theta2_vals = np.linspace(500, 1200, 100)
    pf.visualizeObjective(lr_model,theta1_vals, theta2_vals, X, y)

    # Compute the closed form solution in one line of code
    theta_closed_form = 0  
    ########################################################################
    #replace "0" with closed form solution 
    theta_closed_form =np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    ########################################################################
    print("Theta Closed Form: ", theta_closed_form)
    
#    
def linreg_test_multivariate(file_path, alpha=0.1):
    # load the data
    X, y = load_data(file_path)
    n, d = X.shape
    X=X.to_numpy()
    y=y.to_numpy()
    y=y.reshape(n,1)
    ########################################################################
    #Normalize X by mean and standard deviation (3pts)
    X=(X-np.mean(X))/np.std(X)
    # Add a row of ones for the bias term to X (same as above)
    X=np.append(np.ones((n,1)),X,axis=1)
    ########################################################################
    
    # initialize the model
    init_theta = np.matrix(np.random.randn((d+1))).T
    #print(init_theta)
    n_iter = 2000
    # a good convergence occurs at iterations around 200000

    # Instantiate objects
    lr_model = LinearRegression(init_theta = init_theta, alpha = alpha, n_iter = n_iter)
    lr_model.fit(X, y)

    # Compute the closed form solution in one line of code
    theta_closed_form = 0  
    ########################################################################
    # replace "0" with closed form solution (same as above)
    theta_closed_form =np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    ########################################################################
    print("Theta Closed Form: ", theta_closed_form)

    




