'''
Created on Nov 21, 2016

@author: khushsi
'''
from __future__ import division

from numpy import loadtxt, mean, std, ones, zeros
import numpy
import sys


class LinearRegression():
    def __init__(self,normalize=True):
        self.normalize=normalize
        
    
    def train(self,X,y,alpha=0.001):

        m = y.size
        y.shape = (m,1)
        Xnorm = numpy.concatenate([numpy.ones(shape=y.shape) , X],axis=1)
        theta = zeros(shape=(Xnorm.shape[1],1))
        self.theta = gradient_descent(Xnorm,y,theta,alpha)
            
        return

    def test(self,X,y,Xmean,Xstd):

        m = y.size
        y.shape = (m,1)
        Xt = X
        Xt =  self.normalizetest(Xt,Xmean,Xstd)  
        Xnorm = numpy.concatenate([numpy.ones(shape=y.shape) , Xt],axis=1)
        mse = objectiveFunc(Xnorm, y, self.theta)       
            
        return mse
        
    def normalizedata(self,X):
        normX = X
        meanv=[]
        stdv = []
        feat = X.shape[1]

        for i in range(feat):
            mnt = mean(X[:,i])
            stdt = std(X[:,i])
            meanv.append(mnt)
            stdv.append(stdt)
            normX[:,i] = (normX[:,i] - mnt)/stdt

        return normX,meanv,stdv
    
    def normalizetest(self,X,meanv,stdv):
        normX = X
        feat = X.shape[1]
        for i in range(feat):
            normX[:,i] = (normX[:,i] - meanv[i])/stdv[i]

        return normX
    
def readdata(csvinputfile):
    data = loadtxt(csvinputfile,delimiter=',')
    Xt =  data[:,0:-1]
    Yt = data[:,-1]
    return Xt,Yt

def gradient_descent(X,y,theta,alpha):
    
    ## Gradient Descent
    numFeatures = X.shape[1]
    m=y.size
    prevobj = objectiveFunc(X, y, theta)
    currobj = objectiveFunc(X, y, theta)
    
    loopi = True
    i=0
    while(loopi):    
        i=i+1
        
        yp = X.dot(theta)
        if i % 50 == 0 :
            print "iternation # ",
            print i
            print " MSE on train data ",
            print prevobj.item()
            print "Model param W ",
            print theta.T
        
        for w in range(0,numFeatures):
            Xw = X[:,w]
            Xw.shape=(m,1)
            diff  = (yp - y) * Xw  
            objfunc = diff.sum() / m
            theta[w] = theta[w] - (alpha * objfunc)
        currobj = objectiveFunc(X, y, theta)
        if(prevobj - currobj < 0.00001 ):
            loopi = False
        prevobj = currobj
        
    print "Final Model W ",
    print theta.T
    
    return theta
def objectiveFunc(X, y, theta):
    m = y.size
    yp = X.dot(theta)
    diff = (yp - y)
    J = (1.0 / (2 * m)) * diff.T.dot(diff)
    return J    
    
def part1():    
    csvinputtrain=  sys.argv[2]
    csvinputtest=  sys.argv[3]
    
    abc = LinearRegression()
    X,y = readdata(csvinputtrain)
    alpha=0.005
    Xnorm = X
    Xmean = zeros(Xnorm.shape[1])
    Xstd = ones(Xnorm.shape[1])

    if(abc.normalize):
        Xnorm,Xmean,Xstd =  abc.normalizedata(X)  
        
    a =abc.train(Xnorm, y, alpha)
    Xt,yt = readdata(csvinputtest)
    Xnormt= Xt
    error = abc.test(Xnormt, yt, Xmean, Xstd)
        
    print " MSE on test data : ",
    print error

def part2():    
    csvinputtrain=  sys.argv[2]
    csvinputtest=  sys.argv[3]
    
    abc = LinearRegression()
    X,y = readdata(csvinputtrain)
    alpha=0.005
    Xnorm = X
    Xmean = zeros(Xnorm.shape[1])
    Xstd = ones(Xnorm.shape[1])

    if(abc.normalize):
        Xnorm,Xmean,Xstd =  abc.normalizedata(X)  
        
    a =abc.train(Xnorm, y, alpha)
    Xt,yt = readdata(csvinputtest)
    Xnormt= Xt
    error = abc.test(Xnormt, yt, Xmean, Xstd)
        
    print " MSE on test data : ",
    print error

if __name__ == '__main__':
    program = sys.argv[1]
    
    if(sys.argv[1] == "multi"):
        part2()
    else:
        part1()
        
        

    