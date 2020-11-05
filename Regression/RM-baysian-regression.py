#!/usr/bin/env python
# coding: utf-8

# ### Bayesian linear regression on CPU data
# 
#  * #### Aims
#      - To implement Bayesian inference over the parameters of the linear model for the CPU data.
#      - Practise model selection with marginal likelihood

# #### Task 1:  Bayesian treatment of the CPU regression problem
# In this task, we will perform a Bayesian treatment of the CPU regression problem. 
# 
# #####  We start by loading the data and rescaling it to aid with numerics.

# In[1]:


import numpy as np
import pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
Y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1][:,None]

MYCT = X_train[:,0][:,None]
MMIN = X_train[:,1][:,None]
MMAX = X_train[:,2][:,None]
CACH = X_train[:,3][:,None]
CHMIN = X_train[:,4][:,None]
CHMAX = X_train[:,5][:,None]


# ##### Task 1.1: Step up prior, $p(\mathbf{w})$
# 
# We'll define a Gaussian prior over $\mathbf{w}$, with mean $\mathbf{0}$ and covariance $\left[\begin{array}{cc} 100& 0\\0 & 5\end{array}\right]$. We'll also fix $\sigma^2=2$.

# In[2]:


prior_mean = np.zeros((6,1)) # vector of mean 
prior_cov = np.array([[100,8,100,6,100,9],[4,5,5,5,6,5],[101,32,101,21,101,19],[9,10,1,10,100,10],[102,0,102,0,102,0],[0,20,0,20,100,10]]) # covariance matrix
print(prior_cov.shape)
print(np.linalg.det(prior_cov))
sig_sq = 2 # variance of the additive noise 


# ##### Task 1.2: Let's see what this prior means by sampling some $\mathbf{w}$ vectors from it and plotting the models with all features
# Use `numpy`'s `multivariate_normal` to generate samples from a multivariate Gaussian 
# https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.multivariate_normal.html

# In[3]:


w_samp = np.random.multivariate_normal(prior_mean.flatten(),prior_cov,10) # sampling from multivariate Gaussian distribution 
plt.figure()
plt.plot(X_train,Y_train,'ro') # plot data 
plot_x = X_test # generate new x for plotting the sampled model, e.g. you need construct design matrix for any new x
plt.plot(np.dot(w_samp,plot_x.T).T)


# ##### Task 1.4: Compute the posterior and draw samples from it
# 
# First let's write functions to construct polynomial design matrix, and to compute posterior mean and covariance 
# $$\mathbf{\Sigma} = \left(\frac{1}{\sigma^2}\mathbf{X}^{T}\mathbf{X} + \mathbf{S}^{-1}\right)^{-1}$$
# $$\boldsymbol\mu = \frac{1}{\sigma^2}\mathbf{\Sigma}\mathbf{X}^{T}\mathbf{t}$$
# where $\mathbf{S}$ is the covariance matrix of the prior $p(\mathbf{w})$

# In[4]:


def compute_post_cov(X, prior_cov, sig_sq): # your own function to compute posterior mean
    return(np.linalg.inv((1.0/sig_sq)*np.dot(X.T,X) + np.linalg.inv(prior_cov)) )
 
def compute_post_mean(post_cov, sig_sq, X, t): # your own function to compute posterior covariance
    return(np.dot(post_cov,(1.0/sig_sq)*np.dot(X.T,t) ) )


# In[5]:


X = X_train # construct design matrix
post_cov = compute_post_cov(X, prior_cov, sig_sq) # compute posterior mean 
post_mean = compute_post_mean(post_cov,sig_sq,X,Y_train) # compute posterior covariance 

plt.figure()
plt.title('data and posterior samples') 
w_samp = np.random.multivariate_normal(post_mean.flatten(),post_cov,1) # draw some samples
plt.plot(X_train,Y_train,'ro')
plt.plot(np.dot(w_samp,X_test.T).T,'b') # plot the sampled lines, 
                                               # only need 2 points to plot a straight line
plt.plot(np.dot(X_test,post_mean),'k',linewidth=3) # plot the posterior mean prediction, 
                                                    # only need 2 points to plot a straight line 


# #### Task 2: We'll now look at predictions
# 
# ##### Task 2.1: Functions for posterior prediction
# 
# $$p(t_{new} | \mathbf{X},\mathbf{t},\mathbf{x}_{new},\sigma^2) = {\cal N}(\mathbf{x}_{new}^{T}\boldsymbol\mu,\sigma^2 + \mathbf{x}_{new}^{T}  \mathbf{\Sigma} \mathbf{x}_{new})$$

# In[6]:


testX = X_test # generate some test data

pred_mean = np.dot(testX, post_mean) # compute predictive mean
pred_var = sig_sq + np.diag(np.dot(testX,np.dot((post_cov),testX.T))) # compute predictive variance 


# #### Task 2.2: Plot error bars 

# In[7]:


plt.plot(X_train,Y_train,'ro') # plot data 
plt.plot(X_test,pred_mean,'b') # plot mean prediction 
plt.errorbar(X_test[:,0][:,None].flatten(),pred_mean.flatten(),yerr=pred_var.flatten()) # plot error bars 


# In[8]:


w_samp = np.random.multivariate_normal(post_mean.flatten(),post_cov,1) 
Y_predict = np.dot(testX,w_samp.T)
print(Y_predict.shape)


# In[9]:


test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = Y_predict[:,0]
print(y_pred_pp)
np.savetxt('my_submission_b.csv', y_pred_pp, fmt='%d', delimiter=",",header=test_header, comments="")


# In[ ]:




