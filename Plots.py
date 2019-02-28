import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def coupling(J_orig, J_fit):
     
    """
    It plots the Original and Fit coupling matrices and their difference
    """
    
    J_diff = (J_orig - J_fit)
    minJ = np.min([J_fit.min(), J_orig.min(), J_diff.min()])
    maxJ = np.max([J_fit.max(), J_orig.max(), J_diff.max()])
    fig, axes = plt.subplots(nrows=1, ncols=3)
    plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=2.0)


    ax = axes[0]
    ax.set_yticks([])
    ax.set_xticks([10, 20, 30, 40])
    im = ax.imshow(J_orig, vmin=minJ, vmax=maxJ, interpolation='nearest')
    ax.set_title('Original couplings')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axes[1]
    ax.set_yticks([])
    ax.set_xticks([10, 20, 30, 40])
    im = ax.imshow(J_fit, vmin=minJ, vmax=maxJ, interpolation='nearest')
    ax.set_title('Fit couplings')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = axes[2]
    ax.set_yticks([])
    ax.set_xticks([10, 20, 30, 40])
    im = ax.imshow(J_diff, vmin=minJ, vmax=maxJ, interpolation='nearest')
    ax.set_title('Difference')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.show
    
    return

def covariance(J_orig, J_fit, samples_orig, samples_fit):

    """
    It compares the samples generated from the Original and Fit coupling matrices
    """
    
     
    cov_orig = np.cov(samples_orig, rowvar=0)
    np.fill_diagonal(cov_orig, 0.) 
    
    
    cov_fit = np.cov(samples_fit, rowvar=0)
    np.fill_diagonal(cov_fit, 0.) 
    
    minJ = np.min([J_fit.min(), J_orig.min()])
    maxJ = np.max([J_fit.max(), J_orig.max()])
    
    plt.figure()
    plt.scatter(J_orig.ravel(), J_fit.ravel())
    x = np.linspace(J_orig.min(), J_orig.max(), 1000)
    plt.plot(x, x, c='g')
    plt.xlabel('Original')
    plt.ylabel('Fit')
    plt.figure()
    plt.scatter(J_fit.ravel(), cov_orig.ravel())
    plt.plot(x, x, c='g')
    plt.xlabel('Fit')
    plt.ylabel('Covariance')
    plt.figure()
    x = np.linspace(minJ, maxJ, 1000)
    plt.plot(x, x, c='g')
    plt.scatter(cov_orig.ravel(), cov_fit.ravel(), c='r')
    plt.xlabel('Original Covariance')
    plt.ylabel('Fitted Covariance')
    
    plt.show()
