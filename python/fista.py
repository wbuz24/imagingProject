# FISTA algorithm, implemented in matlab by Owen Meyers & in python by Will Buziak
# EENG 598B - Digital Imaging
# Professor: Dr. Yamuna Phal
# Entirely based on the FISTA paper by Amir Beck & Marc Teboulle

import pandas as pd
import numpy as np
from numpy import asarray
from matplotlib import pyplot as plt
from scipy.fftpack import fft, dct, idct
import cv2
import math
import time
import sys

# Error Calcs
def errorCalc(preImg, postImg):
  # Calculate MSE (Mean Square Error)
  MSE = np.mean((postImg - preImg) ** 2)

  mpv = 255
  PSNR = 10 * math.log(mpv*mpv / MSE, 10) 

  print(f"OMP Mean Squared Error: %.2f\nOMP Peak Signal-to-Noise Ratio: %.2f" % (MSE, PSNR))

# Calculate time elapsed
def timeDiff(start, end):
  return end - start

# Soft thresholding
def soft_thresholding(x, threshold):
  buf = np.max(np.abs(x) - threshold)
  x = np.multiply(np.sign(x), buf)

  return x

# FISTA
def FISTA(A, y, lmda, max_iter, tol):
  # FISTA for L1-regularized least squares with constant stepsize
  # Algorithm described in https://www.ceremade.dauphine.fr/~carlier/FISTA 
  # Pg. 11
  #Args:
  #  A         : Sensing matrix
  #  y         : Observed measurements
  #  lambda    : Regularization parameter for sparsity
  #  max_iter  : Maximum number of iterations
  #  tol       : Convergence tolerance
  #Returns:
  #  x         : Reconstructed sparse signal
  [m, n] = A.shape
  x = np.zeros(n, dtype=np.uint8)
  z = x
  t = 1.0
  L = np.linalg.norm(A, 2)
  L = np.multiply(L, L)
  alpha = 1 / L
  alpha = float(alpha)
  i = 0

  print("Starting FISTA loop\n")
  for k in range(max_iter):
    # Save previous x for convergence check
    if (k % 100 == 0 and k > 0): 
      print("Iteration: ", i)
      print("\n")
    i = i + 1
    x_old = x

    # Gradient step (no need for lstsq, just direct multiplication)
    grad = np.matmul(np.transpose(A), (np.matmul(A, z) - y))
  #  grad = np.clip(grad, -1e10, 1e10)  # Clip the gradient to a reasonable range
    x = soft_thresholding(z - alpha * grad, lmda * alpha) 

    # Update momentum term
    t_new = (1 + math.sqrt(1 + 4 * t*t)) / 2;
    z = x + ((t - 1) / t_new * (np.subtract(x, x_old)))

    # Update t for next iteration
    t = t_new

    if (np.linalg.norm(np.subtract(x, x_old)) < tol):
      print("Convergence reached")
      break;

  return x


# read an image and return as a numpy array
def read_img(filename, M):
  img = cv2.imread(filename)

  # resize the image

  image = cv2.resize(img, dsize=(M, M), interpolation=cv2.INTER_CUBIC)
  image = image[:, :, 1]

  return image

# create a plot with a given title and show the image
def show_img(img, title):
  plt.title(title)
  plt.imshow(img)
  plt.show()

argc = len(sys.argv)

# parameters
if (argc == 2):
  M = int(sys.argv[1])
else:
  M = 50 

N = M*M
K = round(N / 4)
numMeasurements = 4 * round(K * math.log(N / K, 2)) # log_2(N / K)
eta = 0

# FISTA params
lmbda = .01
max_iter = 400
tol = 1e-6

## Generate Measurements
# Load & resize the image
image = read_img("leaf.jpg", M)
print("Image Read\n")

# apply 2D discrete cosine transform
imgDCT = dct(image, 2)

# Generate a random gaussian matrix
sizetup = (numMeasurements, N)
A = np.random.normal(0.0, 1.0, size=sizetup)

# Flatten DCT coefficients to 1D vector
dctvec = imgDCT.flatten()

# Argument sort the flattened frequency response
sorted_indices = np.argsort(abs(dctvec))
# Descending order
sorted_indices = np.flip(sorted_indices)

# create a zero mask the size of dctvec
print("Creating Sparse DCT matrix\n")
sparseDCT = np.zeros(dctvec.size, dtype=float)
for i in range(K):
  # Get indices of largest K coefficients
  sparseDCT[sorted_indices[i]] = dctvec[sorted_indices[i]] 

# Reshape back to a matrix
sparseDCTMatrix = np.reshape(sparseDCT, [M, M])

# Apply inverse DCT to recover the sparse-limited image
imgSparseLimited = idct(sparseDCTMatrix, 2)
imgSparseLimited = imgSparseLimited.astype(float)

# Recreate the compressed signal using the sparse DCT coefficients
y = np.matmul(A, sparseDCT)

# Add noise to signal
print("Adding noise\n")
etaNorm = np.std(y) * eta
y = y + etaNorm * np.random.randn(y.size) # Sizes are weird

imgNoisyDCT = np.linalg.lstsq(A, y)[0] 
print("Reshaping noisy image\n")
imgNoisyDCT = imgNoisyDCT.reshape((M, M))
imgNoisy = idct(imgNoisyDCT, 2)

## Reconstruction 
# Start timing
start_time = time.time()

# reconstruct the sparse signal using FISTA
imgRecon = FISTA(A, y, lmbda, max_iter, tol)

end_time = time.time()
timediff = timeDiff(start_time, end_time)

# Continue with the rest of reconstruction
imgReconReshape = imgRecon.reshape((M, M)) 
imgFISTA = dct(imgReconReshape, 2)

# Results
with open("runtime.txt", "a") as f:
  print(f"M: %d,  N: %d,  K: %d,  FISTA runtime: %.2f seconds" % (M, N, K, timediff), file=f)

# Error Calculations
#errorCalc(image, imgFISTA)

# Show image
show_img(image, "Original Image")
show_img(imgFISTA, "Reconstructed Image")
