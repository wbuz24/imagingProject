import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import scipy.io
import time
import sys

def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def fista(A, y, lambd, max_iter, tol):
    """
    FISTA for L1-regularized least squares with constant step size.
    Args:
        A         : Sensing matrix
        y         : Observed measurements
        lambd     : Regularization parameter for sparsity
        max_iter  : Maximum number of iterations
        tol       : Convergence tolerance
    Returns:
        x         : Reconstructed sparse signal
    """
    # Initialize variables
    _, n = A.shape
    x = np.zeros(n)       # Initial guess for the solution
    z = x.copy()          # Auxiliary variable
    t = 1.0               # Momentum parameter
    L = np.linalg.norm(A, ord=2)**2  # Estimate of Lipschitz constant
    alpha = 1 / L         # Step size for gradient descent
    iter_count = 0

    print(f"Lipschitz constant (L): {L}")
    
    for k in range(max_iter):
        iter_count += 1

        # Save previous x for convergence check
        x_old = x.copy()

        # Gradient step
        grad = A.T @ (A @ z - y)  # Compute gradient of the data term
        x = soft_thresholding(z - alpha * grad, lambd * alpha)

        # Update momentum term
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = x + ((t - 1) / t_new) * (x - x_old)

        # Update t for next iteration
        t = t_new

        # Stopping criterion (convergence check)
        if np.linalg.norm(x - x_old) < tol:
            break

    print(f"Converged in {iter_count} iterations.")
    return x


# def error_calc(original, reconstructed):
#     mse = np.mean((original - reconstructed) ** 2)
#     psnr = 20 * np.log10(255 / np.sqrt(mse))
#     SSIM = ssim(original, reconstructed)
#     return mse, psnr, SSIM
def error_calc(original, reconstructed):
    # Ensure images are in the correct range for OpenCV (0-255, uint8)
    original_uint8 = np.clip(original, 0, 255).astype(np.uint8)
    reconstructed_uint8 = np.clip(reconstructed, 0, 255).astype(np.uint8)

    # Mean Squared Error (MSE)
    mse = np.mean((original - reconstructed) ** 2)

    # Peak Signal-to-Noise Ratio (PSNR)
    psnr = 20 * np.log10(255 / np.sqrt(mse))

    # Structural Similarity Index Measure (SSIM)
    ssim_value = cv2.quality.QualitySSIM_compute(reconstructed_uint8, original_uint8)[0][0]

    return mse, psnr, ssim_value

# Parameters
argc = len(sys.argv)
if (argc == 2):
  M = int(sys.argv[1])
else:
  M = 50 
N = M**2  # Total number of pixels
K = round(N * 0.25)  # Sparsity level
num_measurements = 4 * round(K * np.log2(N / K))  # Number of measurements
eta = 0  # Noise level

# Load and prepare the image using OpenCV
img = cv2.imread('leaf.jpg', cv2.IMREAD_GRAYSCALE)  # Load as grayscale
#img = cv2.resize(img, (M, M))  # Resize to MxM
img = cv2.resize(img, (M, M), interpolation=cv2.INTER_CUBIC)
original_img = img.astype(np.float64)

# Apply 2D Discrete Cosine Transform (DCT)
img_dct = dct(dct(original_img.T, norm='ortho').T, norm='ortho')

# Generate measurement matrix (random Gaussian)
A = np.random.randn(num_measurements, M * M)

# data = scipy.io.loadmat('debug.mat')
# A = data['A']  # Measurement matrix
# y = data['y'].flatten()  # Compressed measurements
# img_dct = data['sparseDCT'].flatten()  # Sparse DCT coefficients
# Flatten DCT coefficients to 1D vector
dct_vec = img_dct.flatten()

# Get the indices of the largest K coefficients
sorted_indices = np.argsort(-np.abs(dct_vec))  # Descending order

# Zero out all but the largest K coefficients
sparse_dct = np.zeros_like(dct_vec)
sparse_dct[sorted_indices[:K]] = dct_vec[sorted_indices[:K]]

# Reshape the sparse DCT coefficients back into MxM matrix
sparse_dct_matrix = sparse_dct.reshape((M, M))

# Apply inverse DCT to recover the sparse-limited image
img_sparse_limited = idct(idct(sparse_dct_matrix.T, norm='ortho').T, norm='ortho')

# Generate compressed signal
y = A @ sparse_dct

# Add noise
eta_norm = np.std(y) * eta
y += eta_norm * np.random.randn(*y.shape)

# Save for debugging
np.savez('debug.npz', A=A, y=y, sparse_dct=sparse_dct)

# Reconstruction using FISTA
lambda_param = 0.01
max_iter = 1000
tol = 1e-12

# GPU acceleration (if available)
try:
    import cupy as cp
    A = cp.asarray(A)
    y = cp.asarray(y)
    fista = cp.fista  # Ensure compatibility with GPU
except ImportError:
    pass  # Fallback to CPU if GPU is unavailable

start_time = time.time()
# Perform reconstruction
reconstructed_dct = fista(A, y, lambda_param, max_iter, tol)

end_time = time.time()

if 'cp' in globals():
    reconstructed_dct = cp.asnumpy(reconstructed_dct)

# Reshape and apply inverse DCT to get reconstructed image
reconstructed_dct_matrix = reconstructed_dct.reshape((M, M))
img_fista = idct(idct(reconstructed_dct_matrix.T, norm='ortho').T, norm='ortho')

timediff = end_time - start_time

# Error Calculation
mse_fista, psnr_fista,ssim = error_calc(original_img, img_fista)

# Results
with open("runtime.txt", "a") as f:
  print(f"%d, %d, %d,  %.2f , %.5f, %.5f, %.5f" % (M, N, K, timediff, mse_fista, psnr_fista, ssim), file=f)

# Display results
fs = 14  # Font size
fc = 'white'  # Font color

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Image Size: {M}x{M}, Sparsity Number: {K}, Ratio K/N: {K/N:.2f}, Noise Level: {eta}')

# Original Image
axs[0].imshow(original_img, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

# Sparse Image
axs[1].imshow(img_sparse_limited, cmap='gray')
axs[1].set_title('Sparse Image Pre-Reconstruction')
axs[1].axis('off')

## FISTA Reconstructed Image
axs[2].imshow(img_fista, cmap='gray')
axs[2].set_title('Reconstructed Image using FISTA')
axs[2].text(0.02, 0.02, f'MSE: {mse_fista:.4f}', color=fc, fontsize=fs, transform=axs[2].transAxes)
axs[2].text(0.02, 0.08, f'PSNR: {psnr_fista:.2f} dB', color=fc, fontsize=fs, transform=axs[2].transAxes)
axs[2].text(0.02, 0.14, f'SSIM: {ssim:.4f}', color=fc, fontsize=fs, transform=axs[2].transAxes)
axs[2].axis('off')

plt.tight_layout()
plt.show()
