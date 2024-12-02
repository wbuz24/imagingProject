clc; clear all; close all;

% Parameters
M = 25; % Image size (MxM)
N = M^2; % Total number of pixels
K = round(N / 2); % Sparsity level (number of non-zero elements)
numMeasurements = 4 * round(K * log2(N/K)); % Theoretically Klog2(N/K) number of measurements for full recovery, increasing numMeasurements increases both algorithms accuracy significantly
eta = 0; % Noise level, can optionally set to 0

%% Generate measurements
% Load and prepare image
img = imread('../leaf.jpg');
img = rgb2gray(img);
img = imresize(img(:, :, 1), [M, M]);

%load('data_tissue.mat');
%img = im_data{1, 1};
%img = imresize(img, [M, M]);

originalImg = double(img);
originalImg = originalImg / max(originalImg(:));
% Apply 2D Discrete Cosine Transform (DCT)
imgDCT = dct2(originalImg);

figure;
imshow(originalImg);

% Generate measurement matrix (random Gaussian fits RIP)
A = randn(numMeasurements, M*M);

% Flatten DCT coefficients to 1D vector
dctVec = imgDCT(:);

% Get the indices of the largest K coefficients
[~, sorted_indices] = sort(abs(dctVec), 'descend');  % Sort coefficients by magnitude

% Zero out all but the largest K coefficients
sparseDCT = zeros(size(dctVec));
sparseDCT(sorted_indices(1:K)) = dctVec(sorted_indices(1:K));  % Keep only the largest K

% Reshape the sparse DCT coefficients back into MxM matrix
sparseDCTMatrix = reshape(sparseDCT, [M, M]);

% Apply inverse DCT to recover the sparse-limited image
imgSparseLimited = idct2(sparseDCTMatrix);

% Convert the image to double for comparison
imgSparseLimited = double(imgSparseLimited);

% Recreate the compressed signal using the sparse DCT coefficients
y = A * sparseDCT;

% Add noise to signal
etaNorm = std(y) * eta; % Normalize noise to signal stanard deviation
y = y + etaNorm * randn(size(y));

% Reconstruct noisy image for display
imgNoisyDCT = A \ y;
imgNoisyDCT = reshape(imgNoisyDCT,[M,M]);
imgNoisy = idct2(imgNoisyDCT);

%% Reconstruction Algorithm

% residual_norm = norm(A' * y, 'inf');
% alpha = 0.1;
% lambda = alpha * residual_norm / sqrt(K);
lambda = 0.01;

max_iter = 1000; % Maximum number of iterations
tol = 1e-6; % Tolerance for convergence

% Start timing
tic;

% Parallel Computing Toolbox (faster if image > 50x50)
%A = gpuArray(A);
%y = gpuArray(y);

% Reconstruct the sparse signal using FISTA
imgRecon = fista(A, y, lambda, max_iter, tol);

% Stop timing and display elapsed time
elapsedTime = toc;
disp(['FISTA runtime: ', num2str(elapsedTime), ' seconds']);

% Continue with the rest of the reconstruction
imgReconReshape = reshape(imgRecon, [M, M]);
imgFISTA = idct2(imgReconReshape);

%% Error Calculation
[MSE_FISTA, PSNR_FISTA] = errorCalc(originalImg, imgFISTA);

% Display
disp(['N: ',num2str(N)]);
disp(['K: ', num2str(K)]);
disp(['Eta: ', num2str(eta)]);
disp(['OMP Mean Squared Error: ', num2str(MSE_FISTA)]);
disp(['OMP Peak Signal-to-Noise Ratio: ', num2str(PSNR_FISTA), ' dB']);


%% Display Results
fs = 14; % Font Size for MSE and PSNR
fc = 'white'; % Font color for MSE and PSNR
figure;
sgtitle(['Image Size: ', num2str(M),'x',num2str(M),', Sparsity Number:', num2str(K), ', Ratio K/N: ', num2str(K/(M^2)),', Noise Level (eta): ', num2str(eta)]);

% Original Image
subplot(1,4,1);
imshow(originalImg);
title('Original Image')

% Sparse image
subplot(1, 4, 2);
imshow((imgSparseLimited));
title('Sparse Image Pre-Reconstruction');

% Noisy image
% subplot(1, 4, 2);
% imshow(uint8(imgNoisy));
% title('Noisy Image (Post DCT)');

% FISTA reconstructed image
subplot(1, 4, 3);
imshow((imgFISTA));
title('Reconstructed Image using FISTA');
text(0.02, 0.02, ['MSE: ', num2str(MSE_FISTA)], 'Units', 'normalized', 'Color', fc, 'FontSize', fs, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
text(0.02, 0.08, ['PSNR: ', num2str(PSNR_FISTA), ' dB'], 'Units', 'normalized', 'Color', fc, 'FontSize', fs, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

%%

% figure;
% subplot(1,2,1)
% imagesc(imgFISTA);
% hold on
% subplot(1,2,2)
% imagesc(originalImg);
