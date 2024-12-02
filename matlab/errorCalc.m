function [MSE, PSNR, SSIM, SAM] = errorCalc(preImg, postImg)
%% Error Calculation
% Calculate MSE (Mean Squared Error)
MSE = sum((preImg(:) - postImg(:)).^2) / numel(preImg);

% Calculate PSNR (Peak Signal-to-Noise Ratio)
mpv = 255; % Assuming 8-bit grayscale images
PSNR = 10 * log10((mpv^2) / MSE);

% Calculate SSIM
SSIM = ssim(postImg, preImg);

% Calculate SAM
%SAM = sam(postImg, preImg);

end