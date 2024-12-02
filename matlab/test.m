% Test script to run testFISTA.m a bunch of times

clc; close all; clear all;

fid = fopen('matlab_runtime.csv','w+');
fprintf(fid, "M:, N:, K:, FISTA runtime (seconds):, MSE: , PSNR: , SSIM: \n");
for M = 15:100
  %fprintf("M = %d", M);
  [N, K, elapsedTime, MSE_FISTA, PSNR_FISTA, SSIM] = testFISTA(M);
  fprintf("M: %d,  N: %d,  K: %d,  FISTA runtime: %.2f\n", M, N, K, elapsedTime);
  fprintf(fid, "%d, %d, %d, %.2f, %.5f, %.5f, %.5f\n", M, N, K, elapsedTime, MSE_FISTA, PSNR_FISTA, SSIM);
end