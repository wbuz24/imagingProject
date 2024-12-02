% Test script to run testFISTA.m a bunch of times

clc; close all; clear all;

for M = 15:100
  fprintf("M = %d", M);
  [N, K, elapsedTime, MSE_FISTA, PSNR_FISTA, SSIM] = testFISTA(M)
end