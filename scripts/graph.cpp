#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <sstream>
#include <vector>
using namespace std;

int main(int argc, char** argv){
  ifstream fin;
  ofstream fout;
  string fname, fname2, line;
  vector <int> M, N, K, M2, N2, K2;
  vector <double> Time, MSE, PSNR, SSIM, Time2, MSE2, PSNR2, SSIM2;
  int m, n, k, i;
  double time, mse, psnr, ssim;

  if (argc != 3){
    printf("USAGE:\n  ./graph.cpp <input1.csv> <input2.csv>\n");
    exit(1);
  }

  fname = argv[1];
  fname2 = argv[2];

  /* Process the first file  */
  fin.open(fname);
  if (fin.is_open()) {
    // Preamble
    getline(fin, line);
    printf("\nProcessing %s\n\n", fname.c_str());
    while (getline(fin, line)){
      // parse the line and extract each number
      sscanf(line.c_str(), "%d, %d, %d, %lf, %lf, %lf, %lf", &m, &n, &k, &time, &mse, &psnr, &ssim);

      // push into vectors
      M.push_back(m);
      N.push_back(n);
      K.push_back(k);

      Time.push_back(time);
      MSE.push_back(mse);
      PSNR.push_back(psnr);
      SSIM.push_back(ssim);
    }
  }
  fin.close();


  /* Process the second file  */
  fin.open(fname2);
  if (fin.is_open()) {
    // Preamble
    getline(fin, line);
    printf("Processing %s\n", fname2.c_str());
    while (getline(fin, line)){
      // parse the line and extract each number
      sscanf(line.c_str(), "%d, %d, %d, %lf, %lf, %lf, %lf", &m, &n, &k, &time, &mse, &psnr, &ssim);

      // push into vectors
      M2.push_back(m);
      N2.push_back(n);
      K2.push_back(k);

      Time2.push_back(time);
      MSE2.push_back(mse);
      PSNR2.push_back(psnr);
      SSIM2.push_back(ssim);
    }
  }
  fin.close();

  // create output file
  fout.open("graph.jgr");
  if (fout.fail()) { printf("graph.jgr failed to open\n"); exit(1); }

  fout << "newgraph\n\nxaxis\n  min 15\n  max 101\n  hash 15\n  mhash 2\n  label : Image length (pixels)\n  size 5\n\nyaxis\n  min 0\n  max 100\n  size 4  label : PSNR\n\nnewcurve marktype none linetype solid color 1 0 0 label : MATLAB\n   pts ";

  for (i = 0; i < M.size(); i++) { 
    fout << M[i] << " " << PSNR[i] << " ";
  }
  fout << "\n\n";


  fout << "\nnewcurve marktype none linetype solid label : Python\n   pts ";
  for (i = 0; i < M2.size(); i++) { 
    fout << M2[i] << " " << PSNR2[i] << " ";
  }
  fout << "\n";


  fout.close();
}
