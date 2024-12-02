#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <sstream>
using namespace std;

int main(int argc, char** argv){
  ifstream fin;
  ofstream fout;
  string fname, line;
  int M, N, K;
  double time, mse, psnr, ssim;

  if (argc != 2){
    printf("USAGE:\n  ./graph.cpp <input.csv>\n");
    exit(1);
  }

  fname = argv[1];

  fin.open(fname);
  if (fin.is_open()) {
    // Preamble
    getline(fin, line);

    // create output file
    fout.open("graph.jgr");
    if (fout.fail()) { printf("graph.jgr failed to open\n"); exit(1); }

    fout << "newgraph\n\nxaxis\n  min 10\n  max 101\n  hash 15\n  mhash 2\n  label : Time (seconds)\n\nyaxis\n  min 0\n  max 301\n  size 2\n\nnewcurve marktype none linetype dotdotdash color 1 0 0\n   pts ";

    while (getline(fin, line)){
      // parse the line and extract each number
      sscanf(line.c_str(), "%d, %d, %d, %lf, %lf, %lf, %lf", &M, &N, &K, &time, &mse, &psnr, &ssim);

      fout << M << " " << time << " ";
    }
    fout << "\n";
  }

  fin.close();
  fout.close();
}
