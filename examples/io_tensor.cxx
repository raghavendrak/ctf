#include <ctf.hpp>
#include <chrono>
#include <float.h>
using namespace CTF;

void spmv(int nIter, int warmup, std::string filename, World& dw) {
  int x = 1102824;
  int y = x;
  int lens[] = {x, y};
  Tensor<double> B(2, true /* is_sparse */, lens, dw);
  Vector<double> a(x, dw);
  Vector<double> c(y, dw);

  auto compute = [&]() {
    a["i"] = B["ij"] * c["j"];
  };

  // Attempt to read in B...
  B.read_sparse_from_file(filename.c_str());
  std::cout << B.nnz_tot << std::endl;

  for (int i = 0; i < warmup; i++) { compute(); }
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < nIter; i++) { compute(); }
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  if (dw.rank == 0) {
    std::cout << "Average execution time: " << (double(ms) / double(nIter)) << "ms." << std::endl;
  }
}

int main(int argc, char** argv) {
  int nIter = -1, warmup = -1;
  std::string filename;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0) {
      nIter = atoi(argv[++i]);
      continue;
    }
    if (strcmp(argv[i], "-warmup") == 0) {
      warmup = atoi(argv[++i]);
      continue;
    }
    if (strcmp(argv[i], "-tensor") == 0) {
      filename = std::string(argv[++i]);
      continue;
    }
  }

  if (nIter == -1 || warmup == -1 || filename.empty()) {
    std::cout << "provide all inputs." << std::endl;
    return -1;
  }

  MPI_Init(&argc, &argv);
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  {
    World dw;
    spmv(nIter, warmup, filename, dw);
  }
  MPI_Finalize();
  return 0;
}
