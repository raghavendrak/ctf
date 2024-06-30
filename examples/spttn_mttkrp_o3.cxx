/** \addtogroup examples 
  * @{ 
  * \defgroup spttn_mttkrp_o3 spttn_mttkrp_o3 
  * @{ 
  * \brief Computes MTTKRP on an order-3 tensor
  */

#include <cstddef>
#include <ctf.hpp>
#include <float.h>
using namespace CTF;

template <typename dtype>
bool execute_spttn_kernel(int n, int ur, int vr, int wr, double sp_frac, World & dw) {
  int lens[3] = {n, n, n};
  bool is_sparse = sp_frac < 1.;
  bool mpass = true;
  double stime;
  double etime;
  {
    /*
    #mttkrp
    ijk,ja->iak iak,ka->ai
    for k:
      for j:
        for i:
          for a:
            buf[a] += T_ijk U_ai
        for a:
          R_ak += buf[a] V_aj
    */
    int lens[3], lens_uc[2], lens_vc[2], lens_wc[2];
    vr = wr = ur;
    lens[0] = n;
    lens[1] = n;
    lens[2] = n;
    lens_uc[0] = ur;
    lens_uc[1] = n;
    lens_vc[0] = vr;
    lens_vc[1] = n;
    lens_wc[0] = wr;
    lens_wc[1] = n;
    Tensor<dtype> T(3, is_sparse, lens, dw);
    T.fill_sp_random(-1., 1., sp_frac);
    Tensor<dtype> U(2, false, lens_uc, dw);
    Tensor<dtype> V(2, false, lens_vc, dw);
    U.fill_random((dtype)0,(dtype)1);
    V.fill_random((dtype)0,(dtype)1);

    Tensor<dtype> UC(2, false, lens_wc, dw);
    Tensor<dtype> * ops[3] = {&U, &V, &UC};
    stime = MPI_Wtime();
    spttn_kernel<dtype>(&T, ops, 3, "ijk,ai,aj->ak");
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("SpTTN-Cyclops MTTKRP implementation: %1.2lf\n", (etime - stime));

    Tensor<dtype> UCyy(2, false, lens_wc, dw);
    Tensor<dtype> * mlist4[3] = {&U, &V, &UCyy};
    int mode = 2;
    stime = MPI_Wtime();
    MTTKRP<dtype>(&T, mlist4, mode, true);
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("CTF MTTKRP multilinear implementation: %1.2lf\n", (etime - stime));
    
    Tensor<dtype> UCxx(2, false, lens_wc, dw);
    stime = MPI_Wtime();
    UCxx["ak"] = T["ijk"] * U["ai"] * V["aj"];
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("CTF MTTKRP: %1.2lf\n", (etime - stime));

    double norm; 
    UCyy["ij"] -= UC["ij"];
    UCyy.norm2(norm);
    int64_t sz = T.get_tot_size(false);
    bool pass = (norm / sz < 1.e-5);
    IASSERT(pass);
    if (dw.rank == 0){
      if (!pass)
        printf("Failed MTTKRP.\n");
      else
        printf("Passed MTTKRP.\n");
    }
    mpass = mpass & pass;
  }
  if (dw.rank == 0) {
    if (!mpass)
      printf("Failed contraction tests.\n");
    else
      printf("Passed all contraction tests.\n");
  }
  if (dw.rank == 0) printf("n: %d ur: %d\n\n-----------------------------\n\n\n", n, ur);
  return mpass;
} 


#ifndef TEST_SUITE
char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int n, ur, vr, wr;
  double sp;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 4;
  } else n = 4;

  if (getCmdOption(input_str, input_str+in_num, "-ur")){
    ur = atoi(getCmdOption(input_str, input_str+in_num, "-ur"));
    if (ur < 0) ur = 4;
  } else ur = 4;

  if (getCmdOption(input_str, input_str+in_num, "-vr")){
    vr = atoi(getCmdOption(input_str, input_str+in_num, "-vr"));
    if (vr < 0) vr = 4;
  } else vr = 4;

  if (getCmdOption(input_str, input_str+in_num, "-wr")){
    wr = atoi(getCmdOption(input_str, input_str+in_num, "-wr"));
    if (wr < 0) wr = 4;
  } else wr = 4;
  
  if (getCmdOption(input_str, input_str+in_num, "-sp")){
    sp = atof(getCmdOption(input_str, input_str+in_num, "-sp"));
    if (sp < 0.0 || sp > 1.0) sp = 0.2;
  } else sp = 0.2;

  {
    World dw;
    bool pass;
    pass = execute_spttn_kernel<double>(n, ur, vr, wr, sp, dw);
    assert(pass);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
