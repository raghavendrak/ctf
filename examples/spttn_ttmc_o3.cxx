/** \addtogroup examples 
  * @{ 
  * \defgroup spttn_ttmc_o3 spttn_ttmc_o3 
  * @{ 
  * \brief Computes TTMc on a 3rd order tensor
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

template <typename dtype>
bool execute_spttn_kernel(int n, int ur, int vr, int wr, double sp_frac, World & dw) {
  vr = wr = ur;
  int lens[3] = {n, n, n};
  bool is_sparse = sp_frac < 1.;
  bool mpass = true;
  double stime;
  double etime;

  {
    // TTMc: mode-(n-1)
    // order-3
    int lens_uc[2], lens_vc[2], lens_wc[3];
    int n1, n2, n3;
    n1 = n2 = n3 = n;
    lens_uc[0] = ur;
    lens_uc[1] = n1;
    lens_vc[0] = vr;
    lens_vc[1] = n2;
    lens_wc[0] = ur;
    lens_wc[1] = vr;
    lens_wc[2] = n3;
    Tensor<dtype> T(3, is_sparse, lens, dw);
    T.fill_sp_random(-1., 1., sp_frac);

    Matrix<dtype> U(lens_uc[0], lens_uc[1], dw);
    Matrix<dtype> V(lens_vc[0], lens_vc[1], dw);
    U.fill_random((dtype)0,(dtype)1);
    V.fill_random((dtype)0,(dtype)1);
    
    Tensor<dtype> UC(3, false, lens_wc, dw);
    Tensor<dtype> * ops[3] = {&U, &V, &UC};
    stime = MPI_Wtime();
    spttn_kernel<dtype>(&T, ops, 3, "ijk,ri,sj->rsk");
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("SPTTN-Cyclops TTMc (NOTE that it includes CSF construction time; please see total time to calculate printed above): %1.2lf\n", (etime - stime));
    /*
    mode-2
    for k:
      for j:
        for i:
          for r:
            T_kj[r] = T_ijk U_ir
        for s:
          for r:
            Z_rsk = T[r] V_js
    */
    Tensor<dtype> UCxx(3, false, lens_wc, dw);
    stime = MPI_Wtime();
    UCxx["rsk"] = T["ijk"] * U["ri"] * V["sj"];
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("CTF TTMc: %1.2lf\n", (etime - stime));

    double norm; 
    UCxx["ijk"] -= UC["ijk"];
    UCxx.norm2(norm);
    int64_t sz = T.get_tot_size(false);
    bool pass = (norm / sz < 1.e-5);
    if (dw.rank == 0){
      if (!pass)
        printf("Failed TTMc - 3 operands.\n");
      else
        printf("Passed TTMc - 3 operands.\n");
    }
    IASSERT(pass);
    mpass = mpass & pass;
  }
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
