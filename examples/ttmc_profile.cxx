/** \addtogroup examples 
  * @{ 
  * \defgroup ttmc ttmc 
  * @{ 
  * \brief Computes TTMC
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

template <typename dtype>
bool ttmc(int n, int ur, int vr, int wr, 
          double sp_frac, int op_mode, int run_dense, World & dw) {
  
  int lens[3] = {n, n, n};
  bool is_sparse = sp_frac < 1.;

  Tensor<dtype> T(3, is_sparse, lens, dw);
  
  int lens_uc[3] = {n, vr, wr};
  int lens_vc[3] = {ur, n, wr};
  int lens_wc[3] = {ur, vr, n};

  T.fill_sp_random(-1.,1.,sp_frac);

  Matrix<dtype> U(n, ur, dw);
  Matrix<dtype> V(n, vr, dw);
  Matrix<dtype> W(n, wr, dw);
  
  U.fill_random((dtype)0,(dtype)1);
  V.fill_random((dtype)0,(dtype)1);
  W.fill_random((dtype)0,(dtype)1);

  double norm1, norm2, norm3, norm3T;
  int64_t sz = T.get_tot_size(false);
  bool pass = false;
  double stime;
  double etime;

  if (run_dense == 0) {
    if (op_mode == 2) {
      Tensor<dtype> * wmlist[2] = {&U, &V};
      int wmodes[2] = {0,1};
      Tensor<dtype> WCk(3, false, lens_wc, dw);
      stime = MPI_Wtime();
      TTMC<dtype>(&T, &WCk, 2, wmodes, wmlist, false);
      etime = MPI_Wtime();
      if (dw.rank == 0) printf("sparse op_mode 2: %1.2lf\n", (etime - stime));
    }
    if (op_mode == 1) {
      Tensor<dtype>* vmlist[2] = {&U, &W};
      int vmodes[2] = {0, 2};
      Tensor<dtype> VCk(3, false, lens_vc, dw);
      stime = MPI_Wtime();
      TTMC<dtype>(&T, &VCk, 2, vmodes, vmlist, false);
      etime = MPI_Wtime();
      if (dw.rank == 0) printf("sparse op_mode 1: %1.2lf\n", (etime - stime));
    }
    if (op_mode == 0) {
      Tensor<dtype>* umlist[2] = {&V, &W};
      int umodes[2] = {1, 2};
      Tensor<dtype> UCk(3, false, lens_uc, dw);
      stime = MPI_Wtime();
      TTMC<dtype>(&T, &UCk, 2, umodes, umlist, false);
      etime = MPI_Wtime();
      if (dw.rank == 0) printf("sparse op_mode 0: %1.2lf\n", (etime - stime));
    }
    return true;
  }

  if (op_mode == 2) {
    Tensor<dtype> WC(3, false, lens_wc, dw);
    stime = MPI_Wtime();
    WC["rsk"] = T["ijk"] * U["ir"] * V["js"]; 
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("dense op_mode 2: %1.2lf\n", (etime - stime));
    Tensor<dtype> * wmlist[2] = {&U, &V};
    int wmodes[2] = {0,1};
    Tensor<dtype> WCk(3, false, lens_wc, dw);
    stime = MPI_Wtime();
    TTMC<dtype>(&T, &WCk, 2, wmodes, wmlist, false);
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("sparse op_mode 2: %1.2lf\n", (etime - stime));
    WCk["ijk"] -= WC["ijk"];
    WCk.norm2(norm1);
    pass = (norm1 / sz < 1.e-5);
  }

  if (op_mode == 1) {
    Tensor<dtype> VC(3, false, lens_vc, dw);
    stime = MPI_Wtime();
    VC["rjs"] = T["ijk"] * U["ir"] * W["ks"]; 
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("dense op_mode 1: %1.2lf\n", (etime - stime));
    Tensor<dtype> * vmlist[2] = {&U, &W};
    int vmodes[2] = {0,2};
    Tensor<dtype> VCk(3, false, lens_vc, dw);
    stime = MPI_Wtime();
    TTMC<dtype>(&T, &VCk, 2, vmodes, vmlist, false);
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("sparse op_mode 1: %1.2lf\n", (etime - stime));
    VCk["ijk"] -= VC["ijk"];
    VCk.norm2(norm2);
    pass = (norm2 / sz < 1.e-5);
  }

  if (op_mode == 0) { 
    Tensor<dtype> UC(3, false, lens_uc, dw);
    stime = MPI_Wtime();
    UC["irs"] = T["ijk"] * V["jr"] * W["ks"];
    //T.print();
    //V.print();
    //W.print();
    //UC.print();
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("dense op_mode 0: %1.2lf\n", (etime - stime));
    Tensor<dtype> * umlist[2] = {&V, &W};
    int umodes[2] = {1,2};
    Tensor<dtype> UCk(3, false, lens_uc, dw);
    stime = MPI_Wtime();
    TTMC<dtype>(&T, &UCk, 2, umodes, umlist, false);
    //printf("lens: %lld %lld %lld\n", UCk.lens[0], UCk.lens[1], UCk.lens[2]);
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("sparse op_mode 0: %1.2lf\n", (etime - stime));
    //UCk.print();
    
    /*
    Tensor<dtype> * umlistT[2] = {&W, &V};
    int umodesT[2] = {0,1};
    int lens_ucT[3] = {wr, vr, n};
    Tensor<dtype> UCkT(3, false, lens_ucT, dw);
    Tensor<dtype> UCkTR(3, false, lens_ucT, dw);
    stime = MPI_Wtime();
    Tensor<dtype> TT(3, is_sparse, lens, dw);
    TT["kji"] = T["ijk"];
    TTMC<dtype>(&TT, &UCkT, 2, umodesT, umlistT, false);
    //printf("lens: %lld %lld %lld\n", UCkT.lens[0], UCkT.lens[1], UCkT.lens[2]);
    //UCkT["kji"] = UCkT["ijk"];
    //UCkTR["ijk"] = UCkT["kji"];
    //printf("lens: %lld %lld %lld\n", UCkTR.lens[0], UCkTR.lens[1], UCkTR.lens[2]);
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("sparse op_mode with transpose 0: %1.2lf\n", (etime - stime));
    //UCkT.print();
    //UC.print();
    */


    UCk["ijk"] -= UC["ijk"];
    //UCkT["ijk"] -= UC["ijk"];
    UCk.norm2(norm3);
    //UCkT.norm2(norm3T);
    //pass = (norm3 / sz < 1.e-5) && (norm3T / sz < 1.e-5);
    pass = (norm3 / sz < 1.e-5);
  }

  if (dw.rank == 0){
    if (!pass)
      printf("Failed TTMC tests for op_mode: %d.\n", op_mode);
    else
      printf("Passed TTMC tests for op_mode: %d.\n", op_mode);
  }
  return pass;
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
  int op_mode;
  int run_dense;
  double sp;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 4;
  } else n = 4;
  
  if (getCmdOption(input_str, input_str+in_num, "-op_mode")){
    op_mode = atoi(getCmdOption(input_str, input_str+in_num, "-op_mode"));
    if (op_mode < 0) op_mode = 0;
  } else op_mode = 0;

  if (getCmdOption(input_str, input_str+in_num, "-run_dense")){
    run_dense = atoi(getCmdOption(input_str, input_str+in_num, "-run_dense"));
    if (run_dense < 0) run_dense = 0;
  } else run_dense = 0;
  
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
    if (sp < 0.0 || sp > 1.0) sp = 1.;
  } else sp = 1.;

  {
    World dw;
    if (dw.rank == 0){
      printf("Running sparse (%lf fraction zeros) TTMC on order 3 tensor with dimension %d\n", sp, n);
    }

    bool pass;
    pass = ttmc<double>(n, ur, vr, wr, sp, op_mode, run_dense, dw);
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
