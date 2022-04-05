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
          double sp_frac, World & dw) {
  
  int lens[3] = {n, n, n};
  bool is_sparse = sp_frac < 1.;

  Tensor<dtype> T(3, is_sparse, lens, dw);
  
  int lens_uc[3] = {n, vr, wr};
  Tensor<dtype> UC(3, false, lens_uc, dw);
  int lens_vc[3] = {ur, n, wr};
  Tensor<dtype> VC(3, false, lens_vc, dw);
  int lens_wc[3] = {ur, vr, n};
  Tensor<dtype> WC(3, false, lens_wc, dw);

  T.fill_sp_random(-1.,1.,sp_frac);

  Matrix<dtype> U(n, ur, dw);
  Matrix<dtype> V(n, vr, dw);
  Matrix<dtype> W(n, wr, dw);
  
  U.fill_random((dtype)0,(dtype)1);
  V.fill_random((dtype)0,(dtype)1);
  W.fill_random((dtype)0,(dtype)1);

  UC["irs"] = T["ijk"] * V["jr"] * W["ks"]; 
  VC["rjs"] = T["ijk"] * U["ir"] * W["ks"]; 
  WC["rsk"] = T["ijk"] * U["ir"] * V["js"]; 

  Tensor<dtype> * wmlist[2] = {&U, &V};
  int wmodes[2] = {0,1};
  //WC.print();
  Tensor<dtype> WCk(3, false, lens_wc, dw);
  TTMC<dtype>(&T, &WCk, 2, wmodes, wmlist, false);
  //WCk.print();

  Tensor<dtype> * vmlist[2] = {&U, &W};
  int vmodes[2] = {0,2};
  //VC.print();
  Tensor<dtype> VCk(3, false, lens_vc, dw);
  TTMC<dtype>(&T, &VCk, 2, vmodes, vmlist, false);
  //VCk.print();
  
  Tensor<dtype> * umlist[2] = {&V, &W};
  int umodes[2] = {1,2};
  //UC.print();
  Tensor<dtype> UCk(3, false, lens_uc, dw);
  TTMC<dtype>(&T, &UCk, 2, umodes, umlist, false);

  WCk["ijk"] -= WC["ijk"];
  VCk["ijk"] -= VC["ijk"];
  UCk["ijk"] -= UC["ijk"];
  double norm1, norm2, norm3;
  WCk.norm2(norm1);
  VCk.norm2(norm2);
  UCk.norm2(norm3);
  int64_t sz = T.get_tot_size(false);
  bool pass;
  pass = (norm1 / sz < 1.e-5) && (norm2 / sz < 1.e-5) && (norm3 / sz < 1.e-5);
  //pass = (norm2 / sz < 1.e-5);
  if (dw.rank == 0){
    if (!pass)
      printf("Failed TTMC tests.\n");
    else
      printf("Passed TTMC tests.\n");
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
    if (sp < 0.0 || sp > 1.0) sp = 1.;
  } else sp = 1.;

  {
    World dw;
    if (dw.rank == 0){
      printf("Running sparse (%lf fraction zeros) TTMC on order 3 tensor with dimension %d\n", sp, n);
    }

    bool pass;
    pass = ttmc<double>(n, ur, vr, wr, sp, dw);
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
