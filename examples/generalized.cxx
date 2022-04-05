/** \addtogroup examples 
  * @{ 
  * \defgroup generalized generalized 
  * @{ 
  * \brief Computes generalized
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;


template <typename dtype>
void print_op(Tensor<dtype> & T,
           Tensor<dtype> & U,
           Tensor<dtype> & V,
           Tensor<dtype> & W,
           Tensor<dtype> & UCxx,
           Tensor<dtype> & UC)
{
  T.print();
  U.print();
  V.print();
  W.print();
  UCxx.print();
  UC.print();
}

template <typename dtype>
bool generalized(int n, int ur, int vr, int wr, 
                 double sp_frac, World & dw) {
  
  int n_ = n;
  n = 200;
  int ur_ = ur;
  ur = vr = wr = n;
  int lens[3] = {n, n, n};
  bool is_sparse = sp_frac < 1.;

  Tensor<dtype> T(3, is_sparse, lens, dw);
  
  int lens_uc[3] = {n, vr, wr};
  Tensor<dtype> UC(3, false, lens_uc, dw);
  int lens_vc[3] = {ur, n, wr};
  Tensor<dtype> VC(3, false, lens_vc, dw);
  int lens_wc[3] = {ur, vr, n};
  Tensor<dtype> WC(3, false, lens_wc, dw);

  T.fill_sp_random(-1., 1., sp_frac);
  /*
  int64_t npair;
  Pair<dtype> *pairs;
  T.get_local_pairs(&npair, &pairs);
  for (int64_t i = 0; i < npair; i++) {
    pairs[i].d = pairs[i].k;
  }
  //T.write(npair, pairs);
  free(pairs);
  */

  Matrix<dtype> U(n, ur, dw);
  Matrix<dtype> V(n, vr, dw);
  Matrix<dtype> W(n, wr, dw);
  
  U.fill_random((dtype)0,(dtype)1);
  V.fill_random((dtype)0,(dtype)1);
  W.fill_random((dtype)0,(dtype)1);

  bool mpass = true;
  double stime;
  double etime;
#ifdef RUN_ALL 
#endif
  {
    // TTTP
    n = 1000;
    ur = vr = wr = 60;
    int lens[3] = {n, n, n};
    Tensor<dtype> T(3, is_sparse, lens, dw);
    T.fill_sp_random(-1., 1., sp_frac);

    Matrix<dtype> U(n, ur, dw);
    Matrix<dtype> V(n, vr, dw);
    Matrix<dtype> W(n, wr, dw);
    U.fill_random((dtype)0,(dtype)1);
    V.fill_random((dtype)0,(dtype)1);
    W.fill_random((dtype)0,(dtype)1);
   
    int lens_uc[3] = {n, n, n};
    Tensor<dtype> UCxx(3, false, lens_uc, dw);
    stime = MPI_Wtime();
    UCxx["ijk"] = T["ijk"] * U["ia"] * V["ja"] * W["ka"];
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("tttp dense: %1.2lf\n", (etime - stime));
    
    Tensor<dtype> UC(3, false, lens_uc, dw);
    Tensor<dtype> * ops[4] = {&U, &V, &W, &UC};
    stime = MPI_Wtime();
    gen_multilinear<dtype>(&T, ops, 4, "ijk,ia,ja,ka->ijk");
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("tttp gen_multilinear: %1.2lf\n", (etime - stime));
    if (dw.rank == 0) printf("n: %d ur: %d sp_frac: %lf-----------------------------\n", n, ur, sp_frac);
    
    double norm; 
    UCxx["ijk"] -= UC["ijk"];
    UCxx.norm2(norm);
    int64_t sz = T.get_tot_size(false);
    bool pass = (norm / sz < 1.e-5);
    IASSERT(pass);
    if (dw.rank == 0){
      if (!pass)
        printf("Failed TTTP.\n");
      else
        printf("Passed TTTP.\n");
    }
    mpass = mpass & pass;
  }
  {
    // TTMc
    n = n_;
    ur = vr = wr = ur_;
    int lens[3] = {n, n, n};
    Tensor<dtype> T(3, is_sparse, lens, dw);
    T.fill_sp_random(-1., 1., sp_frac);

    int lens_uc[3] = {n, vr, wr};
    Tensor<dtype> UCxx(3, false, lens_uc, dw);
    Matrix<dtype> U(n, ur, dw);
    Matrix<dtype> V(n, vr, dw);
    Matrix<dtype> W(n, wr, dw);
    U.fill_random((dtype)0,(dtype)1);
    V.fill_random((dtype)0,(dtype)1);
    W.fill_random((dtype)0,(dtype)1);
    
    stime = MPI_Wtime();
    UCxx["irs"] = T["ijk"] * U["jr"] * V["ks"];
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("ttmc dense: %1.2lf\n", (etime - stime));
    
    Tensor<dtype> UC(3, false, lens_uc, dw);
    Tensor<dtype> * ops[3] = {&U, &V, &UC};
    stime = MPI_Wtime();
    gen_multilinear<dtype>(&T, ops, 3, "ijk,jr,ks->irs");
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("ttmc gen_multilinear: %1.2lf\n", (etime - stime));
    if (dw.rank == 0) printf("n: %d ur: %d sp_frac: %lf-----------------------------\n", n, ur, sp_frac);

    double norm; 
    UCxx["ijk"] -= UC["ijk"];
    UCxx.norm2(norm);
    int64_t sz = T.get_tot_size(false);
    bool pass = (norm / sz < 1.e-5);

    if (dw.rank == 0){
      if (!pass)
        printf("Failed TTMc.\n");
      else
        printf("Passed TTMc.\n");
    }
    //IASSERT(pass);
    mpass = mpass & pass;
  }
  {
    // TTMc - 3 input operands
    n = n_;
    ur = vr = wr = ur_;
    int lens[4] = {n, n, n, n};
    Tensor<dtype> T(4, is_sparse, lens, dw);
    T.fill_sp_random(-1., 1., sp_frac);

    int lens_uc[4] = {n, ur, vr, wr};
    Tensor<dtype> UCxx(4, false, lens_uc, dw);
    Matrix<dtype> U(n, ur, dw);
    Matrix<dtype> V(n, vr, dw);
    Matrix<dtype> W(n, wr, dw);
    U.fill_random((dtype)0,(dtype)1);
    V.fill_random((dtype)0,(dtype)1);
    W.fill_random((dtype)0,(dtype)1);
    
    stime = MPI_Wtime();
    UCxx["irst"] = T["ijkl"] * U["jr"] * V["ks"] * W["lt"];
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("ttmc dense: %1.2lf\n", (etime - stime));
    
    Tensor<dtype> UC(4, false, lens_uc, dw);
    Tensor<dtype> * ops[4] = {&U, &V, &W, &UC};
    stime = MPI_Wtime();
    gen_multilinear<dtype>(&T, ops, 4, "ijkl,jr,ks,lt->irst");
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("ttmc gen_multilinear: %1.2lf\n", (etime - stime));
    if (dw.rank == 0) printf("n: %d ur: %d sp_frac: %lf-----------------------------\n", n, ur, sp_frac);

    double norm; 
    UCxx["ijkl"] -= UC["ijkl"];
    UCxx.norm2(norm);
    int64_t sz = T.get_tot_size(false);
    bool pass = (norm / sz < 1.e-5);

    if (dw.rank == 0){
      if (!pass)
        printf("Failed TTMc - 3 operands.\n");
      else
        printf("Passed TTMc - 3 operands.\n");
    }
    //IASSERT(pass);
    mpass = mpass & pass;
  }

  {
    // MTTKRP 
    n = n_;
    ur = ur_;

    int lens[3] = {n, n, n};
    Tensor<dtype> T(3, is_sparse, lens, dw);
    T.fill_sp_random(-1., 1., sp_frac);
    
    int lens_uc[2] = {n, ur};
    
    Tensor<dtype> UCxx(2, false, lens_uc, dw);
    Tensor<dtype> U(2, false, lens_uc, dw);
    Tensor<dtype> V(2, false, lens_uc, dw);
    U.fill_random((dtype)0,(dtype)1);
    V.fill_random((dtype)0,(dtype)1);
    stime = MPI_Wtime();
    //UCxx["ia"] = T["ijk"] * U["ja"] * V["ka"];
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("mttkrp dense: %1.2lf\n", (etime - stime));

    Tensor<dtype> UC(2, false, lens_uc, dw);
    Tensor<dtype> * ops[3] = {&U, &V, &UC};
    stime = MPI_Wtime();
    gen_multilinear<dtype>(&T, ops, 3, "ijk,ja,ka->ia");
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("mttkrp gen_multilinear: %1.2lf\n", (etime - stime));

    int lens_ucT[2] = {lens_uc[1], lens_uc[0]};
    Tensor<dtype> UT(2, false, lens_ucT, dw);
    UT["ji"] = U["ij"]; 
    Tensor<dtype> VT(2, false, lens_ucT, dw);
    VT["ji"] = V["ij"]; 
    Tensor<dtype> UCy(2, false, lens_ucT, dw);
    Tensor<dtype> * mlist4[3] = {&UCy, &UT, &VT};
    int mode = 0;
    stime = MPI_Wtime();
    MTTKRP<dtype>(&T, mlist4, mode, true);
    etime = MPI_Wtime();
    if (dw.rank == 0) printf("mttkrp sparse implementation: %1.2lf\n", (etime - stime));
    Tensor<dtype> UCyT(2, false, lens_uc, dw);
    UCyT["ji"] = UCy["ij"];
    if (dw.rank == 0) printf("n: %d ur: %d-----------------------------\n", n, ur);
    
    double norm; 
    //UCxx["ij"] -= UC["ij"];
    //UCxx.norm2(norm);
    UCyT["ij"] -= UC["ij"];
    UCyT.norm2(norm);
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
#ifdef RUN_ALL 
  {
    // optimized contraction 
    int lens_uc[3] = {n, 8, wr};
    int lens_2[2] = {n, wr};
    Tensor<dtype> UCxx(3, false, lens_uc, dw);
    Tensor<dtype> T(3, is_sparse, lens_uc, dw);
    //T.fill_sp_random(-1.,1.,sp_frac);
    int64_t npair;
    Pair<dtype> * pairs;
    T.get_local_pairs(&npair, &pairs);
    for (int64_t i = 0; i < npair; i++) {
      pairs[i].d = pairs[i].k;
    }
    T.write(npair, pairs);
    free(pairs);

    int lensY[1] = {8};
    Tensor<dtype> Ux(1, false, lensY, dw);
    //Ux.fill_random((dtype)0,(dtype)1);
    Ux.get_local_pairs(&npair, &pairs);
    for (int64_t i = 0; i < npair; i++) {
      pairs[i].d = pairs[i].k + 100;
    }
    Ux.write(npair, pairs);
    free(pairs);

    Tensor<dtype> Vx(2, false, lens_2, dw);
    Vx.fill_random((dtype)0,(dtype)1);
    Vx.get_local_pairs(&npair, &pairs);
    for (int64_t i = 0; i < npair; i++) {
      pairs[i].d = pairs[i].k + 200;
    }
    Vx.write(npair, pairs);
    free(pairs);

    UCxx["ijk"] = T["ijk"] * Ux["j"] * Vx["ik"];
    
    Tensor<dtype> UC(3, false, lens_uc, dw);
    Tensor<dtype> * ops[3] = {&Ux, &Vx, &UC};
    gen_multilinear<dtype>(&T, ops, 3, "ijk,j,ik->ijk");

    double norm; 
    UCxx["ijk"] -= UC["ijk"];
    UCxx.norm2(norm);
    int64_t sz = T.get_tot_size(false);
    bool pass = (norm / sz < 1.e-5);
    IASSERT(pass);
    if (dw.rank == 0){
      if (!pass)
        printf("Failed optimized contraction tests.\n");
      else
        printf("Passed optimized contraction tests.\n");
    }
    mpass = mpass & pass;
  }
//#endif

  //#("ijk,irs,jp->ijk")
  {
    // optimized contraction 
    int lens_uc[3] = {2, 2, 2};
    int lens_2[2] = {2, 2};
    Tensor<dtype> T(3, is_sparse, lens_uc, dw);
    Tensor<dtype> UCxx(3, false, lens_uc, dw);
    T.fill_sp_random(-1.,1.,sp_frac);
    
    int lensY[1] = {8};
    Tensor<dtype> Ux(3, false, lens_uc, dw);
    Ux.fill_random((dtype)0,(dtype)1);
    Tensor<dtype> Vx(2, false, lens_2, dw);
    Vx.fill_random((dtype)0,(dtype)1);
    UCxx["ijk"] = T["ijk"] * Ux["irs"] * Vx["jp"];
    //UCxx["ijk"] = T["ijk"] * Vx["jp"] * Ux["irs"];
    
    Tensor<dtype> UC(3, false, lens_uc, dw);
    Tensor<dtype> * ops[3] = {&Ux, &Vx, &UC};
    gen_multilinear<dtype>(&T, ops, 3, "ijk,irs,jp->ijk");

    double norm; 
    UCxx["ijk"] -= UC["ijk"];
    UCxx.norm2(norm);
    int64_t sz = T.get_tot_size(false);
    bool pass = (norm / sz < 1.e-5);
    IASSERT(pass);
    mpass = mpass & pass;
    
    Tensor<dtype> UCyy(3, false, lens_uc, dw);
    UCyy["ijk"] = T["ijk"] * Vx["jp"] * Ux["irs"];
    Tensor<dtype> UCy(3, false, lens_uc, dw);
    Tensor<dtype> * opsy[3] = {&Vx, &Ux, &UCy};
    gen_multilinear<dtype>(&T, opsy, 3, "ijk,jp,irs->ijk");
    
    UCyy["ijk"] -= UCy["ijk"];
    UCyy.norm2(norm);
    pass = (norm / sz < 1.e-5);
    IASSERT(pass);

    
    if (dw.rank == 0){
      if (!pass)
        printf("Failed optimized contraction tests.\n");
      else
        printf("Passed optimized contraction tests.\n");
    }
    mpass = mpass & pass;
  }
#endif

  // double norm1, norm2, norm3;
  // bool pass = (norm1 / sz1 < 1.e-5) && (norm2 / sz < 1.e-5) && (norm3 / sz < 1.e-5);
  if (dw.rank == 0) {
    if (!mpass)
      printf("Failed contraction tests.\n");
    else
      printf("Passed all contraction tests.\n");
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
    if (dw.rank == 0){
      //printf("Running sparse (%lf fraction zeros) TTMC on order 3 tensor with dimension %d\n", sp, n);
    }

    bool pass;
    pass = generalized<double>(n, ur, vr, wr, sp, dw);
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
