/** \addtogroup examples 
  * @{ 
  * \defgroup spttn_tucker_solve_kernels spttn_tucker_solve_kernels 
  * @{ 
  * \brief Computes several tensor contractions required for Tucker decomposition/completion
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;
#include <unordered_set>

template <typename dtype>
bool execute_spttn_kernel(int n, int ur, int vr, int wr, 
                          double sp_frac, World & dw) {
  
  bool is_sparse = sp_frac < 1.;
  bool mpass = true;
  double stime;
  double etime;

  {
    int lens[4] = {n, n, n, n};
    int lens_uc[4], lens_vc[4], lens_wc[4];
    int n1, n2, n3, n4;
    n1 = n2 = n3 = n4 = n;
    lens[0] = n1;
    lens[1] = n2;
    lens[2] = n3;
    lens[3] = n4;

    // std::unordered_set<std::string> run = {"tucker_contraction_1", "tucker_contraction_2"};
    std::unordered_set<std::string> run = {"tucker_contraction_4"};

    if (run.count("tucker_contraction_1") > 0) {
      /*
      ijk,ai,bj,abc->ck
      1  ,2 ,4 ,8 
      path chosen: 11
      ta: 1 tb: 2 tab: 3 inds: 14
      ta: 4 tb: 3 tab: 7 inds: 28
      ta: 8 tb: 7 tab: 15 inds: 36
      ijk,ai -> ajk
      bj,ajk -> bak
      abc,bak -> ck
      total loop depth: 12
      term id 0: 4 2 1 8 
      term id 1: 4 2 8 16 
      term id 2: 4 8 16 32 
      niloops: 6  
      for k:
        for j:
          for i:
            for a:
              buf[a] += T_ijk * U_ai
          for a:
            for b:
              buf[a,b] += buf[a] * V_bj
        for a:
          for b:
            for c:
              Z_ck += buf[a,b] * W_abc
      */
      lens_uc[0] = ur; lens_uc[1] = n1; 
      lens_vc[0] = vr; lens_vc[1] = n2;
      lens_wc[0] = wr; lens_wc[1] = n3;
      int lens_oc[3] = {ur, vr, wr};
      
      Tensor<dtype> T(3, is_sparse, lens, dw);
      T.fill_sp_random(-1., 1., sp_frac);
      Matrix<dtype> U(lens_uc[0], lens_uc[1], dw);
      Matrix<dtype> V(lens_vc[0], lens_vc[1], dw);
      U.fill_random((dtype)0,(dtype)1);
      V.fill_random((dtype)0,(dtype)1);
      // core tensor
      Tensor<dtype> C(3, false, lens_oc, dw);
      C.fill_random((dtype)0,(dtype)1);
      Tensor<dtype> UC(2, false, lens_wc, dw);
      Tensor<dtype> * ops[4] = {&U, &V, &C, &UC};

      stime = MPI_Wtime();
      spttn_kernel<dtype>(&T, ops, 4, "ijk,ai,bj,abc->ck");
      etime = MPI_Wtime();
      if (dw.rank == 0) printf("ijk,ai,bj,abc->ck using SpTTN-Cyclops (NOTE that it includes CSF construction time; please see total time to calculate printed above): %1.2lf\n", (etime - stime));

      Tensor<dtype> UCxx(2, false, lens_wc, dw);
      stime = MPI_Wtime();
      UCxx["ck"] = T["ijk"] * U["ai"] * V["bj"] * C["abc"];
      etime = MPI_Wtime();
      if (dw.rank == 0) printf("ijk,ai,bj,abc->ck using CTF: %1.2lf\n", (etime - stime));

      double norm; 
      UCxx["ij"] -= UC["ij"];
      UCxx.norm2(norm);
      int64_t sz = T.get_tot_size(false);
      bool pass = (norm / sz < 1.e-5);
      if (dw.rank == 0) {
        if (!pass)
          printf("Test failed.\n");
        else
          printf("Test passed.\n");
      }
      IASSERT(pass);
      mpass = mpass & pass;
    }
    if (run.count("tucker_contraction_2") > 0) {
      /*
      ijk,ai,bj,abc,li,mj,lmn->cnk
        1, 2, 4,  8,16,32, 64->UC
      path chosen: 12132
      ta: 4 tb: 32 tab: 36 inds: 146
      bj,mj->bmj
      ta: 8 tb: 36 tab: 44 inds: 170
      abc,bmj->acmj
      ta: 16 tb: 64 tab: 80 inds: 385
      li,lmn->imn
      ta: 2 tb: 80 tab: 82 inds: 393
      ai,imn->aimn
      ta: 1 tb: 82 tab: 83 inds: 398
      ijk,almn->ajkmn
      ta: 44 tb: 83 tab: 127 inds: 292
      acmj,ajkmn->cnk
      total loop depth: 28
      term id 0: 16 2 128 
      term id 1: 16 2 8 32 128 
      term id 2: 128 256 1 64 
      term id 3: 128 8 256 1 
      term id 4: 128 8 4 2 1 256 
      term id 5: 128 8 4 2 32 256 
      i, j, k, a,  b,  c,  l,   m,   n
      1, 2, 4, 8, 16, 32, 64, 128, 256
      (3+5+4+4+6+6)=28
      for b:
        for j:
          for m:
            buf[m] <- bj,mj
          for a:
            for c:
              for m:
                buf[acmj] <- abc,buf[m]
      for m:
        for n:
          for i:
            for l:
              buf1[in] <- li,lmn
        for a:
          for n:
            for i:
              buf2[in] <- ai,buf1[in]
          for k:
            for j:
              for i:
                for n:
                  buf[n] <- ijk,buf2[in]
              for c:
                for n:
                  Z_cnk += buf[acmj],buf[n]

      path chosen: 3815
      ijk,ai,bj,abc,li,mj,lmn->cnk
        1, 2, 4,  8,16,32, 64->UC
      ta: 1 tb: 2 tab: 3 inds: 15
      ijk,ai->aijk
      ta: 16 tb: 3 tab: 19 inds: 78
      li,aijk->alijk
      ta: 64 tb: 19 tab: 83 inds: 398
      lmn,alijk->ajkmn
      ta: 32 tb: 83 tab: 115 inds: 270
      mj,ajkmn->ajkn
      ta: 4 tb: 115 tab: 119 inds: 284
      bj,ajkn->abkn
      ta: 8 tb: 119 tab: 127 inds: 292
      abc,abkn->cnk
      total loop depth: 30
      term id 0: 4 8 2 1 
      term id 1: 4 8 2 1 64 
      term id 2: 4 8 2 64 128 256 
      term id 3: 4 8 2 128 256 
      term id 4: 4 8 2 16 256 
      term id 5: 4 8 16 32 256 
      term id 0: 2 3 1 0 
      term id 1: 2 3 1 0 6 
      term id 2: 2 3 1 6 7 8 
      term id 3: 2 3 1 7 8 
      term id 4: 2 3 1 4 8 
      term id 5: 2 3 4 5 8 
      i, j, k, a,  b,  c,  l,   m,   n
      1, 2, 4, 8, 16, 32, 64, 128, 256
      (4+5+6+5+5+5)=30
      for k:
        for a:
          for j:
            for i:
              buf = ijk * ai
              for l:
                buf[l] += li * buf
            for l:
              for m:
                for n:
                  buf[mn] += lmn * buf[l]
            for m:
              for n:
                buf[n] += mj * buf[mn]
            for b:
              for n:
                buf[bn] += bj * buf[n]
          for b:
            for c:
              for n:
                Z_cnk += abc * buf[bn]
      niloops: 11
      */
      lens_uc[0] = ur; lens_uc[1] = n1; 
      lens_vc[0] = vr; lens_vc[1] = n2;
      int lens_oc[3] = {ur, vr, wr};
      lens_wc[0] = wr; lens_wc[1] = wr; lens_wc[2] = n3;
      Tensor<dtype> T(3, is_sparse, lens, dw);
      T.fill_sp_random(-1., 1., sp_frac);
      Matrix<dtype> U(lens_uc[0], lens_uc[1], dw);
      Matrix<dtype> V(lens_vc[0], lens_vc[1], dw);
      U.fill_random((dtype)0,(dtype)1);
      V.fill_random((dtype)0,(dtype)1);
      Matrix<dtype> UT(lens_uc[0], lens_uc[1], dw);
      Matrix<dtype> VT(lens_vc[0], lens_vc[1], dw);
      UT.fill_random((dtype)0,(dtype)1);
      VT.fill_random((dtype)0,(dtype)1);
      Tensor<dtype> C(3, false, lens_oc, dw);
      C.fill_random((dtype)0,(dtype)1);
      Tensor<dtype> CT(3, false, lens_oc, dw);
      CT.fill_random((dtype)0,(dtype)1);
      Tensor<dtype> UC(3, false, lens_wc, dw);
      Tensor<dtype> * ops[7] = {&U, &V, &C, &UT, &VT, &CT, &UC};

      stime = MPI_Wtime();
      spttn_kernel<dtype>(&T, ops, 7, "ijk,ai,bj,abc,li,mj,lmn->cnk");
      etime = MPI_Wtime();
      if (dw.rank == 0) printf("ijk,ai,bj,abc,li,mj,lmn->cnk using SpTTN-Cyclops (NOTE that it includes CSF construction time; please see total time to calculate printed above): %1.2lf\n", (etime - stime));

      Tensor<dtype> UCxx(3, false, lens_wc, dw);
      stime = MPI_Wtime();
      UCxx["cnk"] = T["ijk"] * U["ai"] * V["bj"] * C["abc"] * UT["li"] * VT["mj"] * CT["lmn"];
      etime = MPI_Wtime();
      if (dw.rank == 0) printf("ijk,ai,bj,abc,li,mj,lmn->cnk using CTF: %1.2lf\n", (etime - stime));

      double norm; 
      UCxx["ijk"] -= UC["ijk"];
      UCxx.norm2(norm);
      int64_t sz = T.get_tot_size(false);
      bool pass = (norm / sz < 1.e-5);
      if (dw.rank == 0) {
        if (!pass)
          printf("Test failed.\n");
        else
          printf("Test passed.\n");
      }
      IASSERT(pass);
      mpass = mpass & pass;
    }
    if (run.count("tucker_contraction_3") > 0) {
      /*
      einsolve("kji,ri,rj,rk->kji",Omega,U,V,W,T)
      LHS = ("kji,ri,rj,zi,zj->rzk",Omega,U,V,U,V)
      RHS = ("kji,ri,rj->rk",Omega,U,V)
      Soln = ("zi")

      (3+3+4+5)
      for j:
        for i:
          for r:
            buf[i,j,r] += U[r,i] * V[r,j]
          for z:
            buf[i,j,z] += U[z,i] * V[z,j]
      for k:
        for j:
          for i:
            for r:
              sbuf1 = Omega[i,j,k] * buf[i,j,r]
              for z:
                Z_buf[r,z,k] += sbuf1 * buf[i,j,z]
               

      (4+5+4+4)
      for k:
        for j:
          for i:
            for r:
              sbuf1 = Omega[i,j,k] * U[r,i]
              for z:
                buf[r,z] += sbuf1 * U[z,i]
          for r:
            for z:
              sbuf2 = buf[r,z] * V[r,j]
              Z_buf[r,z,k] += sbuf2 * V[z,j]

      path chosen: 114
      ta: 4 tb: 16 tab: 20 inds: 26
      ta: 2 tb: 8 tab: 10 inds: 25
      ta: 1 tb: 10 tab: 11 inds: 30
      ta: 20 tb: 11 tab: 31 inds: 28
      contraction path
      ijk,ri,rj,zi,zj->rzk
      1  ,2 ,4 ,8 ,16
      rj,zj -> 20
      ri,zi -> 10
      ijk,10 -> 11
      20,11 -> 31
      i, j, k, r, z
      1, 2, 4, 8, 16
      for r:
        for j:
          for z:
            rzj <- rj,zj
      for z:
        for r:
          for i:
            rzi <- ri,zi
        for k:
          for j:
            for i:
              for r:
                rzjk <- ijk,rzi
            for r:
              rzk <- rzj,rzjk
      total loop depth: 15 (3+3+5+4)
      term id: 0 8 2 16
      term id: 1 16 8 1
      term id: 2 16 4 2 1 8
      term id: 3 16 4 2 8
      */
      lens_uc[0] = ur; lens_uc[1] = n1; 
      lens_vc[0] = vr; lens_vc[1] = n2;
      lens_wc[0] = ur; lens_wc[1] = vr; lens_wc[2] = n3;
      Tensor<dtype> T(3, is_sparse, lens, dw);
      T.fill_sp_random(-1., 1., sp_frac);
      Matrix<dtype> U(lens_uc[0], lens_uc[1], dw);
      Matrix<dtype> V(lens_vc[0], lens_vc[1], dw);
      U.fill_random((dtype)0,(dtype)1);
      V.fill_random((dtype)0,(dtype)1);
      Tensor<dtype> UC(3, false, lens_wc, dw);
      //Tensor<dtype> * ops[5] = {&U, &V, &Udummy, &Vdummy, &UC};
      Tensor<dtype> * ops[5] = {&U, &V, &U, &V, &UC};
      //std::string terms[4] = {"ijk,ri->rijk", "rijk,zi->rzjk", "rzjk,rj->rzjk", "rzjk,zj->rzk"};
      //std::string index_orders[4] = {"kjir", "kjirz", "kjrz", "kjrz"};

      stime = MPI_Wtime();
      spttn_kernel<dtype>(&T, ops, 5, "ijk,ri,rj,zi,zj->rzk");
      etime = MPI_Wtime();
      if (dw.rank == 0) printf("ijk,ri,rj,zi,zj->rzk using SpTTN-Cylops (NOTE that it includes CSF construction time; please see total time to calculate printed above): %1.2lf\n", (etime - stime));

      Tensor<dtype> UCxx(3, false, lens_wc, dw);
      stime = MPI_Wtime();
      UCxx["rzk"] = T["ijk"] * U["ri"] * V["rj"] * U["zi"] * V["zj"];
      etime = MPI_Wtime();
      if (dw.rank == 0) printf("ijk,ri,rj,zi,zj->rzk using CTF: %1.2lf\n", (etime - stime));

      double norm; 
      UCxx["ijk"] -= UC["ijk"];
      UCxx.norm2(norm);
      int64_t sz = T.get_tot_size(false);
      bool pass = (norm / sz < 1.e-5);
      if (dw.rank == 0) {
        if (!pass)
          printf("Test failed.\n");
        else
          printf("Test passed.\n");
      }
      IASSERT(pass);
      mpass = mpass & pass;
    }
    if (run.count("tucker_contraction_4") > 0) {
      /*
      with thres_buf_sz = 1
      path chosen: 2
      ta: 4 tb: 8 tab: 12 inds: 54
      ta: 16 tb: 12 tab: 28 inds: 14
      ta: 2 tb: 28 tab: 30 inds: 7
      ta: 1 tb: 30 tab: 31 inds: 7
      total loop depth: 16
      term id 0: 4 2 16 32 
      term id 1: 4 2 16 8 32 
      term id 2: 4 2 1 8 
      term id 3: 4 2 1 
      UCxx["ijk"] = T["ijk"] * U["ai"] * V["bj"] * W["ck"] * C["abc"];

      4 + 5 + 4 + 3 = 16
      for k:
        for j:
          for b:
            for c:
              buf[c] += V[b,j] * W[c,k]
            for a:
              for c:
                buf[a] += buf[c] * C[a,b,c]
          for i:
            for a:
              buf2 += buf[a] * U[a,i]
            Z_ijk += buf2 * T_ijk


      with thres_buf_sz = 2
      path chosen: 0
      ta: 8 tb: 16 tab: 24 inds: 28
      ta: 4 tb: 24 tab: 28 inds: 14
      ta: 2 tb: 28 tab: 30 inds: 7
      ta: 1 tb: 30 tab: 31 inds: 7
      total loop depth: 15
      term id 0: 4 8 16 32
      term id 1: 4 2 8 16
      term id 2: 4 2 1 8
      term id 3: 4 2 1
      niloops: 6

      i: 1 j: 2 k: 4 a: 8 b: 16 c: 32
      T: 1 U: 2 V: 4 W: 8 C: 16
      4 + 4 + 4 + 3 = 15
      for k:
        for a:
          for b:
            for c:
              buf[a,b,c] += W[c,k] * C[a,b,c]
        for j:
          for a:
            for b:
              buf2[a] += buf[a,b,c] * V[b,j]
          for i:
            for a:
              buf += buf2[a] * U[a,i]
            Z_ijk += buf * T[i,j,k]

      */

      lens_uc[0] = ur; lens_uc[1] = n1; 
      lens_vc[0] = vr; lens_vc[1] = n2;
      lens_wc[0] = wr; lens_wc[1] = n3;
      int lens_oc[3] = {ur, vr, wr};
      
      Tensor<dtype> T(3, is_sparse, lens, dw);
      T.fill_sp_random(-1., 1., sp_frac);
      Tensor<dtype> UC(3, is_sparse, lens, dw);
      UC["ijk"] = T["ijk"];
      Tensor<dtype> UCxx(3, is_sparse, lens, dw);
      Matrix<dtype> U(lens_uc[0], lens_uc[1], dw);
      Matrix<dtype> V(lens_vc[0], lens_vc[1], dw);
      Matrix<dtype> W(lens_wc[0], lens_wc[1], dw);
      U.fill_random((dtype)0,(dtype)1);
      V.fill_random((dtype)0,(dtype)1);
      W.fill_random((dtype)0,(dtype)1);
      // core tensor
      Tensor<dtype> C(3, false, lens_oc, dw);
      C.fill_random((dtype)0,(dtype)1);

      Tensor<dtype> * ops[5] = {&U, &V, &W, &C, &UC};
      int max_buf_dim = 2;
      stime = MPI_Wtime();
      spttn_kernel<dtype>(&T, ops, 5, "ijk,ai,bj,ck,abc->ijk", max_buf_dim);
      etime = MPI_Wtime();
      if (dw.rank == 0) printf("ijk,ai,bj,ck,abc->ijk using SpTTN-Cyclops (NOTE that it includes CSF construction time; please see total time to calculate printed above): %1.2lf\n", (etime - stime));

      stime = MPI_Wtime();
      UCxx["ijk"] = T["ijk"];
      UCxx["ijk"] = T["ijk"] * U["ai"] * V["bj"] * W["ck"] * C["abc"];
      etime = MPI_Wtime();
      if (dw.rank == 0) printf("ijk,ai,bj,ck,abc->ijk using CTF: %1.2lf\n", (etime - stime));

      double norm; 
      UCxx["ijk"] -= UC["ijk"];
      UCxx.norm2(norm);
      int64_t sz = T.get_tot_size(false);
      bool pass = (norm / sz < 1.e-5);
      if (dw.rank == 0) {
        if (!pass)
          printf("Test failed.\n");
        else
          printf("Test passed.\n");
      }
      IASSERT(pass);
      mpass = mpass & pass;
      exit(0);
    }
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

int main(int argc, char ** argv) 
{
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
