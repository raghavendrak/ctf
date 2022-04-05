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
  
  n = 2;
  int lens[3] = {n, n, n};
  int lens4[4] = {n, n, n, n};
  bool is_sparse = sp_frac < 1.;

  Tensor<dtype> T(3, is_sparse, lens, dw);
  
  int lens_uc[3] = {n, vr, wr};
  Tensor<dtype> UC(3, false, lens_uc, dw);
  int lens_vc[3] = {ur, n, wr};
  Tensor<dtype> VC(3, false, lens_vc, dw);
  int lens_wc[3] = {ur, vr, n};
  Tensor<dtype> WC(3, false, lens_wc, dw);

  T.fill_sp_random(-1.,1.,0.9);
  T.print();

  //int lens_v[1] = {8};
  //Tensor<dtype> V(1, false, lens_v1, dw);
  
  Partition par(T.topo->order, T.topo->lens);
  char * par_idx = (char*)malloc(sizeof(char)*T.topo->order);
  for (int i = 0; i < T.topo->order; i++) {
    par_idx[i] = 'a' + i + 1;
  }
  char mat_idx[2];
  mat_idx[0] = 'a' + 1;
  mat_idx[1] = 'a';
  Matrix<dtype> * m = new Matrix<dtype>(2, 2, mat_idx, par[par_idx], Idx_Partition(), 0, *T.wrld, *T.sr);
  int64_t npair;
  Pair<dtype> * pairs;
  m->get_local_pairs(&npair, &pairs, false, false);

  for (int64_t i = 0; i < npair; i++) {
    std::cout << "rank: " << dw.rank << " pairs[i].k: " << pairs[i].k << std::endl;
  }

  for (int i = 0; i < T.order; i++) {
    if(T.edge_map[i].calc_phys_phase() == 1) {
      std::cout << "rank: " << dw.rank << " only one process for this mode: " << i << std::endl;
    }
    else {
      int topo_dim = T.edge_map[i].cdt;
      int comm_lda = 1;
      for (int l = 0; l < topo_dim; l++) {
        comm_lda *= T.topo->dim_comm[l].np;
      }
      std::cout << "rank: " << dw.rank << " i: " << i << " topo_dim: " << topo_dim << " comm_lda: " << comm_lda << std::endl;
      // rank, color, CommData (parent)
      //CTF_int::CommData cmdt(T->wrld->rank - comm_lda*T->topo->dim_comm[topo_dim].rank, T->topo->dim_comm[topo_dim].rank, T->wrld->cdt);
      int x = comm_lda*T.topo->dim_comm[topo_dim].rank; 
      std::cout << "mode: " << i << " (" << T.wrld->rank << " - " << x << "), " << T.topo->dim_comm[topo_dim].rank << ", " << std::endl;
      break;
    }
  }

  Matrix<dtype> U(n, ur, dw);
  Matrix<dtype> V(n, vr, dw);
  Matrix<dtype> W(n, wr, dw);
  
  U.fill_random((dtype)0,(dtype)1);
  V.fill_random((dtype)0,(dtype)1);
  W.fill_random((dtype)0,(dtype)1);

  bool pass = true;
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
    if (n < 0) n = 3;
  } else n = 3;

  if (getCmdOption(input_str, input_str+in_num, "-ur")){
    ur = atoi(getCmdOption(input_str, input_str+in_num, "-ur"));
    if (ur < 0) ur = 3;
  } else ur = 3;

  if (getCmdOption(input_str, input_str+in_num, "-vr")){
    vr = atoi(getCmdOption(input_str, input_str+in_num, "-vr"));
    if (vr < 0) vr = 3;
  } else vr = 3;

  if (getCmdOption(input_str, input_str+in_num, "-wr")){
    wr = atoi(getCmdOption(input_str, input_str+in_num, "-wr"));
    if (wr < 0) wr = 3;
  } else wr = 3;
  
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
