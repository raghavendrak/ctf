#include "gen_contraction.h"
#include "../redistribution/nosym_transp.h"
#include "../scaling/strp_tsr.h"
#include "../mapping/mapping.h"
#include "../mapping/distribution.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "sym_seq_ctr.h"
#include "spctr_comm.h"
#include "ctr_tsr.h"
#include "ctr_offload.h"
#include "ctr_2d_general.h"
#include "spctr_offload.h"
#include "spctr_2d_general.h"
#include "sp_gen_ctr.h"
#include "../symmetry/sym_indices.h"
#include "../symmetry/symmetrization.h"
#include "../redistribution/nosym_transp.h"
#include "../redistribution/redist.h"
#include "../sparse_formats/coo.h"
#include "../sparse_formats/csr.h"
#include "../sparse_formats/csf.h"
#include <cfloat>
#include <limits>
#include "../interface/tensor.h"

namespace CTF_int {

  using namespace CTF;
  template class gen_contraction<double>;
  // TODO: need to duplicate 
  //template class gen_contraction<int>;
  //template class gen_contraction<int64_t>;

  template<typename dtype>
  gen_contraction<dtype>::~gen_contraction()
  {
    if (idx_A != NULL) cdealloc(idx_A);
    if (idx_Bs != NULL) {
      for (int i = 0; i < nBs; i++) cdealloc(idx_Bs[i]);
      cdealloc(idx_Bs);
    }
    if (order_Bs != NULL) cdealloc(order_Bs);
    cdealloc(sr_Bs);    
  }

  template<typename dtype>
  gen_contraction<dtype>::gen_contraction(gen_contraction<dtype> const & other)
  {
    // TODO: fill up the copy constructor
  }

  template<typename dtype>
  gen_contraction<dtype>::gen_contraction(tensor *                  A_,
                                          int const *               idx_A_,
                                          tensor **                 Bs_,
                                          int                       nBs_,
                                          const int * const *       idx_Bs_,
                                          char const *              alpha_,
                                          bivar_function const *    func_) 
  {
    // TODO: might not need this function
    IASSERT(0);
  }
  
  template<typename dtype>
  gen_contraction<dtype>::gen_contraction(tensor *                  A_,
                                          char const *              cidx_A,
                                          tensor **                 Bs_,
                                          int                       nBs_,
                                          const char * const *      cidx_Bs,
                                          char const *              alpha_,
                                          bivar_function const *    func_) {
    A = A_;
    Bs = Bs_;
    nBs = nBs_;
    func = func_;
    alpha = alpha_;

    sr_A = A->sr;
    sr_Bs = (const algstrct **) alloc(sizeof(algstrct *) * nBs);
    for (int i = 0; i < nBs; i++) sr_Bs[i] = Bs[i]->sr;

    order_Bs = (int *)alloc(sizeof(int) * nBs);
    for (int i = 0; i < nBs; i++) order_Bs[i] = Bs[i]->order;
    
    gen_conv_idx(A->order, cidx_A, &idx_A, nBs, order_Bs, cidx_Bs, &idx_Bs); 

#ifdef debug0
    std::cout << "-----------\n";
    //std::cout << A->order << std::endl;
    for (int i = 0; i < A->order; i++)
      std::cout << cidx_A[i] << " " << idx_A[i] << " ";
    std::cout << std::endl;
    for (int i = 0; i < nBs; i++) {
      //std::cout << Bs[i]->order << std::endl;
      for (int j = 0; j < Bs[i]->order; j++)
        std::cout << cidx_Bs[i][j] << " " << idx_Bs[i][j] << " ";
      std::cout << std::endl;
    }
    std::cout << "\n-----------\n";
#endif
  }
  
  template<typename dtype>
  void gen_contraction<dtype>::execute()
  {
    Tensor<dtype> ** redist_Bs = (Tensor<dtype>**)malloc(sizeof(Tensor<dtype>*) * nBs);
    int64_t comm_lda[nBs];
    // TODO: can use sr_A
    // sr_A = A->sr as member variable in gen_contraction class (as in ctr_common.h)
    // will be able to use ConstPairIterator
    
    // TODO: should move this rev_idx_map calculation, and compute it only once
    int idx_max;
    int ** rev_idx_map;
    gen_inv_idx(A->order, idx_A, nBs, order_Bs, idx_Bs, &idx_max, &rev_idx_map);

    //distribute_operands(rev_idx_map, redist_Bs);
    {
      dtype **arrs = (dtype **)malloc(sizeof(dtype *) * nBs);

      int *phys_phase = (int *)malloc(A->order * sizeof(int));
      for (int i = 0; i < A->order; i++) {
        phys_phase[i] = A->edge_map[i].calc_phys_phase();
      }

      // partition for A, and assign idx for the dimensions
      Partition par(A->topo->order, A->topo->lens);
      char *par_idx = (char *)malloc(sizeof(char) * A->topo->order);
      for (int i = 0; i < A->topo->order; i++) {
        par_idx[i] = 'a' + i + 1;
      }

      for (int i = 0; i < nBs; i++) {
        char tBs_idx[order_Bs[i]];
        // Tensor<dtype> * tB = Bs[i];
        //  for each B, check if the idx is along A (mode)
        comm_lda[i] = 0;
        for (int j = 0; j < order_Bs[i]; j++) {

          // for each idx_B check if it is along tensor A or should be present
          // on all processes rev_idx_map[nBs] stores rev_idx_map for A
          int mode = rev_idx_map[nBs][idx_Bs[i][j]];
          if (mode == -1 || phys_phase[mode] == 1) {
            tBs_idx[j] = 'a' + j - 32;
          } else {
            int topo_dim = A->edge_map[mode].cdt;
            tBs_idx[j] = par_idx[topo_dim];

            comm_lda[i] = comm_lda[i] * A->topo->dim_comm[topo_dim].np +
                          A->topo->dim_comm[topo_dim].rank;
          }

          // m->operator[]("ij") += mat->operator[]("ij");
          // redist_mats[i] = m;

          // we are considering the phys_phase of the main tensor A because all
          // other operand tensors are placed along these modes
          // if (phys_phase[i] == 1) {
          // arrs[i] = (dtype *)A->sr->alloc(A->lens[modes[i]] * kd);
          // mat->read_all(arrs[i], true);
          // read_all() makes sense for input operands; instead of placing them
          // in one process and broadcasting on all processes, we are just
          // reading data in all processes
          // redist_mats[i] = NULL;
          //}
        }
        std::string name_B("redist_Bs_");
        std::ostringstream s;
        s << i;
        name_B += s.str();
        // std::cout << "creating a new tensor: " << name_B << std::endl;
        Tensor<dtype> *t = new Tensor<dtype>(
            Bs[i]->order, false, Bs[i]->lens, Bs[i]->sym, *A->wrld, tBs_idx,
            par[par_idx], Idx_Partition(), name_B.data(), 0, *A->sr);
        redist_Bs[i] = t;
      }

      // redist_Bs->operator[]("ij") += Bs->operator[]("ij");
      //std::cout << "A->order: " << A->order << std::endl;
      for (int i = 0; i < nBs - 1; i++) {
        std::string s;
        for (int j = 0; j < order_Bs[i]; j++) {
          s += 'a' + j;
        }
        // std::cout << "string: " << s << std::endl;
        redist_Bs[i]->operator[](s.data()) += Bs[i]->operator[](s.data());
        //std::cout << "for Bs_j: " << i << " rank: " << A->wrld->rank << " color: " << comm_lda[i] << std::endl;
        CTF_int::CommData cmdt(A->wrld->rank, comm_lda[i], A->wrld->cdt);
        cmdt.bcast(redist_Bs[i]->data, redist_Bs[i]->size, A->sr->mdtype(), 0);
        /*
        int64_t npair;
        Pair<dtype> *pairs;
        redist_Bs[i]->get_local_pairs(&npair, &pairs);
        for (int64_t k = 0; k < npair; k++) {
          std::cout << "rank: " << A->wrld->rank << " pairs[" << k
                    << "].d: " << pairs[k].d << std::endl;
        }
        free(pairs);
        */
      }
    }

    int64_t npair;
    Pair<dtype> *pairs;
    if (A->is_sparse) {
      pairs = (Pair<dtype>*)A->data;
      npair = A->nnz_loc;
    } else {
      IASSERT(0);
    }
    //std::cout << "My rank: " << A->wrld->rank << " npair: " << npair << std::endl; 
    int * phys_phase = (int *) malloc(A->order * sizeof(int));
    for (int i = 0; i < A->order; i++) {
      phys_phase[i] = A->edge_map[i].calc_phys_phase();
    }
    
    double stime;
    double etime;
    MPI_Barrier(MPI_COMM_WORLD);
    stime = MPI_Wtime();
    CTF_int::CSF<dtype> A_csf(npair, pairs, A->order, A->lens, phys_phase);
    MPI_Barrier(MPI_COMM_WORLD);
    etime = MPI_Wtime();
    if (A->wrld->rank == 0) printf("tree construction time: %1.2lf\n", (etime - stime));
    
    //std::cout << "Tree construction done" << std::endl;

    // test recursive tree traveral
    int levels = A->order - 1;
    /*
    for (int64_t i = 0; i < A_csf.nnz_level[levels]; i++) {
      traverse_CSF(&A_csf, levels, i);
    }
    */
    
    int64_t ** edge_len_Bs = (int64_t **)CTF_int::alloc(sizeof(int64_t *) * nBs);
    for (int i = 0; i < nBs; i++) {
      edge_len_Bs[i] = (int64_t *)CTF_int::alloc(sizeof(int64_t) * Bs[i]->order);
      for (int j = 0; j < Bs[i]->order; j++) {
        //edge_len_Bs[i][j] = Bs[i]->pad_edge_len[j] / Bs[i]->edge_map[j].calc_phys_phase();
        edge_len_Bs[i][j] = redist_Bs[i]->pad_edge_len[j] / redist_Bs[i]->edge_map[j].calc_phys_phase();
      }
    }
    
    char ** Bs_data = (char **)CTF_int::alloc(sizeof(char *) * nBs);
    const int64_t ** len_Bs = (const int64_t **)CTF_int::alloc(sizeof(int64_t *) * nBs);
    for (int i = 0; i < nBs; i++) {
      IASSERT(!Bs[i]->is_sparse);
      //Bs_data[i] = Bs[i]->data;
      Bs_data[i] = redist_Bs[i]->data;
      len_Bs[i] = Bs[i]->lens;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    stime = MPI_Wtime();
    spA_dnBs_gen_ctr(alpha, &A_csf, sr_A, A->order, idx_A, nBs, Bs_data, sr_Bs, order_Bs, len_Bs, edge_len_Bs, rev_idx_map, idx_max, func);
    MPI_Barrier(MPI_COMM_WORLD);
    etime = MPI_Wtime();
    if (A->wrld->rank == 0) printf("spA_dnBs_gen_ctr() total time: %1.2lf\n", (etime - stime));

    stime = MPI_Wtime();
    {
      // int red_len = T->wrld->np / phys_phase[op_mode];
      // if (red_len > 1) {
      // TODO: check this logic once
      int op = nBs - 1;
      int64_t sz = redist_Bs[op]->size;
      // int jr = T->edge_map[op_mode].calc_phys_rank(T->topo);
      int jr = comm_lda[op];
      MPI_Comm cm;
      MPI_Comm_split(A->wrld->comm, jr, A->wrld->rank, &cm);
      int cmr;
      MPI_Comm_rank(cm, &cmr);
      if (cmr == 0) {
        MPI_Reduce(MPI_IN_PLACE, redist_Bs[op]->data, sz, A->sr->mdtype(),
                   A->sr->addmop(), 0, cm);
      } else {
        MPI_Reduce(redist_Bs[op]->data, NULL, sz, A->sr->mdtype(),
                   A->sr->addmop(), 0, cm);
        std::fill(redist_Bs[op]->data, redist_Bs[op]->data + sz,
                  *((dtype *)A->sr->addid()));
      }
      MPI_Comm_free(&cm);
    }
    Bs[nBs-1]->set_zero();
    std::string s;
    for (int j = 0; j < order_Bs[nBs-1]; j++) {
      s += 'a' + j;
    }
    Bs[nBs-1]->operator[](s.data()) += redist_Bs[nBs-1]->operator[](s.data());
    etime = MPI_Wtime();
    if (A->wrld->rank == 0) printf("output redistribution total time: %1.2lf\n", (etime - stime));

    free(phys_phase);
    for (int i = 0; i < nBs; i++) {
      cdealloc(edge_len_Bs[i]);
      delete redist_Bs[i];
      cdealloc(rev_idx_map[i]);
    }
    cdealloc(rev_idx_map[nBs]);
    cdealloc(rev_idx_map);
    cdealloc(edge_len_Bs);
    cdealloc(Bs_data);
    cdealloc(len_Bs);
    free(redist_Bs);
  }

  template<typename dtype>
  void gen_contraction<dtype>::gen_inv_idx(int                   order_A,
                                           int const *           idx_A,
                                           int                   nBs,
                                           int *                 order_Bs,
                                           const int * const *   idx_Bs,
                                           int *                 order_tot,
                                           int ***               idx_arr)
  {
    int64_t i, j, dim_max;

    dim_max = -1;
    for (i = 0; i < order_A; i++) {
      if (idx_A[i] > dim_max) dim_max = idx_A[i];
    }
    for (j = 0; j < nBs; j++) {
      for (i = 0; i < order_Bs[j]; i++) {
        if (idx_Bs[j][i] > dim_max) dim_max = idx_Bs[j][i];
      }
    }
    
    dim_max++;
    *order_tot = dim_max;
    
    // nBs + A
    *idx_arr = (int **)CTF_int::alloc(sizeof(int *) * (nBs + 1));
    for (j = 0; j < (nBs + 1); j++) {
      (*idx_arr)[j] = (int *)CTF_int::alloc(sizeof(int) * dim_max);
      std::fill((*idx_arr)[j], (*idx_arr)[j]+dim_max, -1);
    }

    for (j = 0; j < nBs; j++) {
      for (i = 0; i < order_Bs[j]; i++) {
        (*idx_arr)[j][idx_Bs[j][i]] = i;
      }
    }

    for (i = 0; i < order_A; i++){
      (*idx_arr)[nBs][idx_A[i]] = i;
    }
  }


  template<typename dtype>
  void gen_contraction<dtype>::traverse_CSF(CSF<dtype> *          A_tree, 
                                            int                   level, 
                                            int64_t               pt)
  {
    int64_t idx_idim = A_tree->get_idx(level, pt);
    std::cout << "level:idx_idim: " << level << ":" << idx_idim << " ";
    if (level == 0) {
      double dt_AB = A_tree->get_data(pt);
      std::cout << "traverse tree dt_AB: " << dt_AB << std::endl;
      return;
    }
    int64_t imax = A_tree->num_children(level, pt);
    std::cout << "imax: " << imax << " ";
    int64_t child_pt = A_tree->get_child_ptr(level, pt);
    for (int64_t i = child_pt; i < (child_pt + imax); i++) {
      traverse_CSF(A_tree, (level-1), i);
    }
  }
}
