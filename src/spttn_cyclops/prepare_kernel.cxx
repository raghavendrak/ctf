#include "prepare_kernel.h"
#include "../redistribution/nosym_transp.h"
#include "../scaling/strp_tsr.h"
#include "../mapping/mapping.h"
#include "../mapping/distribution.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "execute_kernel.h"
#include "../contraction/sym_seq_ctr.h"
#include "../contraction/spctr_comm.h"
#include "../contraction/ctr_tsr.h"
#include "../contraction/ctr_offload.h"
#include "../contraction/ctr_2d_general.h"
#include "../contraction/spctr_offload.h"
#include "../contraction/spctr_2d_general.h"
#include "../symmetry/sym_indices.h"
#include "../symmetry/symmetrization.h"
#include "../redistribution/nosym_transp.h"
#include "../redistribution/redist.h"
#include "../sparse_formats/coo.h"
#include "../sparse_formats/csr.h"
#include "csf.h"
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include "../interface/tensor.h"

namespace CTF_int {
  using namespace CTF;
  template class spttn_contraction<double>;
  //template class spttn_contraction<int>;
  //template class spttn_contraction<int64_t>;

  template<typename dtype>
  spttn_contraction<dtype>::~spttn_contraction()
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
  spttn_contraction<dtype>::spttn_contraction(spttn_contraction<dtype> const & other)
  {
    // TODO: fill up the copy constructor
  }

  template<typename dtype>
  spttn_contraction<dtype>::spttn_contraction(tensor *                  A_,
                                          int const *               idx_A_,
                                          tensor **                 Bs_,
                                          int                       nBs_,
                                          const int * const *       idx_Bs_,
                                          const std::string *       terms_,
                                          int                       nterms_,
                                          const std::string &       sindex_order_,
                                          bool                      retain_op_,
                                          tensor **                 redis_op_,
                                          char const *              alpha_,
                                          bivar_function const *    func_) 
  {
    // TODO: might not need this function
    IASSERT(0);
  }
   
  template<typename dtype>
  spttn_contraction<dtype>::spttn_contraction(tensor *                  A_,
                                          char const *              cidx_A,
                                          tensor **                 Bs_,
                                          int                       nBs_,
                                          const char * const *      cidx_Bs,
                                          bool                      retain_op_,
                                          tensor **                 redis_op_,
                                          char const *              alpha_,
                                          bivar_function const *    func_) 
  {
    A = A_;
    Bs = Bs_;
    nBs = nBs_;
    // nBs is number of Bs including the ouput. The number of input tensors including A is nBs
    nterms = nBs-1;
    retain_op = retain_op_;
    redis_op = redis_op_;
    func = func_;
    alpha = alpha_;

    sr_A = A->sr;
    order_A = A->order;
    sr_Bs = (const algstrct **) alloc(sizeof(algstrct *) * nBs);
    for (int i = 0; i < nBs; i++) sr_Bs[i] = Bs[i]->sr;

    order_Bs = (int *)alloc(sizeof(int) * nBs);
    for (int i = 0; i < nBs; i++) order_Bs[i] = Bs[i]->order;

    spttn_conv_idx(A->order, cidx_A, &idx_A, nBs, order_Bs, cidx_Bs, &idx_Bs);

    if (A->wrld->rank == 0) {
      for (int i = 0; i < order_A; i++) {
        std::cout << "idx_A[" << i << "] = " << idx_A[i] << std::endl;
      }
      for (int j = 0; j < nBs; j++) {
        for (int i = 0; i < order_Bs[j]; i++) {
          std::cout << "idx_Bs[" << j << "][" << i << "] = " << idx_Bs[j][i] << std::endl;
        }
      }
    }
    // TODO: duplicate code; move it to a function 
    int dim_max = -1;
    for (int i = 0; i < A->order; i++) {
      if (idx_A[i] > dim_max) dim_max = idx_A[i];
    }
    for (int j = 0; j < nBs; j++) {
      for (int i = 0; i < order_Bs[j]; i++) {
        if (idx_Bs[j][i] > dim_max) dim_max = idx_Bs[j][i];
      }
    }
    dim_max++;
    num_indices = dim_max;
    
    terms = (contraction_terms<dtype> *)CTF_int::alloc(sizeof(contraction_terms<dtype>) * nterms);
    for (int i = 0; i < nterms; i++) {
      new(terms + i) contraction_terms<dtype>(dim_max, nBs);
    }
    select_cp_io_populate_terms(cidx_A, cidx_Bs, nterms, terms, order_A, idx_A, nBs, order_Bs, idx_Bs, num_indices, A->wrld->rank);
  }

  template<typename dtype>
  spttn_contraction<dtype>::spttn_contraction(tensor *                  A_,
                                          char const *              cidx_A,
                                          tensor **                 Bs_,
                                          int                       nBs_,
                                          const char * const *      cidx_Bs,
                                          const std::string *       sterms,
                                          int                       nterms_,
                                          const std::string *       sindex_order,
                                          bool                      retain_op_,
                                          tensor **                 redis_op_,
                                          char const *              alpha_,
                                          bivar_function const *    func_) 
  {
    A = A_;
    Bs = Bs_;
    nBs = nBs_;
    nterms = nterms_;
    retain_op = retain_op_;
    redis_op = redis_op_;
    func = func_;
    alpha = alpha_;

    sr_A = A->sr;
    order_A = A->order;
    sr_Bs = (const algstrct **) alloc(sizeof(algstrct *) * nBs);
    for (int i = 0; i < nBs; i++) sr_Bs[i] = Bs[i]->sr;

    order_Bs = (int *)alloc(sizeof(int) * nBs);
    for (int i = 0; i < nBs; i++) order_Bs[i] = Bs[i]->order;
    
    spttn_conv_idx(A->order, cidx_A, &idx_A, nBs, order_Bs, cidx_Bs, &idx_Bs);

    // TODO: duplicate code; move it to a function 
    int dim_max = -1;
    for (int i = 0; i < A->order; i++) {
      if (idx_A[i] > dim_max) dim_max = idx_A[i];
    }
    for (int j = 0; j < nBs; j++) {
      for (int i = 0; i < order_Bs[j]; i++) {
        if (idx_Bs[j][i] > dim_max) dim_max = idx_Bs[j][i];
      }
    }
    dim_max++;
    num_indices = dim_max;
    
    terms = (contraction_terms<dtype> *)CTF_int::alloc(sizeof(contraction_terms<dtype>) * nterms);
    for (int i = 0; i < nterms; i++)
      new(terms + i) contraction_terms<dtype>(dim_max, nBs);
    for (int i = 0; i < nterms; i++) {
      std::cout << "sterms[" << i << "] = " << sterms[i] << std::endl;
    }
    term_idx<dtype>(cidx_A, cidx_Bs, sterms, nterms, terms, order_A, idx_A, nBs, order_Bs, idx_Bs); 

    index_order = (int *)CTF_int::alloc(sizeof(int) * num_indices);
    std::fill_n(index_order, num_indices, -1);
    for (int i = 0; i < nterms; i++) {
      index_order_idx<dtype>(cidx_A, cidx_Bs, sindex_order[i], terms[i].index_order, terms[i].rev_index_order, order_A, idx_A, nBs, order_Bs, idx_Bs);
      terms[i].index_order_sz = sindex_order[i].size();
      if (A->wrld->rank == 0) {
        if (terms[i].index_order_sz < num_indices) {
          IASSERT(terms[i].index_order[terms[i].index_order_sz] == (num_indices));
        }
      }
    }
  }

  template<typename dtype>
  spttn_contraction<dtype>::spttn_contraction(tensor *                  A_,
                                              char const *              cidx_A,
                                              tensor **                 Bs_,
                                              int                       nBs_,
                                              const char * const *      cidx_Bs,
                                              const std::string *       sterms,
                                              int                       nterms_,
                                              const std::string &       sindex_order,
                                              bool                      retain_op_,
                                              tensor **                 redis_op_,
                                              char const *              alpha_,
                                              bivar_function const *    func_) {
    // assert failure: global index order deprecated
    IASSERT(0);
  }
  
  template<typename dtype>
  void spttn_contraction<dtype>::execute()
  {
    Tensor<dtype> ** redist_Bs = (Tensor<dtype>**)malloc(sizeof(Tensor<dtype>*) * nBs);
    int64_t comm_lda[nBs];
    
    // TODO: should move this rev_idx_map calculation, and compute it only once
    int ** rev_idx_map;
    gen_inv_idx<dtype>(num_indices, &rev_idx_map, order_A, idx_A, nBs, order_Bs, idx_Bs);
    double stime;
    double etime;
    double tot_time = 0;

    //distribute_operands(rev_idx_map, redist_Bs);
    int dten[nBs];
    MPI_Barrier(MPI_COMM_WORLD);
    stime = MPI_Wtime();
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
        // if output is sparse
        if(Bs[i]->is_sparse) {
          IASSERT(i == nBs-1);
          redist_Bs[i] = nullptr;
          continue;
        }
        // handle matrices as special case: need to fix this
        dten[i] = -1;
        if (order_Bs[i] == 2 /*&& i < nBs-1*/) {
          for (int j = 0; j < order_A; j++) {
            int oa = idx_A[j];
            if (phys_phase[oa] == 1 && rev_idx_map[i][oa] != -1){
              IASSERT(rev_idx_map[i][oa] != -1);
              if (A->wrld->np == 1){
                redist_Bs[i] = (Tensor<dtype> *)Bs[i];
                dten[i] = 0;
                break;
              }
            }
          }
          if (dten[i] == 0) {
            continue;
          }
        }

        char tBs_idx[order_Bs[i]];
        //  for each B, check if the idx is along A (mode)
        comm_lda[i] = 0;
        for (int j = 0; j < order_Bs[i]; j++) {
          // for each idx_B check if it is along tensor A or should be present
          // on all processes rev_idx_map[nBs] stores rev_idx_map for A
          int mode = rev_idx_map[nBs][idx_Bs[i][j]];
          if (mode == -1 || phys_phase[mode] == 1) {
            tBs_idx[j] = 'a' + j - 32;
          } 
          else {
            int topo_dim = A->edge_map[mode].cdt;
            tBs_idx[j] = par_idx[topo_dim];
            comm_lda[i] = comm_lda[i] * A->topo->dim_comm[topo_dim].np + A->topo->dim_comm[topo_dim].rank;
          }
        }
        std::string name_B("redist_Bs_");
        std::ostringstream s;
        s << i;
        name_B += s.str();
        Tensor<dtype> *t = new Tensor<dtype>(
            Bs[i]->order, false, Bs[i]->lens, Bs[i]->sym, *A->wrld, tBs_idx,
            par[par_idx], Idx_Partition(), name_B.data(), 0, *A->sr);
        redist_Bs[i] = t;

        if (order_Bs[i] == 2 /*&& i < nBs-1*/) {
          for (int j = 0; j < order_A; j++) {
            int oa = idx_A[j];
            if (phys_phase[oa] == 1 && rev_idx_map[i][oa] != -1){
              IASSERT(rev_idx_map[i][oa] != -1);
              if (A->wrld->np != 1){
                int64_t npairs;
                Bs[i]->allread(&npairs, redist_Bs[i]->data, true);
                dten[i] = 1;
                break;
              }
            }
          }
          if (dten[i] == 1) {
            continue;
          }
        }
      }

      for (int i = 0; i < nBs - 1; i++) {
        if (dten[i] == 0 || dten[i] == 1) continue;
        std::string s;
        for (int j = 0; j < order_Bs[i]; j++) {
          s += 'a' + j;
        }
        redist_Bs[i]->operator[](s.data()) += Bs[i]->operator[](s.data());
        CTF_int::CommData cmdt(A->wrld->rank, comm_lda[i], A->wrld->cdt);
        cmdt.bcast(redist_Bs[i]->data, redist_Bs[i]->size, A->sr->mdtype(), 0);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    etime = MPI_Wtime();
    tot_time += etime - stime;
    if (A->wrld->rank == 0) printf("input redistribution time: %1.2lf\n", (etime - stime));

    int64_t npair;
    Pair<dtype> *pairs;
    if (A->is_sparse) {
      pairs = (Pair<dtype>*)A->data;
      npair = A->nnz_loc;
    } else {
      IASSERT(0);
    }
    int * phys_phase = (int *) malloc(A->order * sizeof(int));
    for (int i = 0; i < A->order; i++) {
      phys_phase[i] = A->edge_map[i].calc_phys_phase();
    }
    
    //MPI_Barrier(MPI_COMM_WORLD);
    stime = MPI_Wtime();
    CTF_int::CSF<dtype> A_csf(npair, pairs, A->order, A->lens, phys_phase);
    MPI_Barrier(MPI_COMM_WORLD);
    etime = MPI_Wtime();
    if (A->wrld->rank == 0) printf("tree construction time: %1.2lf\n", (etime - stime));

    if (Bs[nBs-1]->is_sparse) {
      // allocate the sparse output tensor in the same tree structure as the input tensor
      IASSERT(0);
    }
    
    stime = MPI_Wtime();
    int64_t ** edge_len_Bs = (int64_t **)CTF_int::alloc(sizeof(int64_t *) * nBs);
    for (int i = 0; i < nBs; i++) {
      if (Bs[i]->is_sparse) {
        edge_len_Bs[i] = nullptr;
        continue;
      }
      edge_len_Bs[i] = (int64_t *)CTF_int::alloc(sizeof(int64_t) * Bs[i]->order);
      for (int j = 0; j < Bs[i]->order; j++) {
        edge_len_Bs[i][j] = redist_Bs[i]->pad_edge_len[j] / redist_Bs[i]->edge_map[j].calc_phys_phase();
      }
    }
    char ** Bs_data = (char **)CTF_int::alloc(sizeof(char *) * (nBs + nterms));
    const int64_t ** len_Bs = (const int64_t **)CTF_int::alloc(sizeof(int64_t *) * nBs);
    for (int i = 0; i < nBs; i++) {
      if (Bs[i]->is_sparse) {
        Bs_data[i] = nullptr;
        len_Bs[i] = nullptr;
        continue;
      }
      Bs_data[i] = redist_Bs[i]->data;
      len_Bs[i] = Bs[i]->lens;
    }

    int64_t len_idx[num_indices];
    for (int i = 0; i < num_indices; i++) {
      // Assumption: Indices have idx = (0, 1, 2, ..., num_indices - 1)
      int idx = i;
      int64_t edge_len_idx_B = -1;
      int pos_idx_inA = rev_idx_map[nBs][idx];
      if (pos_idx_inA != -1) {
        // NOTE: 2X2X2 tensor on 3 processes. index i (which is idx=0) is distributed across 2 processes, but the phys_phase[idx] is calculated as 3, and pad_edge_len is 3
        edge_len_idx_B = A->pad_edge_len[idx] / phys_phase[idx];
      }
      else {
        for (int j = 0; j < (nBs-1); j++) {
          int pos_idx_inB = rev_idx_map[j][idx];      
          if (pos_idx_inB != -1) {
            if (edge_len_idx_B == -1) {
              edge_len_idx_B = edge_len_Bs[j][pos_idx_inB];
            }
            else {
              if (edge_len_idx_B != edge_len_Bs[j][pos_idx_inB]) {
                // idx dimensions don't match
                IASSERT(0);
              }
            }
          }
        }
      }
      len_idx[idx] = edge_len_idx_B;
      // NOTE: len_idx[idx] might be 1 if the idx is not contracted. Buffering for non contracted indices is not supported with this implementation - change to include A->lens
    }
    int active_terms_buffer[nterms];
    // TODO: can use std::iota
    for (int i = 0; i < nterms; i++) {
      active_terms_buffer[i] = i;
    }

    bool transpose = false;
    allocate_buffer<dtype>(len_idx, num_indices, nterms, terms, A->wrld->rank);
    id_inner_ids<dtype>(num_indices, nterms, active_terms_buffer, 0, nterms, terms, A->wrld->rank);
    process_inner_ids<dtype>(&transpose, rev_idx_map, len_idx, terms, nterms, num_indices, order_A, idx_A, nBs, order_Bs, idx_Bs, A->wrld->rank);
    prepare_blas_kernels<dtype>(edge_len_Bs, num_indices, len_idx, rev_idx_map, nterms, nBs, terms, order_Bs, idx_Bs, A->wrld->rank);
    
    for (int i = 0; i < nterms; i++) {
      contraction_terms<dtype> & term = terms[i];
      std::fill_n(term.break_rec_idx, num_indices+1, RECURSIVE_LOOP);
      if (term.blas_kernel == SCALAR) {
        int idx = term.index_order[term.index_order_sz-1];
        term.break_rec_idx[idx] = SCALAR;
      }
      // TODO: else-if not needed
      else if (term.blas_kernel == RECURSIVE_LOOP) {
        int idx = term.index_order[term.index_order_sz-1];
        //std::cout << "Chosen idx: " << idx << std::endl;
        term.break_rec_idx[idx] = RECURSIVE_LOOP;
      }
      else {
        IASSERT(term.blas_kernel != SCALAR);
        IASSERT(term.blas_kernel != -1);
        term.break_rec_idx[term.blas_idx] = term.blas_kernel;
      }
    }
    // Tie intermediate tensors as Bs
    for (int i = 0; i < nterms; i++) {
      Bs_data[nBs+i] = (char *)terms[i].tbuffer;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    etime = MPI_Wtime();
    tot_time += etime - stime;
    if (A->wrld->rank == 0) printf("preamble time: %1.2lf\n", (etime - stime));

    // print the terms
    std::cout << "--------------------------------------------" << std::endl;
    for (int i = 0; i < nterms; i++) {
      contraction_terms<dtype> & term = terms[i];
      if (A->wrld->rank == 0) {
        std::cout << "blas_kernel: " << enumToStr<dtype>(static_cast<CTF_int::BREAK_REC>(term.blas_kernel)) << std::endl;
        std::cout << "blas_idx: " << term.blas_idx << std::endl;
        std::cout << "reset_idx: " << term.reset_idx << std::endl;
      }
    }
    std::cout << "--------------------------------------------" << std::endl;
    stime = MPI_Wtime();
    spA_dnBs_gen_ctr(alpha, &A_csf, sr_A, A->order, idx_A, nBs, Bs_data, sr_Bs, order_Bs, len_Bs, edge_len_Bs, len_idx, idx_Bs, rev_idx_map, num_indices, terms, nterms, func);
    MPI_Barrier(MPI_COMM_WORLD);
    etime = MPI_Wtime();
    tot_time += etime - stime;
    if (A->wrld->rank == 0) printf("spA_dnBs_gen_ctr() total time: %1.2lf\n", (etime - stime));

    if (!Bs[nBs-1]->is_sparse) {
      stime = MPI_Wtime();
      if ((dten[nBs-1] == -1 || dten[nBs-1] == 1)) {
        int op = nBs - 1;
        int64_t sz = redist_Bs[op]->size;
        int jr = comm_lda[op];
        MPI_Comm cm;
        MPI_Comm_split(A->wrld->comm, jr, A->wrld->rank, &cm);
        int cmr;
        MPI_Comm_rank(cm, &cmr);
        if (cmr == 0) {
          MPI_Reduce(MPI_IN_PLACE, redist_Bs[op]->data, sz, A->sr->mdtype(),
                    A->sr->addmop(), 0, cm);
        } 
        else {
          MPI_Reduce(redist_Bs[op]->data, NULL, sz, A->sr->mdtype(),
                    A->sr->addmop(), 0, cm);
          std::fill(redist_Bs[op]->data, redist_Bs[op]->data + sz,
                    *((dtype *)A->sr->addid()));
        }
        MPI_Comm_free(&cm);
        Bs[nBs-1]->set_zero();
        std::string s;
        for (int j = 0; j < order_Bs[nBs-1]; j++) {
          s += 'a' + j;
        }
        Bs[nBs-1]->operator[](s.data()) += redist_Bs[nBs-1]->operator[](s.data());
      }
      else if (dten[nBs-1] == 1) {
        std::cout << "Not broadcasting output" << std::endl;
        Bs[nBs-1]->set_zero();
        Bs[nBs-1]->operator[]("ij") += redist_Bs[nBs-1]->operator[]("ij");
      }
      else {
        IASSERT(Bs[nBs-1] == redist_Bs[nBs-1]);
      }
      etime = MPI_Wtime();
      tot_time += etime - stime;
      if (A->wrld->rank == 0) printf("output redistribution total time: %1.2lf\n", (etime - stime));
    }
    else {
      IASSERT(0);
    }
    if (A->wrld->rank == 0) printf("total time to calculate: %1.2lf\n", (tot_time));

    free(phys_phase);
    for (int i = 0; i < nBs; i++) {
      if (Bs[i]->is_sparse) {
        continue;
      }
      cdealloc(edge_len_Bs[i]);
      if (dten[i] == -1 || dten[i] == 1) {
        if (retain_op == true && i == nBs-1) {
          *redis_op = redist_Bs[i];
        }
        else {
          delete redist_Bs[i];
        }
      }
      else if (retain_op == true && i == nBs-1) {
        *redis_op = redist_Bs[i];
      }
      cdealloc(rev_idx_map[i]);
    }
    cdealloc(rev_idx_map[nBs]);
    cdealloc(rev_idx_map);
    cdealloc(edge_len_Bs);
    cdealloc(Bs_data);
    cdealloc(len_Bs);
    free(redist_Bs);
    for (int i = 0; i < nterms; i++) {
      if (terms[i].tbuffer_sz != -1) {
        cdealloc(terms[i].tbuffer);
      }
    }
  }

  template<typename dtype>
  void spttn_contraction<dtype>::traverse_CSF(CSF<dtype> *          A_tree, 
                                            int                   level, 
                                            int64_t               pt)
  {
    int64_t idx_idim = A_tree->get_idx(level, pt);
    if (level == 0) {
      double dt_AB = A_tree->get_data(pt);
      return;
    }
    int64_t imax = A_tree->num_children(level, pt);
    int64_t child_pt = A_tree->get_child_ptr(level, pt);
    for (int64_t i = child_pt; i < (child_pt + imax); i++) {
      traverse_CSF(A_tree, (level-1), i);
    }
  }
}
