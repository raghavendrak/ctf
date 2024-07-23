#ifndef __PREPARE_KERNEL_H__
#define __PREPARE_KERNEL_H__

#include <assert.h>
#include <climits>
#include <bitset>
#include "../contraction/ctr_tsr.h"
#include "execute_kernel.h"
#include "cp_io.h"

namespace CTF_int {
  class tensor; 
  class topology; 
  class distribution;
  class mapping;
  class debug_spttn_cyclops;
  
  template<typename dtype>
  class contraction_terms;
  
  template<typename dtype>
  class CSF;

  /**
   * \brief class for execution distributed spttn_contraction of tensors using multilinear kernel
   */
  template <typename dtype>
  class spttn_contraction {
    public:
      /** \brief main sparse tensor */
      tensor * A;
      /** \brief operand tensors (output, input) */
      tensor ** Bs;
      /** \brief number of operand tensors (output, input) */
      int nBs;
      /** \brief order of operand tensors (output, input) */
      int * order_Bs;
      /** \brief order of main sparse tensor */
      int order_A;

      /** \brief scaling of A*B */
      char const * alpha;
    
      /** \brief indices of main tensor */
      int * idx_A;
      /** \brief indices of operand tensors */
      int ** idx_Bs;
      /** \brief function to execute on elements */
      // TODO: change it to multivar_function
      bivar_function const * func;

      /** \brief terms for contraction */
      contraction_terms<dtype> * terms;
      /** \brief number of contraction terms */
      int nterms;
      /** \brief contraction order by index */
      int * index_order;
      /** \brief maximum intermediate tensor dimension */
      int max_buf_dim;
      /** \brief number of indices */
      int num_indices; 
      /** \brief if true, do not redistribute the output tensor */
      bool retain_op;
      tensor ** redis_op;

      algstrct const * sr_A;
      algstrct const ** sr_Bs;

      /** \brief lazy constructor */
      spttn_contraction(){ idx_A = NULL; idx_Bs = NULL; order_Bs=NULL; alpha=NULL; };
      
      /** \brief destructor */
      ~spttn_contraction();

      /** \brief copy constructor \param[in] other object to copy */
      spttn_contraction(spttn_contraction const & other);
     
      /**
       * \brief constructor definining multilinear spttn_contraction with custom function
       * \param[in] A main tensor
       * \param[in] idx_A indices of main tensor
       * \param[in] Bs operand tensors
       * \param[in] idx_Bs indices of operand tensors 
       * \param[in] alpha scaling factor (can be NULL) alpha * A[idx_A] * B[i][idx_B];
       * \param[in] func custom elementwise function 
                      func(A[idx_A],B[i][idx_B])
       */
      spttn_contraction(tensor *                  A,
                      int const *               idx_A,
                      tensor **                 Bs,
                      int                       nBs,
                      const int * const *       idx_Bs,
                      const std::string *       terms,
                      int                       nterms,
                      const std::string &       index_order,
                      int                       max_buf_dim,
                      bool                      retain_op=false,
                      tensor **                 redis_op=nullptr,
                      char const *              alpha=NULL,
                      bivar_function const *    func=NULL);
      spttn_contraction(tensor *                  A,
                      char const *              idx_A,
                      tensor **                 Bs,
                      int                       nBs,
                      const char * const *      idx_Bs,
                      const std::string *       terms,
                      int                       nterms,
                      const std::string &       index_order,
                      int                       max_buf_dim,
                      bool                      retain_op=false,
                      tensor **                 redis_op=nullptr,
                      char const *              alpha=NULL,
                      bivar_function const *    func=NULL);
      spttn_contraction(tensor *                  A,
                      char const *              idx_A,
                      tensor **                 Bs,
                      int                       nBs,
                      const char * const *      idx_Bs,
                      const std::string *       terms,
                      int                       nterms,
                      const std::string *       index_order,
                      int                       max_buf_dim,
                      bool                      retain_op=false,
                      tensor **                 redis_op=nullptr,
                      char const *              alpha=NULL,
                      bivar_function const *    func=NULL);
      spttn_contraction(tensor *                  A,
                      char const *              idx_A,
                      tensor **                 Bs,
                      int                       nBs,
                      const char * const *      idx_Bs,
                      int                       max_buf_dim,
                      bool                      retain_op=false,
                      tensor **                 redis_op=nullptr,
                      char const *              alpha=NULL,
                      bivar_function const *    func=NULL);

      /** \brief run spttn_contraction */
      void execute();
  
      void traverse_CSF(CSF<dtype> *          A_tree, 
                        int                   level, 
                        int64_t               pt);

  };
  
  enum BLAS_OPERANDS {MAIN_TENSOR, PREV_TERM_BUF, CURRENT_TERM_BUF, INP_B, INTERMEDIATE_TENSOR};
  enum BREAK_REC {NOT_SET, SCALAR, SPARSE, SPARSE_xAXPY, SPARSE_xAXPY_OP_NOT_BUFFER, DENSE, DENSE_xAXPY_2D, DENSE_xAXPY_3D, DENSE_STRIDED, DENSE_3D, DENSE_3D_TO_xAXPY, xAXPY, xVEC_MUL, xGER, xDOT, xGER_TO_xAXPY, xGEMV, RECURSIVE_LOOP};

  template <typename dtype>
  const char* enumToStr(BREAK_REC br) {
    switch (br) {
        case NOT_SET: return "NOT_SET";
        case SCALAR: return "SCALAR";
        case SPARSE: return "SPARSE";
        case SPARSE_xAXPY: return "SPARSE_xAXPY";
        case SPARSE_xAXPY_OP_NOT_BUFFER: return "SPARSE_xAXPY_NOT_BUFFER";
        case DENSE: return "DENSE";
        case DENSE_xAXPY_2D: return "DENSE_xAXPY_2D";
        case DENSE_xAXPY_3D: return "DENSE_xAXPY_3D";
        case DENSE_STRIDED: return "DENSE_STRIDED";
        case DENSE_3D: return "DENSE_3D";
        case xAXPY: return "xAXPY";
        case xVEC_MUL: return "xVEC_MUL";
        case xGER: return "xGER";
        case xDOT: return "xDOT";
        case xGEMV: return "xGEMV";
        case RECURSIVE_LOOP: return "RECURSIVE_LOOP";
        default: return "Unknown value";
    }
  }

  template <typename dtype>
  void prepare_blas_kernels(const int64_t * const *     edge_len_Bs,
                            int                         idx_max,
                            const int64_t *             len_idx,
                            const int * const *         rev_idx_map,
                            int                         nterms,
                            int                         nBs,
                            contraction_terms<dtype> *  terms,
                            int *                       order_Bs,
                            int **                      idx_Bs,
                            const int                   rank)
  {
    debug_spttn_cyclops spttn_print;
    spttn_print << "----------------------prepare_blas_kernels----------------------" << std::endl;
    int64_t ** lda_Bs;
    lda_Bs = (int64_t **) CTF_int::alloc(sizeof(int64_t *) * (nBs+nterms));
    calc_ldas(nBs, order_Bs, edge_len_Bs, idx_Bs, terms, nterms, idx_max, lda_Bs);
    for (int i = 0; i < nterms; i++) {
      spttn_print << "prepare blas kernels: term_id: " << i << std::endl;
      contraction_terms<dtype> & term = terms[i];
      switch(term.blas_kernel) {
        case NOT_SET: {
          // use all the cases of RECURSIVE_LOOP
          term.blas_kernel = RECURSIVE_LOOP;
          if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "NOT_SET" << std::endl;
        }
        case SCALAR: {
          // use all the cases of SCALAR; need to have blas_kernel set to RECUSIVE_LOOP to handle when executing the contraction
          if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "SCALAR" << std::endl;
        }
        case RECURSIVE_LOOP: {
          // TODO: i==0 is dependent on contracting the tree first; do away with this dependency by just checking if the main sparse tensor is in the term
          if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "RECURSIVE_LOOP" << std::endl;
          if (i == 0) {
            // two dense factors are contracted in the first term
            if (term.Bs_in_term[nBs] == false) {
              term.ALPHA = -1;
              term.X = -1;
              for (int j = 0; j < nBs - 1; j++) {
                if (term.Bs_in_term[j] == true && term.ALPHA == -1) {
                  term.ALPHA = j;
                }
                else if (term.Bs_in_term[j] == true) {
                  term.X = j;
                  break;
                }
              }
              IASSERT(term.ALPHA != -1 && term.X != -1);
              // both are input tensors
              // TODO: can replace the above code with 
              // if(term.blas_B_ids[0] != nBs) { term.ALPHA = term.blas_B_ids[0]; term.X = term.blas_B_ids[1]; }
              IASSERT(term.blas_B_ids[0] == term.ALPHA && term.blas_B_ids[1] == term.X);
            }
            else {
              term.ALPHA = -1;
              for (int j = 0; j < nBs - 1; j++) {
                if (term.Bs_in_term[j] == true) {
                  term.X = j;
                  break;
                }
              }
              // sparse tensor is contracted with an input tensor
              IASSERT(term.blas_B_ids[0] == nBs && term.blas_B_ids[1] == term.X);
            }
            term.Y = nBs + i;
            IASSERT(term.blas_B_ids[2] == term.Y);
          }
          else {
            /*
            term.ALPHA = nBs + term.inp_buf_id;
            term.X = -1;
            for (int j = 0; j < nBs - 1; j++) {
              if (term.Bs_in_term[j] == true) {
                term.X = j;
                break;
              }
            }
            if (term.X == -1) {
              IASSERT(term.Bs_in_term[nBs] == true);
              // interchange: takes care of the case where the sparse tensor is contracted not in the first term
              term.X = nBs + term.inp_buf_id;
              term.ALPHA = -1;
            }
            */
            // term.blas_B_ids[0] == nBs for both sparse tensor and intermediate tensor from term 0 since term.blas_B_ids is set as nBs + i
            // if (term.blas_B_ids[0] == nBs) {
            if (term.Bs_in_term[nBs] == true) {
              spttn_print << "term.blas_B_ids[0]: " << term.blas_B_ids[0] << std::endl;
              IASSERT(term.blas_B_ids[0] == nBs);
              term.ALPHA = -1;
              term.X = term.blas_B_ids[1];
            }
            else {
              term.ALPHA = term.blas_B_ids[0];
              term.X = term.blas_B_ids[1];
            }
            if (i == (nterms - 1)) term.Y = nBs - 1;
            else term.Y = nBs + i;
            IASSERT(term.blas_B_ids[2] == term.Y);
          }
        }
        break;
        case DENSE_3D_TO_xAXPY: {
          // assert failure: currently does not work with two intermediate tensors in the term
          IASSERT(term.i_inp_buf_id < 2);
          spttn_print << "term.i_inp_buf_id: " << term.i_inp_buf_id << std::endl;
          int rev_blas_idx = term.rev_index_order[term.blas_idx];
          int idx = term.index_order[rev_blas_idx+2];
          // if (idx == terms[term.inp_buf_id].idx_tbuffer[0]) {
          IASSERT(term.Bs_in_term[nBs] == false);
          IASSERT(term.blas_B_ids[0]-nBs == term.inp_buf_ids[0]);
          IASSERT(term.blas_B_ids[1] < nBs);
          if (idx == terms[term.blas_B_ids[0]-nBs].idx_tbuffer[0]) {
            term.ALPHA = term.blas_B_ids[1];
            // term.X = term.inp_buf_id + nBs;
            term.X = term.blas_B_ids[0];
            term.INCX = lda_Bs[term.X][idx]; 
          }
          else if (rev_idx_map[term.blas_B_ids[1]][idx] != -1) {
            // term.ALPHA = term.inp_buf_id + nBs;
            term.ALPHA = term.blas_B_ids[0];
            term.X = term.blas_B_ids[1]; 
            term.INCX =lda_Bs[term.X][idx]; 
          }
          else {
            term.blas_kernel = RECURSIVE_LOOP;
            i--;
            break;
          }
          spttn_print << "term.ALPHA: " << term.ALPHA << " term.X: " << term.X << " term.INCX: " << term.INCX << std::endl;
          term.Y = term.blas_B_ids[2];
          term.INCY = lda_Bs[term.Y][idx];
          term.N = len_idx[idx];
          term.blas_idx = idx;
          term.blas_kernel = xAXPY;
        }
        break;
        case DENSE_3D: {
          if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "DENSE_3D" << std::endl;
          // term.blas_kernel = DENSE_3D_TO_xAXPY;
          // i--;
          term.ALPHA = term.blas_B_ids[0];
          term.X = term.blas_B_ids[1];
          term.Y = term.blas_B_ids[2];
        }
        break;
        case xAXPY: {
          if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "xAXPY" << std::endl;
          // assert failure: can use SPARSE_xAXPY instead
          IASSERT(term.Bs_in_term[nBs] == false);
          int idx = term.blas_idx;
          /*
          if (term.blas_ops[0] == INTERMEDIATE_TENSOR) {
            term.ALPHA = nBs + term.inp_buf_id;
          }
          else {
            term.ALPHA = term.blas_B_ids[0];
          }
          term.X = term.blas_B_ids[1];
          if (i != (nterms-1)) {
            term.Y = nBs + i;
            term.INCX = lda_Bs[term.X][idx];
          }
          else {
            term.Y = nBs - 1;
            term.INCX = 1;
          }
          term.INCY = lda_Bs[term.Y][idx];
          term.N = len_idx[idx];
          */
          term.ALPHA = term.blas_B_ids[0];
          term.X = term.blas_B_ids[1];
          term.Y = term.blas_B_ids[2];
          term.INCX = lda_Bs[term.X][idx];
          term.INCY = lda_Bs[term.Y][idx];
          term.N = len_idx[idx];
        }
        break;
        case xGER_TO_xAXPY: {
          // assert failure: currently does not work with two intermediate tensors in the term
          IASSERT(term.i_inp_buf_id < 2);
          int rev_blas_idx = term.rev_index_order[term.blas_idx];
          int idx_X1 = term.index_order[rev_blas_idx];
          int idx_X2 = term.index_order[rev_blas_idx+1];
          // if (idx_X2 == terms[term.inp_buf_id].idx_tbuffer[0]) {
          IASSERT(term.Bs_in_term[nBs] == false);
          IASSERT(term.blas_B_ids[0]-nBs == term.inp_buf_ids[0]);
          IASSERT(term.blas_B_ids[1] < nBs);
          // if (idx_X2 == terms[term.inp_buf_id].idx_tbuffer[0]) {
          if (idx_X2 == terms[term.blas_B_ids[0]-nBs].idx_tbuffer[0]) {
            term.ALPHA = term.blas_B_ids[1];
            // term.X = term.inp_buf_id + nBs;
            term.X = term.blas_B_ids[0];
            term.INCX = lda_Bs[term.X][idx_X2]; 
          }
          else if (idx_X2 == idx_Bs[term.blas_B_ids[1]][0]) {
            // term.ALPHA = term.inp_buf_id + nBs;
            term.ALPHA = term.blas_B_ids[0];
            term.X = term.blas_B_ids[1]; 
            term.INCX =lda_Bs[term.X][idx_X2]; 
          }
          // else if (idx_X2 == terms[term.inp_buf_id].idx_tbuffer[1]) {
          else if (idx_X2 == terms[term.blas_B_ids[0]-nBs].idx_tbuffer[1]) {
            term.ALPHA = term.blas_B_ids[1];
            // term.X = term.inp_buf_id + nBs;
            term.X = term.blas_B_ids[0];
            term.INCX = lda_Bs[term.X][idx_X2]; 
          } 
          else {
            term.blas_kernel = RECURSIVE_LOOP;
            i--;
            break;
          }
          if (i != (nterms-1)) {
            term.Y = nBs + i;
          }
          else {
            term.Y = nBs - 1;
          }
          IASSERT(term.blas_B_ids[2] == term.Y);
          term.INCY = lda_Bs[term.Y][idx_X2];
          term.N = len_idx[idx_X2];
          term.blas_idx = idx_X2;
          term.blas_kernel = xAXPY;
        }
        break;
        case SPARSE_xAXPY: {
          if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "SPARSE_xAXPY" << std::endl;
          if (term.blas_B_ids[0] != nBs) {
            term.blas_kernel = xAXPY;
            i--;
            break;
          }
          /*
          if (term.blas_ops[0] != MAIN_TENSOR) {
            if (term.index_order[3] != 1) {
              term.blas_idx = term.index_order[3];
              term.blas_kernel = xAXPY;
              i--;
              break;
            }
            else {
              term.blas_kernel = RECURSIVE_LOOP;
              i--;
              break;
            }
          }
          */
          term.ALPHA = -1;
          int idx = -1;
          term.X = term.blas_B_ids[1];
          term.Y = term.blas_B_ids[2];
          idx = term.Y >= nBs ? terms[term.Y-nBs].idx_tbuffer[0] : idx_Bs[term.Y][0];
          int idx_X1 = term.X >= nBs ? terms[term.X-nBs].idx_tbuffer[0] : idx_Bs[term.X][0];
          if (idx != idx_X1) {
            term.blas_kernel = RECURSIVE_LOOP;
            i--;
            break;
          }
          term.INCX = lda_Bs[term.X][idx];
          term.INCY = lda_Bs[term.Y][idx];
          IASSERT(term.INCX == 1 && term.INCY == 1);
          term.N = len_idx[idx];
          if (term.Y == nBs - 1) {
            // the output has a sparse index in this case
            term.blas_kernel = SPARSE_xAXPY_OP_NOT_BUFFER;
          }
          /*
          if (i != (nterms-1)) {
            term.X = term.blas_B_ids[1]; // an input tensor
            term.Y = nBs + i; // buffer
            idx = term.idx_tbuffer[0];
            if (idx_Bs[term.X][0] != term.idx_tbuffer[0]) {
              term.blas_kernel = RECURSIVE_LOOP;
              i--;
              break;
            }
          }
          else {
            term.blas_kernel = SPARSE_xAXPY_OP_NOT_BUFFER;
            // an intermediate tensor (one of the other input in the pairwise contraction is the main tensor)
            // term.X = nBs + term.inp_buf_id;
            term.X = nBs + term.inp_buf_ids[0]; 
            IASSERT(term.X == term.blas_B_ids[1]);
            term.Y = nBs - 1;
            idx = idx_Bs[term.Y][0];
            if (idx != terms[term.inp_buf_id].idx_tbuffer[0]) {
              term.blas_kernel = RECURSIVE_LOOP;
              i--;
              break;
            }
          }
          IASSERT(idx != -1);
          term.INCX = lda_Bs[term.X][idx];
          term.INCY = lda_Bs[term.Y][idx];
          IASSERT(term.INCX == 1 && term.INCY == 1);
          term.N = len_idx[idx];
          */
          spttn_print << "done SPARSE_xAXPY" << std::endl;
        }
        break;
        case DENSE_xAXPY_2D: {
          if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "DENSE_xAXPY_2D" << std::endl;
          // assert failure: currently does not work with two intermediate tensors in the term
          IASSERT(term.i_inp_buf_id < 2);
          int idx_X1 = terms[term.blas_B_ids[0]-nBs].idx_tbuffer[0];
          int idx_X2 = terms[term.blas_B_ids[0]-nBs].idx_tbuffer[1];
          IASSERT(term.Bs_in_term[nBs] == false);
          IASSERT(term.blas_B_ids[0]-nBs == term.inp_buf_ids[0]);
          IASSERT(term.blas_B_ids[1] < nBs);
          if (i == (nterms -1) && (idx_X1 != idx_Bs[nBs-1][0] || idx_X2 != idx_Bs[nBs-1][1])) {
            // a stride of idx
            if (idx_X1 == idx_Bs[nBs-1][1] && idx_X2 == idx_Bs[nBs-1][2]) {
              term.INCX = 1;
              term.INCY = len_idx[idx_Bs[nBs-1][0]];
              term.blas_kernel = DENSE_STRIDED;
              term.ALPHA = term.blas_B_ids[1];
              // term.X = term.inp_buf_id + nBs;
              term.X = term.blas_B_ids[0];
              term.N = len_idx[idx_X1] * len_idx[idx_X2];
              term.Y = nBs-1;
            }
            else {
              // TODO: ttmr_o3_allm rkji ksjr ktrs
              term.blas_kernel = RECURSIVE_LOOP;
              i--;
              break;
            }
            break;
          }
          term.ALPHA = term.blas_B_ids[1];
          // term.X = term.inp_buf_id + nBs;
          term.X = term.blas_B_ids[0];
          term.N = len_idx[idx_X1] * len_idx[idx_X2];
          if (i == (nterms-1)) {
            // assert failure: xAXPY_2D with strides is not supported
            IASSERT(idx_X1 == idx_Bs[nBs-1][0] && idx_X2 == idx_Bs[nBs-1][1]);
            term.Y = nBs-1;
          }
          else {
            // assert failure: output is not the final result tensor
            IASSERT(idx_X1 == terms[i].idx_tbuffer[0]);
            IASSERT(idx_X2 == terms[i].idx_tbuffer[1]);
            term.Y = nBs + i;
          }
        }
        break;
        case DENSE_xAXPY_3D: {
          if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "DENSE_xAXPY_3D" << std::endl;
          // assert failure: currently does not work with two intermediate tensors in the term
          IASSERT(term.i_inp_buf_id < 2);
          int idx_X1 = terms[term.blas_B_ids[0]-nBs].idx_tbuffer[0];
          int idx_X2 = terms[term.blas_B_ids[0]-nBs].idx_tbuffer[1];
          int idx_X3 = terms[term.blas_B_ids[0]-nBs].idx_tbuffer[2];
          IASSERT(term.Bs_in_term[nBs] == false);
          IASSERT(term.blas_B_ids[0]-nBs == term.inp_buf_ids[0]);
          IASSERT(term.blas_B_ids[1] < nBs);
          term.ALPHA = term.blas_B_ids[1];
          // term.X = term.inp_buf_id + nBs;
          term.X = term.blas_B_ids[0];
          term.N = len_idx[idx_X1] * len_idx[idx_X2] * len_idx[idx_X3];
          if (i == (nterms-1)) {
            // assert failure: xAXPY_2D with strides is not supported
            IASSERT(idx_X1 == idx_Bs[nBs-1][0] && idx_X2 == idx_Bs[nBs-1][1] && idx_X3 == idx_Bs[nBs-1][2]);
            term.Y = nBs-1;
          }
          else {
            // assert failure: output is not the final result tensor
            IASSERT(0);
          }
        }
        break;
        case xGER: {
          if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "xGER" << std::endl;
          // A <- alpha * x * y^T + A
          term.ALPHA = 1.0;
          int idx_X;
          int idx_Y;
          // output
          term.A = term.blas_B_ids[2];
          if (term.A == nBs - 1) {
            idx_X = idx_Bs[term.A][0];
            idx_Y = idx_Bs[term.A][1];
          }
          else {
            IASSERT(term.blas_B_ids[2] == i + nBs);
            idx_X = terms[i].idx_tbuffer[0];
            idx_Y = terms[i].idx_tbuffer[1];
          }
          // input
          int idx_X1 = term.blas_B_ids[0] >= nBs ? terms[term.blas_B_ids[0]-nBs].idx_tbuffer[0] : idx_Bs[term.blas_B_ids[0]][0];
          int idx_X2 = term.blas_B_ids[1] >= nBs ? terms[term.blas_B_ids[1]-nBs].idx_tbuffer[0] : idx_Bs[term.blas_B_ids[1]][0];
          if (idx_X == idx_X1) {
            if (idx_Y != idx_X2) {
              term.blas_kernel = RECURSIVE_LOOP;
              i--;
              break;
            }
            term.X = term.blas_B_ids[0];
            term.INCX = lda_Bs[term.X][idx_X];
            term.Y = term.blas_B_ids[1];
            term.INCY = lda_Bs[term.Y][idx_Y];
          }
          else if (idx_X == idx_X2) {
            if (idx_Y != idx_X1) {
              term.blas_kernel = RECURSIVE_LOOP;
              i--;
              break;
            }
            term.X = term.blas_B_ids[1];
            term.INCX = lda_Bs[term.X][idx_X];
            term.Y = term.blas_B_ids[0];
            term.INCY = lda_Bs[term.Y][idx_Y];
          }
          else {
            if (rev_idx_map[nBs][idx_X] != -1) {
              term.blas_kernel = RECURSIVE_LOOP;
            }
            else {
              term.blas_kernel = xGER_TO_xAXPY;
            }
            i--;
            break;          
          }
          term.M = len_idx[idx_X];
          term.N = len_idx[idx_Y];
          term.LDA = term.M;
          /*
          if (term.blas_ops[0] == INTERMEDIATE_TENSOR) {
            IASSERT(term.inp_buf_id != -1); 
            if (i == (nterms-1)) {
              term.A = nBs - 1;
              idx_X = idx_Bs[nBs-1][0];
              idx_Y = idx_Bs[nBs-1][1];
            }
            else {
              term.A = i + nBs;
              idx_X = term.idx_tbuffer[0];
              idx_Y = term.idx_tbuffer[1]; 
              spttn_print << "idx_X: " << idx_X << " idx_Y: " << idx_Y << std::endl;
            }
            // if the intermediate tensor of this term's fastest moving index is the same as the input intermediate tensor
            if (idx_X == terms[term.inp_buf_id].idx_tbuffer[0]) {
              if (idx_Y != idx_Bs[term.blas_B_ids[1]][0]) {
                term.blas_kernel = RECURSIVE_LOOP;
                i--;
                break;
              }
              term.X = term.inp_buf_id + nBs;
              term.INCX = lda_Bs[term.X][idx_X];
              term.Y = term.blas_B_ids[1];
              term.INCY = lda_Bs[term.Y][idx_Y];
            }
            else if (idx_X == idx_Bs[term.blas_B_ids[1]][0]) {
              // if the intermediate tensor of this term's fastest moving index is the same as input B
              // assert failure: the first index of the output is not the same as the first index of input B; factor matrix not transposed
              IASSERT(idx_Y == terms[term.inp_buf_id].idx_tbuffer[0]);
              term.X = term.blas_B_ids[1];
              term.INCX = lda_Bs[term.X][idx_X];
              term.Y = term.inp_buf_id + nBs;
              term.INCY = lda_Bs[term.Y][idx_Y];
            }
            else {
              if (rev_idx_map[nBs][idx_X] != -1) {
                term.blas_kernel = RECURSIVE_LOOP;
              }
              else {
                term.blas_kernel = xGER_TO_xAXPY;
              }
              i--;
              break;
            }
            term.M = len_idx[idx_X];
            term.N = len_idx[idx_Y];
            term.LDA = term.M;
          }
          else {
            // assert failure: xGER with two inputs Bs not supported
            IASSERT(0);
          }
          */
        }
        break;
        case xVEC_MUL: {
          if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "xVEC_MUL" << std::endl;
          int idx = term.blas_idx;
          /*
          if (term.blas_ops[0] == INTERMEDIATE_TENSOR) {
            term.ALPHA = nBs + term.inp_buf_id;
          }
          else {
            term.ALPHA = term.blas_B_ids[0];
          }
          */
          term.ALPHA = term.blas_B_ids[0];
          term.X = term.blas_B_ids[1];
          /*
          if (i == (nterms-1)) {
            term.Y = nBs - 1;
          }
          else {
            term.Y = nBs + i;
          }
          */
          term.Y = term.blas_B_ids[2];
          term.INCX = lda_Bs[term.X][idx];
          term.INCY = lda_Bs[term.Y][idx];
          // assert failure: xVEC_MUL with strides is not supported
          IASSERT(term.INCX == 1 && term.INCY == 1);
          term.N = len_idx[idx];
        }
        break;
        case xGEMV: {
          if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "xGEMV" << std::endl;
          // y <- alpha * A * x + beta * y
          term.ALPHA = 1.0;
          int idx_X;
          int idx_Y;
          // output
          term.Y = term.blas_B_ids[2];
          idx_Y = term.Y >= nBs ? terms[term.Y-nBs].idx_tbuffer[0] : idx_Bs[term.Y][0];
          // input
          int idx_X1 = term.blas_B_ids[0] >= nBs ? terms[term.blas_B_ids[0]-nBs].idx_tbuffer[0] : idx_Bs[term.blas_B_ids[0]][0];
          int idx_Y1 = term.blas_B_ids[0] >= nBs ? terms[term.blas_B_ids[0]-nBs].idx_tbuffer[1] : idx_Bs[term.blas_B_ids[0]][1];
          int idx_X2 = term.blas_B_ids[1] >= nBs ? terms[term.blas_B_ids[1]-nBs].idx_tbuffer[0] : idx_Bs[term.blas_B_ids[1]][0];
          int idx_Y2 = term.blas_B_ids[1] >= nBs ? terms[term.blas_B_ids[1]-nBs].idx_tbuffer[1] : idx_Bs[term.blas_B_ids[1]][1];
          if (idx_X1 == idx_Y || idx_Y1 == idx_Y) {
            // first input tensor is A
            term.A = term.blas_B_ids[0];
            if (idx_X1 == idx_Y) {
              if (idx_Y1 != idx_X2) {
                term.blas_kernel = RECURSIVE_LOOP;
                i--;
                break;
              }
              // no transpose
              term.TRANS = 'N';
            }
            if (idx_Y1 == idx_Y) {
              if (idx_X1 != idx_X2) {
                term.blas_kernel = RECURSIVE_LOOP;
                i--;
                break;
              }
              // transpose A
              term.TRANS = 'T';
            }
            term.M = len_idx[idx_X1];
            term.N = len_idx[idx_Y1];
            term.X = term.blas_B_ids[1];
            term.INCX = lda_Bs[term.X][idx_X2];
          }
          else {
            // second input tensor is A
            term.A = term.blas_B_ids[1];
            if (idx_X2 == idx_Y) {
              if (idx_Y2 != idx_X1) {
                term.blas_kernel = RECURSIVE_LOOP;
                i--;
                break;
              }
              // no transpose
              term.TRANS = 'N';
            }
            if (idx_Y2 == idx_Y) {
              if (idx_X2 != idx_X1) {
                term.blas_kernel = RECURSIVE_LOOP;
                i--;
                break;
              }
              // transpose A
              term.TRANS = 'T';
            }
            term.M = len_idx[idx_X2];
            term.N = len_idx[idx_Y2];
            term.X = term.blas_B_ids[0];
            term.INCX = lda_Bs[term.X][idx_X1];
          }
          term.LDA = term.M;
          term.INCY = lda_Bs[term.Y][idx_Y];
        }
        break;
        case xDOT: {
          // dot <- x^T * y
          term.ALPHA = term.blas_B_ids[2];
          term.X = term.blas_B_ids[0];
          term.Y = term.blas_B_ids[1];
          term.INCX = lda_Bs[term.X][term.blas_idx];
          term.INCY = lda_Bs[term.Y][term.blas_idx];
          term.N = len_idx[term.blas_idx];
        }
        break;
        default:
          break;
      }
    }

    for (int i = 0; i < nBs; i++) {
      cdealloc(lda_Bs[i]);
    }
    for (int i = 0; i < nterms; i++) {
      if (terms[i].tbuffer_order == -1) continue;
      cdealloc(lda_Bs[i+nBs]);
    }
    cdealloc(lda_Bs);
    spttn_print << "------------------------------------------------------------------" << std::endl;
  }
   
  template <typename dtype>
  void select_blas_kernel(int                         term_id,
                          bool **                     in_term_idx,
                          int  *                      nidx_term,
                          int                         num_idx,
                          const int64_t *             len_idx,
                          contraction_terms<dtype> *  terms,
                          int                         num_indices,
                          int **                      idx_Bs,
                          int                         nBs,
                          const int                   rank)
  {
    debug_spttn_cyclops spttn_print;
    // IASSERT(nidx_term[2] > 0);
    contraction_terms<dtype> & term = terms[term_id]; 
    bool recursive_loop = false;
    if (recursive_loop == true) {
      if (term.blas_kernel != SCALAR) {
        term.blas_kernel = RECURSIVE_LOOP;
      }
      return;
    }
    IASSERT(term.inner_idx != -1);
    if (num_idx == 1) {
      if (nidx_term[0] == 0 && nidx_term[1] == 1) {
        if (term.sparse_idx != -1) {
          term.blas_idx = term.sparse_idx;
          term.blas_kernel = SPARSE_xAXPY;
          // term.blas_kernel = RECURSIVE_LOOP;
          if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "SPARSE_xAXPY" << std::endl;
        }
        else {
          term.blas_kernel = xAXPY;
          if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "xAXPY" << std::endl;
        }
      }
      else if (nidx_term[0] == 1 && nidx_term[1] == 1 && nidx_term[2] == 1) {
        int idx_X, idx_Y;
        idx_X = term.blas_B_ids[0] >= nBs ? terms[term.blas_B_ids[0]-nBs].idx_tbuffer[0] : idx_Bs[term.blas_B_ids[0]][0];
        idx_Y = term.blas_B_ids[1] >= nBs ? terms[term.blas_B_ids[1]-nBs].idx_tbuffer[0] : idx_Bs[term.blas_B_ids[1]][0];
        /*
        if (term.blas_ops[0] == INTERMEDIATE_TENSOR) {
          idx_X = terms[term.inp_buf_id].idx_tbuffer[0];
        }
        else {
          idx_X = idx_Bs[term.blas_B_ids[0]][0]; 
        }
        idx_Y = idx_Bs[term.blas_B_ids[1]][0];
        */
        if (idx_X != idx_Y) {
          term.blas_kernel = RECURSIVE_LOOP;
          if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "RECURSIVE_LOOP" << std::endl;
          return;
        }
        term.blas_kernel = xVEC_MUL;
        if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "xVEC_MUL" << std::endl;
      }
      else if (nidx_term[0] == 1 && nidx_term[1] == 1 && nidx_term[2] == 0) {
        term.blas_kernel = xDOT;
        if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "xDOT" << std::endl;
      }
      else {
        // TODO: ttmc_o3_allm rkji rskj trks
        // FIXME: should have been handled in process_inner_ids
        term.blas_kernel = RECURSIVE_LOOP;
      }
    }
    else if (num_idx == 2) {
      if (nidx_term[0] == 1 && nidx_term[1] == 1 && nidx_term[2] == 2) {
        term.blas_kernel = xGER;
        if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "xGER" << std::endl;
      }
      else if (nidx_term[0] == 2 && nidx_term[1] == 0) {
        // TODO: ttmc_o3_allm rkji rskj tkrs
        term.blas_kernel = RECURSIVE_LOOP;
        if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "RECURSIVE_LOOP" << std::endl;
      }
      else if ((nidx_term[0] == 1 && nidx_term[1] == 2 && nidx_term[2] == 1) || (nidx_term[0] == 2 && nidx_term[1] == 1 && nidx_term[2] == 1)) {
        term.blas_kernel = xGEMV;
        if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "xGEMV" << std::endl;
      }
      else {
        term.blas_kernel = RECURSIVE_LOOP;
        if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "RECURSIVE_LOOP" << std::endl;
      }
    }
    else if (num_idx == 3) {
      if (nidx_term[0] == 2 && nidx_term[1] == 1) {
        int j = 0;
        for (; j < num_indices; j++) {
          if (in_term_idx[1][j] == true) {
            break;
          }
        }
        if (in_term_idx[0][j] == false) {
          IASSERT(nidx_term[2] == 3);
          term.dense_idx = -1;
          term.blas_kernel = DENSE_xAXPY_2D;
          if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "DENSE_xAXPY_2D" << std::endl;
        }
        else {
          term.blas_kernel = RECURSIVE_LOOP;
          if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "RECURSIVE_LOOP" << std::endl;
        }
      }
      else if (((nidx_term[0] == 2 && nidx_term[1] == 3) || (nidx_term[0] == 3 && nidx_term[1] == 2)) && nidx_term[2] == 1) {       
        term.blas_kernel = DENSE_3D;
        if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "DENSE_3D" << std::endl;
      }
      else {
        // one possibility nidx_term[0] == 3 && nidx_term[1] == 1 && nidx_term[2] == 3
        term.blas_kernel = DENSE_3D;
        if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "DENSE_3D" << std::endl;
      }
    }
    else if (num_idx == 4) {
      term.blas_kernel = RECURSIVE_LOOP;
      if (rank == 0) spttn_print << "term_id: " << term_id << " blas_kernel: " << "RECURSIVE_LOOP" << std::endl;
    }
    else {
      IASSERT(0);
    }
  } 

  template <typename dtype>
  void process_inner_ids(const int * const *        rev_idx_map,
                         const int64_t *            len_idx,
                         contraction_terms<dtype> * terms,
                         int                        nterms,
                         int                        num_indices,
                         int                        order_A,
                         int *                      idx_A,
                         int                        nBs,
                         int *                      order_Bs,
                         int **                     idx_Bs,
                         const int                  rank)
  {
    debug_spttn_cyclops spttn_print;
    int idx;
    bool ** in_term_idx;
    in_term_idx = (bool **)CTF_int::alloc(sizeof(bool*) * 3);
    for (int i = 0; i < 3; i++) {
      in_term_idx[i] = (bool *)CTF_int::alloc(sizeof(bool) * num_indices);
    }
    int nidx_term[3];

    for (int i = 0; i < nterms; i++) {
      if (terms[i].inner_idx == -1) {
        // reset_idx = -1 because we do not accumulate into the buffer
        terms[i].reset_idx = -1;
        IASSERT(terms[i].dense_idx == -1);
        IASSERT(terms[i].sparse_idx == -1);
        terms[i].blas_kernel = SCALAR;
        // set blas_B_ids
        int inp = 0;
        if (terms[i].Bs_in_term[nBs] == true) terms[i].blas_B_ids[inp++] = nBs;
        for (int k = 0; k < terms[i].i_inp_buf_id; k++) {
          int inp_buf_id = terms[i].inp_buf_ids[k];
          terms[i].blas_B_ids[inp++] = inp_buf_id + nBs;
        }
        for (int k = 0; k < nBs-1; k++) {
          if (terms[i].Bs_in_term[k] == true) {
            terms[i].blas_B_ids[inp++] = k;
          }
        }
        if (i < nterms-1) {
          terms[i].blas_B_ids[inp++] = i + nBs;
        }
        else {
          terms[i].blas_B_ids[inp++] = nBs - 1;
        }
        IASSERT(inp == 3);

        continue; 
      }
      for (int k = 0; k < 3; k++) {
        for (int j = 0; j < num_indices; j++) {
          in_term_idx[k][j] = false;
        }
      }
      int inp = 0;
      for (int j = 0; j < 3; j++) nidx_term[j] = 0;
      // for a term: {(sparse_tensor/inp_B/intermediate) * (inp_B/intermediate) -> (intermediate/op_B)}
      // follow the convention of how nidx_term[] was populated: main tensor followed by intermediate tensors followed by input tensors followed by output tensor
      // check if the main tensor is in the term
      if (terms[i].Bs_in_term[nBs] == true) {
        terms[i].blas_B_ids[inp] = nBs;
        for (int j = 0; j < order_A; j++) {
          idx = idx_A[j];
          if (terms[i].rev_index_order[idx] >= terms[i].inner_rev_idx) {
            in_term_idx[inp][idx] = true;
            nidx_term[inp]++;
          }
        }
        inp++;
      }
      // check if any intermediate tensors are in the term
      if (i > 0) {
        // all terms except the first are using tbuffer
        for (int k = 0; k < terms[i].i_inp_buf_id; k++) {
          int inp_buf_id = terms[i].inp_buf_ids[k];
          terms[i].blas_B_ids[inp] = inp_buf_id + nBs;
          for (int j = 0; j < terms[inp_buf_id].tbuffer_order; j++) {
            idx = terms[inp_buf_id].idx_tbuffer[j];
            if (terms[i].rev_index_order[idx] >= terms[i].inner_rev_idx) {
              in_term_idx[inp][idx] = true;
              nidx_term[inp]++;
            }
          }
          inp++;
        }
      }
      // check if any input tensors are in the term
      for (int k = 0; k < nBs-1; k++) {
        if (terms[i].Bs_in_term[k] == true) {
          terms[i].blas_B_ids[inp] = k;
          for (int j = 0; j < order_Bs[k]; j++) {
            idx = idx_Bs[k][j];
            if (terms[i].rev_index_order[idx] >= terms[i].inner_rev_idx) {
              in_term_idx[inp][idx] = true;
              nidx_term[inp]++;
            }
          }
          inp++;
        }
      }
      IASSERT(inp == 2);
      if (i < nterms-1) {
        terms[i].blas_B_ids[inp++] = i + nBs;
        // all terms except the last are writing to tbuffer allocated in this term
        for (int j = 0; j < terms[i].tbuffer_order; j++) {
          idx = terms[i].idx_tbuffer[j];
          if (terms[i].rev_index_order[idx] >= terms[i].inner_rev_idx) {
            in_term_idx[2][idx] = true;
            nidx_term[2]++;
          }
        }
      }
      else {
        terms[i].blas_B_ids[inp++] = nBs - 1;
        // IASSERT(terms[i].Bs_in_term[nBs-1] == true);
        // assume the output is written in the last term - we are not setting it in the code (so the above assert fails); for optimization later
        for (int j = 0; j < order_Bs[nBs-1]; j++) {
          idx = idx_Bs[nBs-1][j];
          if (terms[i].rev_index_order[idx] >= terms[i].inner_rev_idx) {
            in_term_idx[2][idx] = true;
            nidx_term[2]++;
          }
        }
      }
      IASSERT(inp == 3);

      // check if more than one sparse index requires sparse_loop() infra
      int nsp_idx = 0;
      int dense_sp_idx = -1;
      int io = 0;
      // find the last sparse index in this term
      int lsp_idx = -1;
      for (; io < terms[i].index_order_sz; io++) {
        if (rev_idx_map[nBs][terms[i].index_order[io]] != -1) {
          lsp_idx = terms[i].index_order[io];
        }
      }
      // check if all the parent sparse indices have been iterated before it
      // Assumption: the index order is built using the sparse main tensor first, so they appear first in the order and also use the inverted order 
      // i.e. (k,j,i) tensor order is stored as 2,1,0, so for 0, need to check if 1 and 2 have been iterated over
      for (int pidx = lsp_idx+1; pidx < order_A; pidx++) {
        int k = 0;
        for (; k < io; k++) {
          if (terms[i].index_order[k] == pidx) {
            break;
          }
        }
        if (k == io) {
          nsp_idx++;
          // find the term which can create a sparse loop
          for (int j = 0; j < i; j++) {
            if (terms[j].rev_index_order[pidx] != -1) {
              // does it iterate over all the parent sparse indices
              int nsp_idx_j = 1;
              for (int pidx_j = lsp_idx+1; pidx_j < order_A; pidx_j++) {
                if (terms[j].rev_index_order[pidx_j] != -1) {
                  nsp_idx_j++;
                }
              }
              if (nsp_idx_j == order_A) {
                // TODO: should be an array if we need to support multiple sparse loops
                terms[i].dense_sp_loop_in_term = j;
                dense_sp_idx = lsp_idx;
                break;
              }
            }
          }
        }
      }
      IASSERT(nsp_idx < 2);
      if (nsp_idx == 1) {
        IASSERT(terms[i].dense_sp_loop_in_term != -1);
        IASSERT(dense_sp_idx != -1);
        terms[terms[i].dense_sp_loop_in_term].dense_sp_loop = (bool *)CTF_int::alloc(sizeof(bool) * len_idx[dense_sp_idx]);
      }
      int num_idx = 0;
      int rem_list_idx[num_indices];
      int rl = 0;
      int st = terms[i].inner_rev_idx;
      for (; st < terms[i].index_order_sz; st++) {
        idx = terms[i].index_order[st];
        if (rev_idx_map[nBs][idx] == -1) {
          num_idx++;
          if (num_idx >= 4) {
            // encountered four dense indices; dense_idx is -1; so now note it down to be rem_list_idx[0]
            if (terms[i].dense_idx != -1) {
              IASSERT(terms[i].dense_idx == rem_list_idx[0]);
              for (int k = 1; k < rl; k++) {
                rem_list_idx[k-1] = rem_list_idx[k];
              }
              rl--;
              num_idx--;
            }
            if (terms[i].sparse_idx != -1) {
              int sp_idx = terms[i].sparse_idx;
              for (int j = 0; j < 3; j++) {
                if (in_term_idx[j][sp_idx] == true) {
                  in_term_idx[j][sp_idx] = false;
                  nidx_term[j]--;
                }
              }
              terms[i].sparse_idx = -1;
            }
            terms[i].dense_idx = rem_list_idx[0];
          }
          rem_list_idx[rl++] = idx;
        }
        else {
          if (terms[i].dense_idx != -1) {
            // encountered a sparse index after dense indices
            IASSERT(rl != 0);
            for (int j = 0; j < 3; j++) {
              for (int k = 0; k < rl; k++) {
                if (in_term_idx[j][rem_list_idx[k]] == true) {
                  in_term_idx[j][rem_list_idx[k]] = false;
                  nidx_term[j]--;
                }
              }
            }
            rl = 0;
            terms[i].dense_idx = -1;
          }
          terms[i].sparse_idx = idx;
          num_idx = 0;
        }
      }
      IASSERT(num_idx <= 4);
      // BLAS kernel selection
      // remove sparse indices
      st = terms[i].inner_rev_idx;
      for (; st < terms[i].index_order_sz; st++) {
        idx = terms[i].index_order[st];
        if (rev_idx_map[nBs][idx] != -1) {
          for (int j = 0; j < 3; j++) {
            if (in_term_idx[j][idx] == true) {
              in_term_idx[j][idx] = false;
              nidx_term[j]--;
            }
          }
        }
      }

      if (num_idx > 0) {
        /*
        int op = 0;
        if (terms[i].Bs_in_term[nBs] == true) {
          terms[i].blas_ops[op++] = MAIN_TENSOR;
        }
        if (i > 0 && terms[i].inp_buf_id != -1) {
          terms[i].blas_ops[op++] = INTERMEDIATE_TENSOR;
        }
        for (int k = 0; k < nBs-1; k++) {
          if (terms[i].Bs_in_term[k] == true) {
            terms[i].blas_ops[op] = INP_B;
            terms[i].blas_B_ids[op++] = k;
          }
        }
        IASSERT(op == 2);
        */

        for (int io = 0; io < num_indices; io++) {
          idx = terms[i].index_order[io];
          if (in_term_idx[0][idx] == true || in_term_idx[1][idx] == true) {
            terms[i].blas_idx = idx;
            break;
          }
        }
        /*
        if (nidx_term[2] == 0) {
          spttn_print << "term_id: " << i << " nidx_term[0]: " << nidx_term[0] << " nidx_term[1]: " << nidx_term[1] << " nidx_term[2]: " << nidx_term[2] << " inner_idx: " << terms[i].inner_idx << " reset_idx: " << terms[i].reset_idx << std::endl;
          IASSERT(terms[i].inner_idx != -1);
          for i:
            for a:
              buf2 += buf[a] * U[a,i]
            Z_ijk += buf2 * T_ijk
          the first term has inner_idx of 1, but the output is a buffer that needs to be accumulated, so reset the buffer at 'a' in this case
          IASSERT(terms[i].reset_idx != -1);
          // treat this as a RECURSIVE_LOOP and not as SCALAR
          terms[i].blas_kernel = RECURSIVE_LOOP;
          spttn_print << "term_id: " << i << " blas_kernel: " << "RECURSIVE_LOOP" << std::endl;
          continue;
        }
        */
        select_blas_kernel<dtype>(i, in_term_idx, nidx_term, num_idx, len_idx, terms, num_indices, idx_Bs, nBs, rank);
      }
      else {
        terms[i].blas_kernel = RECURSIVE_LOOP;
        if (rank == 0) spttn_print << "term_id: " << i << " blas_kernel: " << "RECURSIVE_LOOP" << std::endl;
      }
    }
    for(int i = 0; i < 3; i++) CTF_int::cdealloc(in_term_idx[i]);
    CTF_int::cdealloc(in_term_idx);
  } 
  
  template<typename dtype>
  void allocate_buffer(const int64_t *            len_idx,
                       const int                  num_indices,
                       const int                  nterms,
                       contraction_terms<dtype> * terms,
                       const int                  rank)
  {
    debug_spttn_cyclops spttn_print;
    if (rank == 0) {
      spttn_print << "----------------Allocating intermediate tensors------------------------" << std::endl;      
    }
    for (int i = 0; i < nterms - 1; i++) {
      IASSERT(terms[i].out_buf_id != -1);
      int j = terms[i].out_buf_id;
      // common indices
      int c = 0;
      while (terms[i].index_order[c] == terms[j].index_order[c]) c++;

      int inp = c;
      int op = c;
      int tbo = 0;
      int tbuf_size = 1;
      for (; inp < num_indices; inp++) {
        int idx = terms[i].index_order[inp];
        if (idx == -1) break;
        for (int k = op; k < num_indices; k++) {
          if (terms[j].index_order[k] == idx) {
            terms[i].idx_tbuffer[tbo] = idx;
            terms[i].rev_idx_tbuffer[idx] = tbo;
            terms[i].len_idx[tbo] = len_idx[idx];
            tbuf_size *= len_idx[idx];
            tbo++;
          }
        }
      }
      if (tbo != 0) {
        terms[i].tbuffer_order = tbo;
        terms[i].tbuffer = (dtype *)CTF_int::alloc(sizeof(dtype) * tbuf_size);
        std::memset(terms[i].tbuffer, 0, sizeof(dtype) * tbuf_size);
        terms[i].tbuffer_sz = tbuf_size;
      }
      else {
        // buffer is scalar
        tbuf_size = 1;
        terms[i].tbuffer = (dtype *)CTF_int::alloc(sizeof(dtype) * tbuf_size);
        std::memset(terms[i].tbuffer, 0, sizeof(dtype) * tbuf_size);
        terms[i].tbuffer_sz = tbuf_size;
      }
      if (rank == 0) {
        spttn_print << "term id: " << i << " tbuffer size: " << tbuf_size << " tbuffer order: " << tbo << std::endl;
        for (int k = 0; k < tbo; k++) {
          spttn_print << "idx: " << terms[i].idx_tbuffer[k] << " len: " << terms[i].len_idx[k] << std::endl;
        }
      }
    }
    if (rank == 0) {
      spttn_print << "-------------------------------------------------------------" << std::endl;      
    }
  }

  template<typename dtype>
  void gen_inv_idx(int                   num_indices,
                   int ***               idx_arr,
                   int                   order_A,
                   int *                 idx_A,
                   int                   nBs,
                   int *                 order_Bs,
                   int **                idx_Bs)
  {
    // nBs + A
    *idx_arr = (int **)CTF_int::alloc(sizeof(int *) * (nBs + 1));
    for (int j = 0; j < (nBs + 1); j++) {
      (*idx_arr)[j] = (int *)CTF_int::alloc(sizeof(int) * num_indices);
      std::fill((*idx_arr)[j], (*idx_arr)[j]+num_indices, -1);
    }

    for (int j = 0; j < nBs; j++) {
      for (int i = 0; i < order_Bs[j]; i++) {
        (*idx_arr)[j][idx_Bs[j][i]] = i;
      }
    }

    for (int i = 0; i < order_A; i++){
      (*idx_arr)[nBs][idx_A[i]] = i;
    }
  }
  
  template<typename dtype>
  void id_inner_ids(int                        num_indices,                    
                    int                        num_active_terms,
                    const int *                active_terms_ind,
                    int                        iidx,
                    const int                  nterms,
                    contraction_terms<dtype> * terms,
                    const int                  rank)
  {
    if (num_active_terms == 1) {
      // inner indices
      if (terms[active_terms_ind[0]].index_order[iidx] == -1) {
        // the term is a leaf
        terms[active_terms_ind[0]].inner_idx = -1; 
        terms[active_terms_ind[0]].inner_rev_idx = -1;
      }
      else {
        terms[active_terms_ind[0]].inner_idx = terms[active_terms_ind[0]].index_order[iidx];
        terms[active_terms_ind[0]].inner_rev_idx = iidx;
      }
      return;
    }
    iidx++;
    int idx = terms[active_terms_ind[0]].index_order[iidx];
    int k = 1;
    while (idx == -1) {
      idx = terms[active_terms_ind[k++]].index_order[iidx];
      if (k == num_active_terms) break;
    }
    if (idx == -1) {
      // the terms in the branch share an index (multiple terms as leaves)
      for (int i = 0; i < num_active_terms; i++) {
        terms[active_terms_ind[i]].inner_idx = -1;
        terms[active_terms_ind[i]].inner_rev_idx = -1;
      }
      return;
    }

    int idx_terms[nterms];
    int num_it = 0;

    for (int i = 0; i < num_active_terms; i++) {
      int ind = active_terms_ind[i];
      if (terms[ind].index_order[iidx] == idx) {
        idx_terms[num_it++] = ind;
      }
      else {
        // split in the execution tree
        for (int j = 0; j < num_it; j++) {
          bool reset_buf = true;
          // reset_buf is set to true in the idx where in sp_gen_ctr.cxx, this idx is used to reset the buffer just before the loop is generated for it
          int x = idx_terms[j];
          for (int k = j+1; k < num_it; k++) {
            int y = idx_terms[k];
            if (terms[x].out_buf_id == y) {
              reset_buf = false;
              break;
            }
          }
          if (reset_buf == true) {
            terms[x].reset_idx = idx;
          }
        }
        if (num_it > 0) {
          id_inner_ids(num_indices, num_it, idx_terms, iidx, nterms, terms, rank);
        }
        num_it = 0;
        // move on to the next term
        idx = terms[ind].index_order[iidx];
        idx_terms[num_it++] = ind;
      }
    }
    if (num_it > 0) {
      id_inner_ids(num_indices, num_it, idx_terms, iidx, nterms, terms, rank);
    }
  }

  template<typename dtype>
  void index_order_idx(char const *          cidx_A,
                       const char * const *  cidx_Bs,
                       const std::string &   sindex_order,
                       int *                 term_index_order,
                       int *                 term_rev_index_order,
                       int                   order_A,
                       int *                 idx_A,
                       int                   nBs,
                       int *                 order_Bs,
                       int **                idx_Bs)
  {
    char cidx;
    int iidx;
    int j;
    int k;
    for (size_t i = 0; i < sindex_order.size(); i++) {
      char idx = sindex_order[i];
      for (j = 0; j < order_A; j++) {
        cidx = cidx_A[j];
        if (cidx == idx) {
          iidx = idx_A[j];
          term_index_order[i] = iidx;
          term_rev_index_order[iidx] = i;
          break;
        }
      }
      if (j == order_A) {
        for (k = 0; k < nBs; k++) {  
          for (j = 0; j < order_Bs[k]; j++) {
            cidx = cidx_Bs[k][j];
            if (cidx == idx) {
              iidx = idx_Bs[k][j];
              term_index_order[i] = iidx;
              term_rev_index_order[iidx] = i;
              break;
            }
          }
          if (j != order_Bs[k]) break;
        }
        IASSERT(k != nBs);
      }
    }
  }

  template<typename dtype>
  void term_idx(char const *          cidx_A,
       const char * const *           cidx_Bs,
       const std::string *            sterms,
       int                            nterms,
       contraction_terms<dtype> *     terms,
       int                            order_A,
       int *                          idx_A,
       int                            nBs,
       int *                          order_Bs,
       int **                         idx_Bs)
  {
    char cidx;
    int iidx;
    std::string op_sterms[nterms];
    std::string inp_sterms[nterms];
    std::string delim = "->";
    for (int i = 0; i < nterms; i++) {
      size_t inp = 0;
      size_t op = 0;
      op = sterms[i].find(delim, inp);
      IASSERT(op != std::string::npos);
      inp_sterms[i] = sterms[i].substr(inp, op - inp);
      op_sterms[i] = sterms[i].substr(op+delim.length());
    }
    // calculate Bs_in_term
    std::string sidx_A(cidx_A);
    for (int i = 0; i < nterms; i++) {
      std::stringstream inpss(inp_sterms[i]);
      std::string iinps;
      std::string pairs[2];
      int p = 0;
      while(getline(inpss, iinps, ',')) {
        pairs[p++] = iinps;
        for (int j = 0; j < nBs; j++) {
          std::string sidx_B(cidx_Bs[j]);
          if (sidx_B.compare(iinps) == 0) {
            terms[i].Bs_in_term[j] = true;
            break;
          }
        }
        if (sidx_A.compare(iinps) == 0) {
          terms[i].Bs_in_term[nBs] = true;
        }
      }
      // can one of the inputs to this term be an output of another term and vice versa
      for (int j = 0; j < i; j++) {
        if (op_sterms[j].compare(pairs[0]) == 0 || op_sterms[j].compare(pairs[1]) == 0) {
          // assert failure: both the inputs to this term are intermediate tensors
          // IASSERT(terms[i].inp_buf_id == -1);
          // terms[i].inp_buf_id = j;
          terms[j].out_buf_id = i;
          IASSERT(terms[i].inp_buf_ids[terms[i].i_inp_buf_id] == -1);
          terms[i].inp_buf_ids[terms[i].i_inp_buf_id++] = j;
        }
      }
    }
    // TODO: can use sindex_order and index_order to fuse the two for loops and iterate over indices only once to set in_term_idx and in_op_idx of all terms 
    for (int j = 0; j < order_A; j++) {
      cidx = cidx_A[j];
      iidx = idx_A[j];
      for (int i = 0; i < nterms; i++) {
        if (inp_sterms[i].find(cidx) != std::string::npos) {
          terms[i].in_term_idx[iidx] = true;
        }
        if (op_sterms[i].find(cidx) != std::string::npos) {
          terms[i].in_op_idx[iidx] = true;
        }
      }
    }

    for (int k = 0; k < nBs; k++) {  
      for (int j = 0; j < order_Bs[k]; j++) {
        cidx = cidx_Bs[k][j];
        iidx = idx_Bs[k][j];
        for (int i = 0; i < nterms; i++) {
          if (inp_sterms[i].find(cidx) != std::string::npos) {
            terms[i].in_term_idx[iidx] = true;
          }
          if (op_sterms[i].find(cidx) != std::string::npos) {
            terms[i].in_op_idx[iidx] = true;
          }
        }
      }
    }
  }

  template <typename dtype>
  void select_cp_io_populate_terms(char const *                   cidx_A,
                                   const char * const *           cidx_Bs,
                                   int                            nterms,
                                   contraction_terms<dtype> *     terms,
                                   int                            order_A,
                                   int *                          idx_A,
                                   int                            nBs,
                                   int *                          order_Bs,
                                   int **                         idx_Bs,
                                   int                            num_indices,
                                   int                            max_buf_dim,
                                   const int                      rank)
  {
    debug_spttn_cyclops spttn_print;
    // sparse tensor + (nBs-1), excluding the output tensor
    int ntensors = nBs;
    uint16_t op_inds = 0;
    uint16_t all_inds = 0;
    uint16_t sp_inds = 0;
    int64_t cp_cache_size = 1 << ntensors;
    CPCache * cp_cache = new CPCache[cp_cache_size];
    // populate the cache; main tensor id = 1, and Bs = 2, 3, 4, ...
    int tid = 1;
    for (int i = 0; i < order_A; i++) {
      cp_cache[tid].inds |= (1 << idx_A[i]);
      all_inds |= (1 << idx_A[i]);
      sp_inds |= (1 << idx_A[i]);
    }
    cp_cache[1].cost = 0;
    // ntensor-1 because the last tensor is the result
    for (int i = 0; i < (ntensors-1); i++) {
      tid = tid << 1;
      for (int j = 0; j < order_Bs[i]; j++) {
        cp_cache[tid].inds |= (1 << idx_Bs[i][j]);
        all_inds |= (1 << idx_Bs[i][j]);
        cp_cache[tid].cost = 0;
      }
    }
    for (int j = 0; j < order_Bs[ntensors-1]; j++) {
      op_inds |= (1 << idx_Bs[ntensors-1][j]);
      all_inds |= (1 << idx_Bs[ntensors-1][j]);
    }

    tid = 0;
    uint16_t lt[ntensors];
    for (int i = 0; i < ntensors; i++) {
      lt[i] = 1 << i;
      tid |= lt[i];
    }
    CTF_int::contraction_path * cp = new CTF_int::contraction_path(cp_cache, op_inds, ntensors);
    uint16_t optimal_cp_cost = cp->optimal_contraction_paths(lt, ntensors);
    std::vector<std::vector<CTerm> > paths = cp->enumerate_all_paths(tid);
    if (rank == 0) {
      spttn_print << "num paths: " << paths.size() << " optimal contraction path cost: " << (int)optimal_cp_cost << std::endl;
    }
    uint8_t numones[65536];
    popcount_init(numones);
    int8_t niloops = -1;
    int path = -1;
    std::vector<std::vector<uint16_t> > optimal_io;
    uint16_t pick_cp_cost = optimal_cp_cost;
    uint16_t max_cp_cost = 0;
    
    // calculate cost for each path
    uint16_t * path_cost = (uint16_t *)alloc(sizeof(uint16_t) * paths.size());
    for (size_t i = 0; i < paths.size(); i++) {
      IASSERT((int)paths[i].size() == nterms);
      path_cost[i] = 0;
      for (size_t j = 0; j < paths[i].size(); j++) {
        int8_t flops;
        uint16_t cinds;
        cp->contract(paths[i][j].ta, paths[i][j].tb, cinds, flops);
        path_cost[i] += flops;
      }
      if (path_cost[i] > max_cp_cost) max_cp_cost = path_cost[i];
    }

    bool sp_buffer = false;
    while (niloops == -1) {    
      for (size_t i = 0; i < paths.size(); i++) {
        // choose a contraction path that can be implemented with the given constraints
        if (path_cost[i] != pick_cp_cost) continue;
        // int thres_buf_sz = 2;
        local_index_order * lio = new local_index_order(nterms, all_inds, sp_inds, numones, paths[i], cp_cache, max_buf_dim, sp_buffer);
        uint16_t S = 0;
        uint8_t sT = 0;
        uint8_t eT = nterms-1;
        lio->io_cost(S, sT, eT);
        if (lio->icache[S][sT][eT].computed == false) {
          if (rank == 0) {
            spttn_print << "\nCould not find an optimal loop nest for the below path" << std::endl;
            for (size_t j = 0; j < paths[i].size(); j++) {
              paths[i][j].print();
            }
          }
          delete lio;
          continue;
        }
        // print index order for this path
        // lio->icache[S][sT][eT].print_element_in_icache(S, sT, eT);
        if (niloops < lio->icache[S][sT][eT].niloops[0]) {
          niloops = lio->icache[S][sT][eT].niloops[0];
          path = i;
          optimal_io = lio->icache[S][sT][eT].inds_order[0];
        }
        delete lio;
      }
      if (niloops == -1) {
        pick_cp_cost++;
        if (pick_cp_cost > max_cp_cost && sp_buffer == false) {
          if (rank == 0) {
            spttn_print << "Could not find any optimal loop nest for the given constraints with dense buffer indices" << std::endl;
          }
          pick_cp_cost = optimal_cp_cost;
          sp_buffer = true;
        }
        else if (pick_cp_cost > max_cp_cost) {
          if (rank == 0) {
            spttn_print << "Could not find any optimal loop nest for the given constraints with sparse buffer indices" << std::endl;
          }
          IASSERT(0);
        }
      }
    }

    if (rank == 0) {
      spttn_print << "============================" << std::endl;   
      spttn_print << "path chosen: " << path << std::endl;
    }
    if (niloops != -1) {
      // populate term[].index_order, term[].rev_index_order, term[].index_order_sz
      // from term_idx(): Bs_in_term[], inp_buf_id & out_buf_id, in_term_idx and in_op_idx
      // print tensors
      // term_idx<dtype>(cidx_A, cidx_Bs, sterms, nterms, terms, order_A, idx_A, nBs, order_Bs, idx_Bs); 
      int8_t tflops = 0;
      for (size_t j = 0; j < paths[path].size(); j++) {
        if (rank == 0) {
          paths[path][j].print();
        }
        int8_t flops;
        uint16_t cinds;
        cp->contract(paths[path][j].ta, paths[path][j].tb, cinds, flops);
        tflops += flops;
      }
      if (rank == 0) {
        spttn_print << "total loop depth: " << (int)tflops << std::endl;
        for (int i = 0; i < nterms; i++) {
          spttn_print << "term id " << i << ": ";
          for (size_t k = 0; k < optimal_io[i].size(); k++) {
            spttn_print << optimal_io[i][k] << " ";
          }
          spttn_print << std::endl;
        }
      }
      IASSERT((int)paths[path].size() == nterms);
      for (int i = 0; i < nterms; i++) {
        uint16_t ta = paths[path][i].ta;
        uint16_t tb = paths[path][i].tb;
        uint16_t tab = paths[path][i].tab;
        if (ta == 1 || tb == 1) {
          terms[i].Bs_in_term[nBs] = true;
        }
        // not marking the output tensor tensor[nBs-1]
        for (int j = 0; j < nBs-1; j++) {
          int tidB = 1 << (j+1);
          if (ta == tidB || tb == tidB) {
            terms[i].Bs_in_term[j] = true;
          }
        }
        for (int k = i+1; k < nterms; k++) {
          if (tab == paths[path][k].ta || tab == paths[path][k].tb) {
            terms[i].out_buf_id = k;
            // assert failure: both the inputs to this term are intermediate tensors
            IASSERT(terms[k].inp_buf_ids[terms[k].i_inp_buf_id] == -1);
            terms[k].inp_buf_ids[terms[k].i_inp_buf_id++] = i;
          }
        }
        // main tensor indices
        if (ta == 1 || tb == 1) {
          for (int j = 0; j < order_A; j++) {
            int iidx = idx_A[j];
            terms[i].in_term_idx[iidx] = true;
          }
        }
        // Bs indices
        for (int j = 0; j < nBs; j++) {
          if (terms[i].Bs_in_term[j]) {
            for (int k = 0; k < order_Bs[j]; k++) {
              int iidx = idx_Bs[j][k];
              terms[i].in_term_idx[iidx] = true;
            }
          }
        }
        // buffer indices
        for (int k = 0; k < i; k++) {
          if (terms[k].out_buf_id == i) {
            uint16_t oinds = paths[path][k].inds;
            int j = 0;
            while (oinds != 0) {
              if (oinds & 1) {
                terms[i].in_term_idx[j] = true;
                terms[k].in_op_idx[j] = true;
              }
              oinds = oinds >> 1;
              j++;
            }
          }
        }
        // output B indices
        if (i == (nterms-1)) {
          IASSERT(terms[i].out_buf_id == -1);
          for (int j = 0; j < order_Bs[nBs-1]; j++) {
            int iidx = idx_Bs[nBs-1][j];
            terms[i].in_term_idx[iidx] = true;
            terms[i].in_op_idx[iidx] = true;
          }
        }
      }
      for (int i = 0; i < nterms; i++) {
        if (rank == 0) spttn_print << "term id " << i << ": ";
        for (size_t k = 0; k < optimal_io[i].size(); k++) {
          int iidx = log2(optimal_io[i][k]);
          terms[i].index_order[k] = iidx;
          terms[i].rev_index_order[iidx] = k;
        }
        terms[i].index_order_sz = optimal_io[i].size();
        if (rank == 0) {
          spttn_print << "terms[" << i << "].index_order_sz = " << terms[i].index_order_sz << std::endl;
          for (int k = 0; k < num_indices; k++) {
            spttn_print << terms[i].index_order[k] << " ";
          }
          spttn_print << std::endl;
        }
        if (terms[i].index_order_sz < num_indices) {
          // index_order is filled with num_indices in contraction_terms constructor
          IASSERT(terms[i].index_order[terms[i].index_order_sz] == -1);
        }
      }
      if (rank == 0) {
        spttn_print << "niloops: " << (int)niloops << std::endl;
      }
    }
    else {
      IASSERT(0);
      if (rank == 0) {
        spttn_print << "Could not find an optimal loop nest for the given constraints" << std::endl; 
      }
    }
    if (rank == 0) {
      spttn_print << "============================" << std::endl;
    }
    delete cp;
    delete [] cp_cache;
  }

}
#endif
