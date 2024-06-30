#include "../shared/iter_tsr.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits.h>
#include <type_traits>
#include <utility>
#include "prepare_kernel.h"
#include "execute_kernel.h"
#include "../shared/offload.h"
#include "../shared/util.h"
#include "csf.h"
#include "../shared/blas_symbs.h"
#include <queue>

#ifdef USE_MKL
#include "../shared/mkl_symbs.h"
#endif

namespace CTF_int{
  
  template class CSF<double>;
  template class CSF<int>;

  template<int level>                                                                          
  void spA_dnBs_ctrloop(char const *                alpha,
                        CSF<double> *               A_tree,                                
                        algstrct const *            sr_A,
                        int                         order_A,
                        int const *                 idx_A,
                        int                         nBs,
                        char **                     Bs,
                        const algstrct * const *    sr_Bs,
                        int *                       order_Bs,
                        const int64_t **            len_Bs,
                        const int64_t * const *     lda_Bs,
                        const int * const *         idx_Bs,
                        const int64_t *             len_idx,
                        bivar_function const *      func,
                        const int * const *         rev_idx_map,
                        int                         num_indices,
                        int64_t                     tree_pt_st,
                        int64_t                     tree_pt_en,
                        contraction_terms<double> * terms,
                        int                         nterms,
                        int *                       active_terms_ind,
                        int                         num_active_terms,
                        char **                     tbuffer,
                        const int64_t * const *     lda_tbuffers,
                        int                         tree_level)
  {
    int iidx = (num_indices - 1) - level;
    int tid = active_terms_ind[0];
    int idx = terms[tid].index_order[iidx];
    if(idx == num_indices) {
      contraction_terms<double> & termx = terms[active_terms_ind[0]]; 
      IASSERT(termx.blas_kernel == RECURSIVE_LOOP);
      double * dX = (double *)Bs[termx.X]; 
      double * dY = (double *)Bs[termx.Y];
      if (termx.ALPHA == -1) {
        IASSERT(termx.Y != (nBs-1));
        double alpha = A_tree->dt[tree_pt_st];
        *dY += alpha * *dX;
      }
      else {
        double * alpha = (double *)Bs[(int)termx.ALPHA]; 
        *dY += *alpha * *dX;
      }
      return;
    }
    int active_terms_buffer[nterms];
    int n_act_buf = 0;
    for (int i = 0; i < num_active_terms; i++) {
      int ind = active_terms_ind[i];
      contraction_terms<double> & term = terms[ind]; 
      if (term.index_order[iidx] == idx) {
        active_terms_buffer[n_act_buf++] = ind;
      }
      else { 
        for (int j = 0; j < n_act_buf; j++) {
          int indj = active_terms_buffer[j];
          contraction_terms<double> & termj = terms[indj];
          if (termj.reset_idx == termj.index_order[iidx]) {
            memset(termj.tbuffer, 0, termj.tbuffer_sz * sizeof(double));
          }
        }
        if (n_act_buf == 1 && terms[active_terms_buffer[0]].blas_kernel != RECURSIVE_LOOP && terms[active_terms_buffer[0]].blas_idx == idx) {
          call_blas(alpha, A_tree, sr_A, order_A, idx_A, nBs, Bs, sr_Bs, order_Bs, len_Bs, lda_Bs, idx_Bs, len_idx, func, rev_idx_map, num_indices, tree_pt_st, tree_pt_en, terms, nterms, active_terms_buffer, n_act_buf, tbuffer, lda_tbuffers, tree_level, level, terms[active_terms_buffer[0]]);
        }
        else {
          call_ctrloop(alpha, A_tree, sr_A, order_A, idx_A, nBs, Bs, sr_Bs, order_Bs, len_Bs, lda_Bs, idx_Bs, len_idx, func, rev_idx_map, num_indices, tree_pt_st, tree_pt_en, terms, nterms, active_terms_buffer, n_act_buf, tbuffer, lda_tbuffers, tree_level, level);
        }
        n_act_buf = 0;
        idx = term.index_order[iidx];
        i--;
      }
    }
    if (n_act_buf == 1 && terms[active_terms_buffer[0]].blas_kernel != RECURSIVE_LOOP && terms[active_terms_buffer[0]].blas_idx == idx) {
      call_blas(alpha, A_tree, sr_A, order_A, idx_A, nBs, Bs, sr_Bs, order_Bs, len_Bs, lda_Bs, idx_Bs, len_idx, func, rev_idx_map, num_indices, tree_pt_st, tree_pt_en, terms, nterms, active_terms_buffer, n_act_buf, tbuffer, lda_tbuffers, tree_level, level, terms[active_terms_buffer[0]]);
    }
    else {
      // TODO: handle case where n_act_buf > 1 and still term1.index_order[iidx] == num_indices
      call_ctrloop(alpha, A_tree, sr_A, order_A, idx_A, nBs, Bs, sr_Bs, order_Bs, len_Bs, lda_Bs, idx_Bs, len_idx, func, rev_idx_map, num_indices, tree_pt_st, tree_pt_en, terms, nterms, active_terms_buffer, n_act_buf, tbuffer, lda_tbuffers, tree_level, level);
    }
  }
  
  template<>                                                                          
  void spA_dnBs_ctrloop<MAX_ORD>(char const *                alpha,
                                 CSF<double> *               A_tree,                                
                                 algstrct const *            sr_A,
                                 int                         order_A,
                                 int const *                 idx_A,
                                 int                         nBs,
                                 char **                     Bs,
                                 const algstrct * const *    sr_Bs,
                                 int *                       order_Bs,
                                 const int64_t **            len_Bs,
                                 const int64_t * const *     lda_Bs,
                                 const int * const *         idx_Bs,
                                 const int64_t *             len_idx,
                                 bivar_function const *      func,
                                 const int * const *         rev_idx_map,
                                 int                         num_indices,
                                 int64_t                     tree_pt_st,
                                 int64_t                     tree_pt_en,
                                 contraction_terms<double> * terms,
                                 int                         nterms,
                                 int *                       active_terms_ind,
                                 int                         num_active_terms,
                                 char **                     tbuffer,
                                 const int64_t * const *     lda_tbuffers,
                                 int                         tree_level);



  void call_blas(char const *                alpha,
                    CSF<double> *               A_tree,                                
                    algstrct const *            sr_A,
                    int                         order_A,
                    int const *                 idx_A,
                    int                         nBs,
                    char **                     Bs,
                    const algstrct * const *    sr_Bs,
                    int *                       order_Bs,
                    const int64_t **            len_Bs,
                    const int64_t * const *     lda_Bs,
                    const int * const *         idx_Bs,
                    const int64_t *             len_idx,
                    bivar_function const *      func,
                    const int * const *         rev_idx_map,
                    int                         num_indices,
                    int64_t                     tree_pt_st,
                    int64_t                     tree_pt_en,
                    contraction_terms<double> * terms,
                    int                         nterms,
                    int *                       active_terms_ind,
                    int                         num_active_terms,
                    char **                     tbuffer,
                    const int64_t * const *     lda_tbuffers,
                    int                         tree_level,
                    int                         level,
                    contraction_terms<double> & term)
  {
    int iidx = (num_indices - 1) - level;
    int idx = term.index_order[iidx];
    switch (term.break_rec_idx[idx]) {
      case SPARSE_xAXPY: {
        double * dY = (double *)Bs[term.Y];
        for (int64_t it = tree_pt_st; it < tree_pt_en; it++) {
          double alpha = A_tree->dt[it];
          int64_t idx_i = A_tree->idx[0][it];
          double * dX = (double *)((double *)Bs[term.X] + lda_Bs[term.X][idx] * idx_i);
          #pragma ivdep
          for (int64_t i = 0; i < term.N; i++){
            dY[i] += alpha * dX[i];
          }
        }
      }
      break;
      case xAXPY: {
        double * dX = (double *)Bs[term.X];
        double * dY = (double *)Bs[term.Y];
        int N = term.N;
        double * alpha = (double *)Bs[(int)term.ALPHA];
        #pragma ivdep
        for (int64_t j = 0; j < N; j++) {
          dY[j*term.INCY] += *alpha * dX[j*term.INCX];
        }
      }
      break;
      case xGER: {
        double * dX = (double *)Bs[term.X];
        double * dY = (double *)Bs[term.Y];
        CTF_BLAS::DGER(&term.M, &term.N, &term.ALPHA, dX, &term.INCX, dY, &term.INCY, (double *)Bs[term.A], &term.LDA);
      }
      break;
      case DENSE_xAXPY_3D:
      case DENSE_xAXPY_2D: {
        int64_t n[1];
        n[0] = len_idx[idx];
        double * dX = (double *)Bs[term.X]; 
        int idx_alpha = idx_Bs[(int)term.ALPHA][0];
        GENERATE_NESTED_LOOP(1, n, {
          double * v = (double *)((double *)Bs[(int)term.ALPHA] + lda_Bs[(int)term.ALPHA][idx_alpha] * _ii);
          double alpha = v[0];
          double * dY = (double *)((double *)Bs[term.Y] + lda_Bs[term.Y][idx_alpha] * _ii);
          for (int64_t j = 0; j < term.N; j++) {
            dY[j] += alpha * dX[j];
          }
        });
      }
      break;
      case DENSE_STRIDED: {
        int64_t n[1];
        n[0] = len_idx[idx];
        double * dX = (double *)Bs[term.X]; 
        int idx_alpha = idx_Bs[(int)term.ALPHA][0];
        GENERATE_NESTED_LOOP(1, n, {
          double * v = (double *)((double *)Bs[(int)term.ALPHA] + lda_Bs[(int)term.ALPHA][idx_alpha] * _ii);
          double alpha = v[0];
          double * dY = (double *)((double *)Bs[term.Y] + lda_Bs[term.Y][idx_alpha] * _ii);
          for (int64_t j = 0; j < term.N; j++) {
            dY[j*term.INCY] += alpha * dX[j*term.INCX];
          }
        });
      }
      break;
      case xVEC_MUL: {
        double * dX = (double *)Bs[term.X];
        double * dY = (double *)Bs[term.Y];
        int N = term.N;
        double * alpha = (double *)Bs[(int)term.ALPHA];
        #pragma ivdep
        for (int64_t j = 0; j < N; j++) {
          dY[j] += alpha[j] * dX[j];
        }
      }
      break;
      case DENSE_3D: {
        int64_t idx1 = term.index_order[iidx];
        int terms[3];
        terms[0] = term.X; terms[1] = term.Y; terms[2] = term.ALPHA;
        for (int64_t i = 0; i < len_idx[idx1]; i++) {
          char *tBs_idx1[3];
          for (int j = 0; j < 3; j++) {
            int pos_idx;
            if (terms[j] > nBs) pos_idx = term.rev_idx_tbuffer[idx1];
            else pos_idx = rev_idx_map[terms[j]][idx1]; 
            if (pos_idx != -1) {
              tBs_idx1[j] = (char *)((double *)Bs[terms[j]] + lda_Bs[terms[j]][idx1] * i);
            }
            else {
              tBs_idx1[j] = Bs[terms[j]];
            }
          }
          int64_t idx2 = term.index_order[iidx+1];
          for (int64_t ii = 0; ii < len_idx[idx2]; ii++) {
            char *tBs_idx2[3];
            for (int j = 0; j < 3; j++) {
              int pos_idx;
              if (terms[j] > nBs) pos_idx = term.rev_idx_tbuffer[idx2];
              else pos_idx = rev_idx_map[terms[j]][idx2]; 
              if (pos_idx != -1) {
                tBs_idx2[j] = (char *)((double *)tBs_idx1[j] + lda_Bs[terms[j]][idx2] * ii);
              }
              else {
                tBs_idx2[j] = tBs_idx1[j];
              }
            }
            int64_t idx3 = term.index_order[iidx+2];
            for (int64_t iii = 0; iii < len_idx[idx3]; iii++) {
              char *tBs_idx3[3];
              for (int j = 0; j < 3; j++) {
                int pos_idx;
                if (terms[j] > nBs) pos_idx = term.rev_idx_tbuffer[idx3];
                else pos_idx = rev_idx_map[terms[j]][idx3]; 
                if (pos_idx != -1) {
                  tBs_idx3[j] = (char *)((double *)tBs_idx2[j] + lda_Bs[terms[j]][idx3] * iii);
                }
                else {
                  tBs_idx3[j] = tBs_idx2[j];
                }
              }
              double * dX = (double *)tBs_idx3[0];
              double * dY = (double *)tBs_idx3[1];
              double * alpha = (double *)tBs_idx3[2];
              *dY += *alpha * *dX;
            }
          }
        }
      }
      break;
      default: {
        IASSERT(0);
        IASSERT(term.blas_kernel == RECURSIVE_LOOP);
        IASSERT(term.break_rec_idx[idx] == RECURSIVE_LOOP);
        double * dX = (double *)Bs[term.X]; 
        double * dY = (double *)Bs[term.Y];
        if (term.ALPHA == -1) {
          IASSERT(term.Y != (nBs-1));
          double alpha = A_tree->dt[tree_pt_st];
          *dY += alpha * *dX;
        }
        else {
          double * alpha = (double *)Bs[(int)term.ALPHA]; 
          *dY += *alpha * *dX;
        }
      }
      break;
    }
  }
 
  void call_ctrloop(char const *                alpha,
                    CSF<double> *               A_tree,                                
                    algstrct const *            sr_A,
                    int                         order_A,
                    int const *                 idx_A,
                    int                         nBs,
                    char **                     Bs,
                    const algstrct * const *    sr_Bs,
                    int *                       order_Bs,
                    const int64_t **            len_Bs,
                    const int64_t * const *     lda_Bs,
                    const int * const *         idx_Bs,
                    const int64_t *             len_idx,
                    bivar_function const *      func,
                    const int * const *         rev_idx_map,
                    int                         num_indices,
                    int64_t                     tree_pt_st,
                    int64_t                     tree_pt_en,
                    contraction_terms<double> * terms,
                    int                         nterms,
                    int *                       active_terms_ind,
                    int                         num_active_terms,
                    char **                     tbuffer,
                    const int64_t * const *     lda_tbuffers,
                    int                         tree_level,
                    int                         level)
  {
    int iidx = (num_indices - 1) - level;
    int tid = active_terms_ind[0];
    int idx = terms[tid].index_order[iidx];
    char *tBs[nBs+nterms];
    char *ttbuf[nterms];

    bool traverse_tree = terms[active_terms_ind[0]].Bs_in_term[nBs] ? true : false;

    // idx not in A or the element of tree is already contracted so do not recurse tree
    // DENSE buffer; need to disable the second check for SPARSE
    char * active_Bs[nBs+nterms];
    char ** active_tBs[nBs+nterms];
    int64_t active_ldas[nBs+nterms];
    int szb = 0;
    for (int j = 0; j < nBs; j++) {
      if (lda_Bs[j] == nullptr) {
        tBs[j] = nullptr;
        continue;
      }
      if (rev_idx_map[j][idx] != -1) {
        active_Bs[szb] = Bs[j];
        active_tBs[szb] = &tBs[j];
        active_ldas[szb++] = lda_Bs[j][idx];
      }
      else {
        tBs[j] = Bs[j];
      }
    }
    for (int j = 0; j < nterms; j++) {
      int pos_idx_buf = terms[j].rev_idx_tbuffer[idx];
      if (pos_idx_buf != -1) {
        active_Bs[szb] = Bs[nBs+j];
        active_tBs[szb] = &tBs[nBs+j];
        //active_ldas[szb++] = lda_tbuffers[j][idx];
        active_ldas[szb++] = lda_Bs[nBs+j][idx];
      }
      else {
        tBs[nBs+j] = Bs[nBs+j];
      }
    }
    if (rev_idx_map[nBs][idx] == -1 /*|| !traverse_tree*/) {
      if (rev_idx_map[nBs][idx] != -1) {
        for (int64_t i = 0; i < len_idx[idx]; i++) {
          // TODO: should convert this loop also to sparse-dense single loop
          if (terms[active_terms_ind[0]-1].dense_sp_loop[i] == true) {
            terms[active_terms_ind[0]-1].dense_sp_loop[i] = false;
            #pragma ivdep
            for (int j = 0; j < szb; j++) {
              *active_tBs[j] = (char *)((double *)active_Bs[j] + active_ldas[j] * i);
            }
            SWITCH_ORD_CALL(spA_dnBs_ctrloop, level-1, alpha, A_tree, sr_A, order_A, idx_A, nBs, tBs, sr_Bs, order_Bs, len_Bs, lda_Bs, idx_Bs, len_idx, func, rev_idx_map, num_indices, tree_pt_st, tree_pt_en, terms, nterms, active_terms_ind, num_active_terms, ttbuf, lda_tbuffers, tree_level);
          }
        }
      }
      else {
        for (int64_t i = 0; i < len_idx[idx]; i++) {
          #pragma ivdep
          for (int j = 0; j < szb; j++) {
            *active_tBs[j] = (char *)((double *)active_Bs[j] + active_ldas[j] * i);
          }
          SWITCH_ORD_CALL(spA_dnBs_ctrloop, level-1, alpha, A_tree, sr_A, order_A, idx_A, nBs, tBs, sr_Bs, order_Bs, len_Bs, lda_Bs, idx_Bs, len_idx, func, rev_idx_map, num_indices, tree_pt_st, tree_pt_en, terms, nterms, active_terms_ind, num_active_terms, ttbuf, lda_tbuffers, tree_level);
        }
      } 
    }
    else {
      for (int64_t i = tree_pt_st; i < tree_pt_en; i++) {
        int64_t idx_idim = A_tree->get_idx(tree_level, i);
        #pragma ivdep
        for (int j = 0; j < szb; j++) {
          *active_tBs[j] = (char *)((double *)active_Bs[j] + active_ldas[j] * idx_idim);
        }
        if (tree_level != 0) {
          int64_t imax = A_tree->num_children(tree_level, i);
          int64_t child_pt = A_tree->get_child_ptr(tree_level, i);
          SWITCH_ORD_CALL(spA_dnBs_ctrloop, level-1, alpha, A_tree, sr_A, order_A, idx_A, nBs, tBs, sr_Bs, order_Bs, len_Bs, lda_Bs, idx_Bs, len_idx, func, rev_idx_map, num_indices, child_pt, (child_pt+imax), terms, nterms, active_terms_ind, num_active_terms, ttbuf, lda_tbuffers, tree_level-1);
        }
        else {
          SWITCH_ORD_CALL(spA_dnBs_ctrloop, level-1, alpha, A_tree, sr_A, order_A, idx_A, nBs, tBs, sr_Bs, order_Bs, len_Bs, lda_Bs, idx_Bs, len_idx, func, rev_idx_map, num_indices, i, -1, terms, nterms, active_terms_ind, num_active_terms, ttbuf, lda_tbuffers, tree_level);
        }
      }
    }
  }

  void spA_dnBs_gen_ctr(char const *                alpha,
                        CSF<double> *               A_tree,
                        algstrct const *            sr_A,
                        int                         order_A,
                        int const *                 idx_A,
                        int                         nBs,
                        char **                     Bs,
                        const algstrct * const *    sr_Bs,
                        int *                       order_Bs,
                        const int64_t **            len_Bs,
                        const int64_t * const *     edge_len_Bs,
                        const int64_t *             len_idx,
                        const int * const *         idx_Bs,
                        const int * const *         rev_idx_map,
                        int                         num_indices,
                        contraction_terms<double> * terms,
                        int                         nterms,
                        bivar_function const *      func) 
  {

    int64_t ** lda_Bs;
    int64_t ** lda_tbuffers;

    lda_Bs = (int64_t **) CTF_int::alloc(sizeof(int64_t *) * (nBs+nterms));
    for (int i = 0; i < nBs; i++) {
      if (edge_len_Bs[i] == nullptr) {
        lda_Bs[i] = nullptr;
        IASSERT(Bs[i] == nullptr);
        continue;
      }
      lda_Bs[i] = (int64_t *) CTF_int::alloc(sizeof(int64_t) * num_indices);
      lda_Bs[i][idx_Bs[i][0]] = 1;
      for (int j = 1; j < order_Bs[i]; j++) {
        lda_Bs[i][idx_Bs[i][j]] = lda_Bs[i][idx_Bs[i][j-1]] * edge_len_Bs[i][j-1];
      }
    }

    // tie the last term's buffer to the output tensor to skip an "if" when contracting using BLAS
    terms[nterms-1].tbuffer = (double *)Bs[nBs-1];
   
    char *ttbuf[nterms];
    for (int i = 0; i < nterms; i++) {
      ttbuf[i] = (char *)terms[i].tbuffer;
    }

    for (int i = 0; i < nterms; i++) {
      if (terms[i].tbuffer_order == -1) continue;
      lda_Bs[nBs+i] = (int64_t *) CTF_int::alloc(sizeof(int64_t) * num_indices);
      //(i,a): i is the fastest moving index
      lda_Bs[nBs+i][terms[i].idx_tbuffer[0]] = 1;
      for (int j = 1; j < terms[i].tbuffer_order; j++) {
        lda_Bs[nBs+i][terms[i].idx_tbuffer[j]] = lda_Bs[nBs+i][terms[i].idx_tbuffer[j-1]] * terms[i].len_idx[j-1];
      }
    }

    int level = num_indices - 1;
    int active_terms_buffer[nterms];
    // TODO: can use std::iota
    for (int i = 0; i < nterms; i++) {
      active_terms_buffer[i] = i;
    }
    int64_t imax = A_tree->nnz_level[order_A-1];

    SWITCH_ORD_CALL(spA_dnBs_ctrloop, level, alpha, A_tree, sr_A, order_A, idx_A, nBs, Bs, sr_Bs, order_Bs, len_Bs, lda_Bs, idx_Bs, len_idx, func, rev_idx_map, num_indices, 0, imax, terms, nterms, active_terms_buffer, nterms, ttbuf, lda_tbuffers, order_A-1); 
    for (int i = 0; i < nBs; i++) {
      cdealloc(lda_Bs[i]);
    }
    for (int i = 0; i < nterms; i++) {
      if (terms[i].tbuffer_order == -1) continue;
      cdealloc(lda_Bs[i+nBs]);
    }
    cdealloc(lda_Bs);
  }

  void calc_ldas(int                         nBs,
                 int *                       order_Bs,
                 const int64_t * const *     edge_len_Bs,
                 const int * const *         idx_Bs,
                 contraction_terms<double> * terms,
                 int                         nterms,
                 int                         num_indices,
                 int64_t **                  lda_Bs)
  {
    for (int i = 0; i < nBs; i++) {
      if (edge_len_Bs[i] == nullptr) {
        lda_Bs[i] = nullptr;
        continue;
      }
      lda_Bs[i] = (int64_t *) CTF_int::alloc(sizeof(int64_t) * num_indices);
      lda_Bs[i][idx_Bs[i][0]] = 1;
      for (int j = 1; j < order_Bs[i]; j++) {
        lda_Bs[i][idx_Bs[i][j]] = lda_Bs[i][idx_Bs[i][j-1]] * edge_len_Bs[i][j-1];
      }
    }
    for (int i = 0; i < nterms; i++) {
      if (terms[i].tbuffer_order == -1) continue;
      lda_Bs[nBs+i] = (int64_t *) CTF_int::alloc(sizeof(int64_t) * num_indices);
      //(i,a): i is the fastest moving index
      lda_Bs[nBs+i][terms[i].idx_tbuffer[0]] = 1;
      for (int j = 1; j < terms[i].tbuffer_order; j++) {
        lda_Bs[nBs+i][terms[i].idx_tbuffer[j]] = lda_Bs[nBs+i][terms[i].idx_tbuffer[j-1]] * terms[i].len_idx[j-1];
      }
    }
  }
}