#include "../shared/iter_tsr.h"
#include <algorithm>
#include <limits.h>
#include <utility>
#include "sp_gen_ctr.h"
#include "../shared/offload.h"
#include "../shared/util.h"
#include "../sparse_formats/csf.h"
#include <queue>

namespace CTF_int{
  
  template class CSF<double>;

  template<int level>                                                                          
  void spA_dnBs_ctrloop(char const *              alpha,
                        CSF<double> *       A_tree,                                
                        algstrct const *          sr_A,
                        int                       order_A,
                        int const *               idx_map_A,
                        int                       nBs,
                        char **      Bs,
                        const algstrct * const *  sr_Bs,
                        int *                     order_Bs,
                        const int64_t **          len_Bs,
                        const int64_t * const *   lda_Bs,
                        bivar_function const *    func,
                        const int * const *       rev_idx_map,
                        int                       idx_max,
                        int64_t                   pt,
                        std::vector<int> &        tidx_Bs,
                        std::vector<std::pair<int, int64_t> > & nidx_Bs,
                        std::vector<int> &        mBs) 
  {
    int64_t idx_idim = A_tree->get_idx(level, pt);
    int64_t idim = level; // TODO: level would work because A is processed first; but make it generic
    char *tBs[nBs];
    for (int j = 0; j < nBs; j++) {
      if (rev_idx_map[j][idx_map_A[idim]] != -1) {
        tBs[j] = (char *)((double *)Bs[j] + lda_Bs[j][rev_idx_map[j][idx_map_A[idim]]] * idx_idim);
      }
      else {
        tBs[j] = Bs[j];
      }
    }
    
    int64_t imax = A_tree->num_children(level, pt);
    int64_t child_pt = A_tree->get_child_ptr(level, pt);
    for (int64_t i = child_pt; i < (child_pt + imax); i++) {
      spA_dnBs_ctrloop<level-1>(alpha, A_tree, sr_A, order_A, idx_map_A, nBs, tBs, sr_Bs, order_Bs, len_Bs, lda_Bs, func, rev_idx_map, idx_max, i, tidx_Bs, nidx_Bs, mBs); 
    }
  }
  
  template<>                                                                          
  void spA_dnBs_ctrloop<0>(char const *              alpha,
                           CSF<double> *       A_tree,
                           algstrct const *          sr_A,
                           int                       order_A,
                           int const *               idx_map_A,
                           int                       nBs,
                           char **      Bs,
                           const algstrct * const *  sr_Bs,
                           int *                     order_Bs,
                           const int64_t **          len_Bs,
                           const int64_t * const *   lda_Bs,
                           bivar_function const *    func,
                           const int * const *       rev_idx_map,
                           int                       idx_max,
                           int64_t                   pt,
                           std::vector<int> &        tidx_Bs,
                           std::vector<std::pair<int, int64_t> > & nidx_Bs,
                           std::vector<int> &        mBs) 
  {
    int level = 0;
    int64_t idx_idim = A_tree->get_idx(level, pt);
    int64_t idim = level; // TODO: level would work because A is processed first; but make it generic
    char *tBs[nBs];
    for (int j = 0; j < nBs; j++) {
      if (rev_idx_map[j][idx_map_A[idim]] != -1) {
        tBs[j] = (char *)((double *)Bs[j] + lda_Bs[j][rev_idx_map[j][idx_map_A[idim]]] * idx_idim);
      } 
      else {
        tBs[j] = Bs[j];
      }
    }
    
    double dt_AB = A_tree->get_data(pt);
    
    // contract operands that are done along with the tree traversal 
    for (int i = 0; i < mBs.size(); i++) {
      dt_AB *= *(double *)tBs[mBs[i]];
    }

    dnBs_loop(alpha, nBs, tBs, sr_Bs, lda_Bs, func, rev_idx_map, dt_AB, nidx_Bs, tidx_Bs); 
  }

  void dnBs_loop(char const *              alpha,
                           int                       nBs,
                           char **      Bs,
                           const algstrct * const *  sr_Bs,
                           const int64_t * const *   lda_Bs,
                           bivar_function const *    func,
                           const int * const *       rev_idx_map,
                           double                    dt_AB,
                           std::vector<std::pair<int, int64_t> >          nidx_Bs,
                           std::vector<int>          tidx_Bs)
  {

    if (nidx_Bs.size() == 0) {
      //IASSERT(tidx_Bs[nBs-1] == 0);
      double * dt = (double *)Bs[nBs-1];
      *dt = *dt + dt_AB;
      return;
    }
    double recv_dt_AB = dt_AB;
    char *tBs[nBs];
    // get the index to be processed
    int idx = nidx_Bs[0].first;
    int64_t len_idx = nidx_Bs[0].second;
    // TODO: make it pos, so that the erase can be replaced
    nidx_Bs.erase(nidx_Bs.begin());


    for (int j = 0; j < nBs; j++) {
      if (rev_idx_map[j][idx] != -1) {
        tidx_Bs[j]--;
        //IASSERT(tidx_Bs[j] >= 0);
      }
    }

    std::vector<int> mBs;
    for (int j = 0; j < (nBs-1); j++) {
      if (rev_idx_map[j][idx] != -1) {
        mBs.push_back(j);
      }
      else {
        tBs[j] = Bs[j];
      }
    }
    if (nidx_Bs.size() == 0) {
      // find all the Bs that match this idx
      tBs[nBs-1] = Bs[nBs-1];
      for (int64_t i = 0; i < len_idx; i++) {
        double mul_dt_AB = recv_dt_AB;
        // advance output
        if (rev_idx_map[nBs-1][idx] != -1) {
          tBs[nBs-1] = (char *)((double *)Bs[nBs-1] + lda_Bs[nBs-1][rev_idx_map[nBs-1][idx]] * i);
        }
        // advance operand tensors
        for (int k = 0; k < mBs.size(); k++) {
          int j = mBs[k];
          tBs[j] = (char *)((double *)Bs[j] + lda_Bs[j][rev_idx_map[j][idx]] * i);
          mul_dt_AB = mul_dt_AB * *(double *)tBs[j];
        }
        // end of the world: update output operand
        double *dt = (double *)tBs[nBs - 1];
        *dt = *dt + mul_dt_AB;
      }
      return;
    }

    tBs[nBs-1] = Bs[nBs-1];
    for (int64_t i = 0; i < len_idx; i++) {
      // advance output tensor
      if (rev_idx_map[nBs-1][idx] != -1) {
        tBs[nBs-1] = (char *)((double *)Bs[nBs-1] + lda_Bs[nBs-1][rev_idx_map[nBs-1][idx]] * i);
      }
      double child_dt_AB = recv_dt_AB;
      for (int k = 0; k < mBs.size(); k++) {
        int j = mBs[k];  
        tBs[j] = (char *)((double *)Bs[j] + lda_Bs[j][rev_idx_map[j][idx]] * i);
        if (tidx_Bs[j] == 0) {
          // update dt_AB only for the ones that got completed in this recursive call
          child_dt_AB = child_dt_AB * *(double *)tBs[j];  
        }
      }
      dnBs_loop(alpha, nBs, tBs, sr_Bs, lda_Bs, func, rev_idx_map, child_dt_AB, nidx_Bs, tidx_Bs); 
    }
  }

  void optimize_contraction_order(std::vector<std::pair<int, int64_t> > &  nidx_Bs,
                                  int                 nBs,
                                  const int * const *       rev_idx_map,
                                  std::vector<int> &         tidx_Bs)
  {
    // TODO: try permutations of which ordering is better
    // <count, <idx, len_idx> >
    std::priority_queue<std::pair<int, std::pair<int, int64_t> > > pq;
    while(!nidx_Bs.empty()) {
      std::pair<int, int64_t> il = nidx_Bs.back();
      nidx_Bs.pop_back();
      int count_cBs = 0;
      for (int i = 0; i < nBs; i++) {
        if (rev_idx_map[i][il.first] != -1) {
          if ((tidx_Bs[i] - 1) == 0) {
            // taking this index allows this B to complete
            count_cBs++;
          }
        }
      }
      pq.push(std::make_pair(count_cBs, il));
    }
    while(!pq.empty()) {
      nidx_Bs.push_back(pq.top().second);
      pq.pop();
    }
  }
  
  template<>                                                                          
  void spA_dnBs_ctrloop< MAX_ORD >
                      (char const *              alpha,
                       CSF<double> *       A_tree,                                
                       algstrct const *          sr_A,
                       int                       order_A,
                       int const *               idx_map_A,
                       int                       nBs,
                       char **      Bs,
                       const algstrct * const *  sr_Bs,
                       int *                     order_Bs,
                       const int64_t **          len_Bs,
                       const int64_t * const *   lda_Bs,
                       bivar_function const *    func,
                       const int * const *       rev_idx_map,
                       int                       idx_max,
                       int64_t                   pt,
                       std::vector<int> &        tidx_Bs,
                       std::vector<std::pair<int, int64_t> > & nidx_Bs,
                       std::vector<int> &        mBs); 

  void spA_dnBs_gen_ctr(char const *              alpha,
                        CSF<double> *       A_tree,
                        algstrct const *          sr_A,
                        int                       order_A,
                        int const *               idx_map_A,
                        int                       nBs,
                        char **      Bs,
                        const algstrct * const *  sr_Bs,
                        int *                     order_Bs,
                        const int64_t **          len_Bs,
                        const int64_t * const *   edge_len_Bs,
                        const int * const *       rev_idx_map,
                        int                       idx_max,
                        bivar_function const *    func) 
  {

    int64_t ** lda_Bs;
    std::vector<int> mBs;

    ASSERT(idx_max <= MAX_ORD);
    if (idx_max == 0) {
      // TODO
      ASSERT(0);
      return;
    }

    lda_Bs = (int64_t **) CTF_int::alloc(sizeof(int64_t *) * nBs);
    for (int i = 0; i < nBs; i++) {
      lda_Bs[i] = (int64_t *) CTF_int::alloc(sizeof(int64_t) * order_Bs[i]);
      lda_Bs[i][0] = 1;
      for (int j = 1; j < order_Bs[i]; j++) {
        lda_Bs[i][j] = lda_Bs[i][j-1] * edge_len_Bs[i][j-1];
      }
    }

    // traversed Bs idx when tree traversal is done
    std::vector<int> tidx_Bs(nBs);
    for (int i = 0; i < nBs; i++) {
      tidx_Bs[i] = order_Bs[i];
    }
    for (int i = 0; i < nBs; i++) {
      for (int j = 0; j < order_A; j++) {
        // TODO: this works because the indices of A are assigned iidx first; make it generic
        if (rev_idx_map[i][idx_map_A[j]] != -1) {
          tidx_Bs[i]--;
          if (tidx_Bs[i] == 0 && i < (nBs-1)) {
            mBs.push_back(i);
          }
          IASSERT(tidx_Bs[i] >= 0);
        }
      }
    }
    // Bs idx that are yet to be processed
    // (idx_B, len_idx_B)
    std::vector<std::pair<int, int64_t> > nidx_Bs;
    for (int i = order_A; i < idx_max; i++) {
      // find the idx in B
      int64_t len_idx_B = -1;
      for (int j = 0; j < (nBs-1); j++) {
        int pos_idx_inB = rev_idx_map[j][i];      
        if (pos_idx_inB != -1) {
          if (len_idx_B == -1) {
            len_idx_B = len_Bs[j][pos_idx_inB];
          }
          else {
            if (len_idx_B != len_Bs[j][pos_idx_inB]) {
              // idx dimensions don't match
              IASSERT(0);
            }
          }
        }
      }
      IASSERT(len_idx_B != -1);
      nidx_Bs.push_back(std::make_pair(i, len_idx_B));
    }
    optimize_contraction_order(nidx_Bs, nBs, rev_idx_map, tidx_Bs);
 
    int level = order_A - 1;
    for (int64_t i = 0; i < A_tree->nnz_level[level]; i++) {
      SWITCH_ORD_CALL(spA_dnBs_ctrloop, level, alpha, A_tree, sr_A, order_A, idx_map_A, nBs, Bs, sr_Bs, order_Bs, len_Bs, lda_Bs, func, rev_idx_map, idx_max, i, tidx_Bs, nidx_Bs, mBs); 
    }

    for (int i = 0; i < nBs; i++) {
      cdealloc(lda_Bs[i]);
    }
    cdealloc(lda_Bs);
  }
}