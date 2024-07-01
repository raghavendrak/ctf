#ifndef __EXECUTE_KERNEL_H__
#define __EXECUTE_KERNEL_H__

#include "prepare_kernel.h"
namespace CTF_int{
  template<typename dtype>
  class CSF;
  template <typename dtype>
  class contraction_terms {
    public:
      bool  *  in_op_idx;
      bool  *  in_term_idx;
      bool  *  Bs_in_term;
      dtype *  tbuffer;
      int      tbuffer_order;
      int   *  idx_tbuffer;
      int   *  len_idx;
      int   *  rev_idx_tbuffer;
      int64_t  tbuffer_sz;

      int      inner_idx;
      int      inner_rev_idx;
      // reset the intermediate tensor
      int      reset_idx;
      int      sparse_idx;
      int      dense_idx;
      int      blas_idx;
      int  *   break_rec_idx;

      // blas operands, can be INTERMEDIATE_TENSOR, MAIN_TENSOR, or INP_B
      int      blas_ops[3];
      // prepare BLAS kernels
      int      blas_kernel;
      int      blas_B_ids[3];
      // if Level 1 BLAS, ALPHA will point to the B operand that holds the value
      // if Level 2 BLAS, ALPHA will hold the value 
      double   ALPHA;
      // Level 1 BLAS
      int      X;
      int      Y;
      int      INCX;
      int      INCY;
      // Level 2 BLAS
      // M X N: rows X columns
      // CTF interfaces with the Fortran BLAS (either by using underscore or calling Dxxx directly)
      // LDA = M, and the intermediate tensor is stored in column major order
      int      A;
      int      LDA;
      double   BETA;
      int      M;
      int      N;

      bool  *  dense_sp_loop;
      int      dense_sp_loop_in_term;

      int   *  index_order;
      int   *  rev_index_order;
      int      index_order_sz;

      // chaining terms; input buffer id: term id where the buffer is produced that this term consumes
      int      inp_buf_id;
      int      out_buf_id;
  
      contraction_terms(int num_indices, int num_Bs)
      {
        in_op_idx = (bool *)CTF_int::alloc(sizeof(bool) * num_indices);
        std::fill_n(in_op_idx, num_indices, false);
        in_term_idx = (bool *)CTF_int::alloc(sizeof(bool) * num_indices);
        std::fill_n(in_term_idx, num_indices, false);
        // over allocating (tbuffer might not have all the indices)
        idx_tbuffer = (int *)CTF_int::alloc(sizeof(int) * num_indices);
        std::fill_n(idx_tbuffer, num_indices, -1);
        len_idx = (int *)CTF_int::alloc(sizeof(int) * num_indices);
        std::fill_n(len_idx, num_indices, -1);
        Bs_in_term = (bool *)CTF_int::alloc(sizeof(bool) * (num_Bs+1));
        std::fill_n(Bs_in_term, num_Bs+1, false);
        tbuffer = nullptr;
        tbuffer_order = -1;
        rev_idx_tbuffer = (int *)CTF_int::alloc(sizeof(int) * num_indices);
        std::fill_n(rev_idx_tbuffer, num_indices, -1);
        std::fill_n(blas_ops, 3, -1);
        std::fill_n(blas_B_ids, 3, -1);

        dense_sp_loop = nullptr;
        dense_sp_loop_in_term = -1;
        inner_idx = -1;
        inner_rev_idx = -1;
        reset_idx = -1;
        dense_idx = -1;
        sparse_idx = -1;
        blas_idx = -1;
        tbuffer_sz = -1;
        break_rec_idx = (int *)CTF_int::alloc(sizeof(int) * (num_indices+1));

        index_order = (int *)CTF_int::alloc(sizeof(int) * (num_indices+1));
        std::fill_n(index_order, (num_indices+1), -1);
        rev_index_order = (int *)CTF_int::alloc(sizeof(int) * num_indices);
        std::fill_n(rev_index_order, num_indices, -1);
        index_order_sz = -1;

        inp_buf_id = -1;
        out_buf_id = -1;
      }

      ~contraction_terms()
      {
        free(in_op_idx);
        free(in_term_idx);
        free(idx_tbuffer);
        free(len_idx);
        free(Bs_in_term);
        free(rev_idx_tbuffer);
        free(break_rec_idx);
      }
  };

  void spA_dnBs_gen_ctr(char const *                alpha,
                        CSF<double> *               A_tree,
                        algstrct const *            sr_A,
                        int                         order_A,
                        int const *                 idx_map_A,
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
                        bivar_function const *      func);

  void call_blas(char const *                alpha,
                    CSF<double> *               A_tree,                                
                    algstrct const *            sr_A,
                    int                         order_A,
                    int const *                 idx_map_A,
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
                    int64_t                     tree_pt_st,
                    int64_t                     tree_pt_en,
                    contraction_terms<double> * terms,
                    int                         nterms,
                    int *                       active_terms_ind,
                    int                         num_active_terms,
                    char **                     tbuffer,
                    int                         tree_level,
                    int                         level,
                    contraction_terms<double> & term);

  void call_ctrloop(char const *                alpha,
                    CSF<double> *               A_tree,                                
                    algstrct const *            sr_A,
                    int                         order_A,
                    int const *                 idx_map_A,
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
                    int64_t                     tree_pt_st,
                    int64_t                     tree_pt_en,
                    contraction_terms<double> * terms,
                    int                         nterms,
                    int *                       active_terms_ind,
                    int                         num_active_terms,
                    char **                     tbuffer,
                    int                         tree_level,
                    int                         level);

  void calc_ldas(int                         nBs,
                 int *                       order_Bs,
                 const int64_t * const *     edge_len_Bs,
                 const int * const *         idx_Bs,
                 contraction_terms<double> * terms,
                 int                         nterms,
                 int                         num_indices,
                 int64_t **                  lda_Bs);
#ifdef YET_TO_COMPILE
#ifdef OLD_CODE

   /*
   void gen_inv_idx(int                     order_A,
                    int const *             idx_A,
                    int                     nBs,
                    int *                   order_Bs,
                    const int * const *     idx_Bs,
                    int *                   order_tot,
                    int ***                 idx_arr);
    */

   /*
    * this function can be invoked from gen_contraction::execute()
    * definition visible to the compiler
    */
   /*
   template <typename dtype> 
   void traverse_CSF(CSF<dtype> * A_tree) {}
   */

    void dnBs_loop(char const *              alpha,
                           int                       nBs,
                           char **      Bs,
                           const algstrct * const *  sr_Bs,
                           const int64_t * const *   lda_Bs,
                           bivar_function const *    func,
                           const int * const *       rev_idx_map,
                           double                    dt_AB,
                           std::vector<std::pair<int, int64_t> >          nidx_Bs,
                           std::vector<int>          tidx_Bs);
   

    void optimize_contraction_order(std::vector<std::pair<int, int64_t> > & nidx_Bs,
                                  int                 nBs,
                                  const int * const *       rev_idx_map,
                                  std::vector<int> &         tidx_Bs);

#endif
#endif
}
#endif
