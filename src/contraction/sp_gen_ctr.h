#ifndef __SP_GEN_CTR_H__
#define __SP_GEN_CTR_H__

#include "gen_contraction.h"
namespace CTF_int{
  void spA_dnBs_gen_ctr(char const *              alpha,
                        CSF<double> *       A,
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
                        bivar_function const *    func);

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

}
#endif
