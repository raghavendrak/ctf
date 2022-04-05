#ifndef __GEN_CONTRACTION_H__
#define __GEN_CONTRACTION_H__

#include <assert.h>
#include "ctr_tsr.h"
//#include "../interface/tensor.h"

namespace CTF_int {
  class tensor; 
  class topology; 
  class distribution;
  class mapping;
  
  template<typename dtype>
  class CSF;

  /**
   * \brief class for execution distributed gen_contraction of tensors using multilinear kernel
   */
  template <typename dtype>
  class gen_contraction {
    public:
      /** \brief main sparse tensor */
      tensor * A;
      /** \brief operand tensors (output, input) */
      tensor ** Bs;
      /** \brief number of operand tensors (output, input) */
      int nBs;
      /** \brief order of operand tensors (output, input) */
      int * order_Bs;

      /** \brief scaling of A*B */
      char const * alpha;
    
      /** \brief indices of main tensor */
      int * idx_A;
      /** \brief indices of operand tensors */
      int ** idx_Bs;
      /** \brief function to execute on elements */
      // TODO: change it to multivar_function
      bivar_function const * func;

      // TODO: do we need it?
      algstrct const * sr_A;
      algstrct const ** sr_Bs;

      /** \brief lazy constructor */
      gen_contraction(){ idx_A = NULL; idx_Bs = NULL; order_Bs=NULL; alpha=NULL; };
      
      /** \brief destructor */
      ~gen_contraction();

      /** \brief copy constructor \param[in] other object to copy */
      gen_contraction(gen_contraction const & other);
     
      /**
       * \brief constructor definining multilinear gen_contraction with custom function
       * \param[in] A main tensor
       * \param[in] idx_A indices of main tensor
       * \param[in] Bs operand tensors
       * \param[in] idx_Bs indices of operand tensors 
       * \param[in] alpha scaling factor (can be NULL) alpha * A[idx_A] * B[i][idx_B];
       * \param[in] func custom elementwise function 
                      func(A[idx_A],B[i][idx_B])
       */
      gen_contraction(tensor *                  A,
                      int const *               idx_A,
                      tensor **                 Bs,
                      int                       nBs,
                      const int * const *       idx_Bs,
                      char const *              alpha=NULL,
                      bivar_function const *    func=NULL);
      gen_contraction(tensor *                  A,
                      char const *              idx_A,
                      tensor **                 Bs,
                      int                       nBs,
                      const char * const *      idx_Bs,
                      char const *              alpha=NULL,
                      bivar_function const *    func=NULL);


      /** \brief distribute operand tensor along contraction modes */
      /*
      void distribute_operands(const int * const *      rev_idx_map,
                               CTF::Tensor<dtype> **    redist_Bs);
      */
      
      /** \brief run gen_contraction */
      void execute();
  
      void traverse_CSF(CSF<dtype> *          A_tree, 
                        int                   level, 
                        int64_t               pt);
      
  
      void gen_inv_idx(int                    order_A,
                       int const *            idx_A,
                       int                    nBs,
                       int *                  order_Bs,
                       const int * const *    idx_Bs,
                       int *                  order_tot,
                       int ***                idx_arr);
 };
}
#endif
