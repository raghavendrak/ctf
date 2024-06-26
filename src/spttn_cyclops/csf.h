#ifndef __CSF_H__
#define __CSF_H__

#include "../tensor/algstrct.h"
#include "../interface/tensor.h"
#include "../interface/set.h"

namespace CTF {
  template<typename dtype=double>
    class CSF {

      public:
        int64_t ** idx;
        int64_t ** ptr;
        dtype * dt;
        int64_t * ldas;
        int64_t * nnz_level;
        int * phys_phase;
        int order;
        int64_t nnz_local;

        CSF() {}

        CSF(Tensor<dtype> * T)
        {

          nnz_local = T->nnz_loc;
          if (nnz_local == 0) {
            nnz_level = new int64_t[order]();
            return;
          }

          Pair<dtype> * pairs;
          phys_phase = new int[T->order];
          for (int i = 0; i < T->order; i++) {
            phys_phase[i] = T->edge_map[i].calc_phys_phase();
          }
          pairs = (Pair<double> *)T->data;
          order = T->order;
          int64_t * lens = T->lens;

          ldas = new int64_t[order]();
          ldas[0] = 1;
          for (int i = 1; i < order; i++) {
            ldas[i] = ldas[i-1] * lens[i-1];
          }

          nnz_level = new int64_t[order]();
          for (int i = 0; i < order; i++) {
            nnz_level[i]++;
          }
          int64_t prev = pairs[0].k;
          for (int64_t i = 0; i < nnz_local; i++) {
            for (int j = 1; j < order; j++) {
              if (pairs[i].k >= (prev - (prev % ldas[j])) + ldas[j]) {
                nnz_level[j]++;
              }
            }
            prev = pairs[i].k;
          }
          nnz_level[0] = nnz_local;

          // all modes except mode-0 have idx and the corresponding pointer to the idx of the next level
          idx = new int64_t*[order];
          ptr = new int64_t*[order];
          for (int i = 1; i < order; i++) {
            idx[i] = new int64_t[nnz_level[i]];
            ptr[i] = new int64_t[nnz_level[i] + 1];
          }
          idx[0] = new int64_t[nnz_local];
          dt = new dtype[nnz_local];

          // Initialize idx and ptr
          int64_t it[order];
          for (int i = 0; i < order; i++) {
            it[i] = 0;
          }
          // Process the first element
          prev = pairs[0].k;
          for (int j = 0; j < order; j++) {
            int64_t idx_j = (pairs[0].k / ldas[j]) % lens[j];
            idx_j = idx_j / phys_phase[j];
            if (j == 0) dt[it[j]] = pairs[0].d;
            if (j != 0) ptr[j][it[j]] = 0;
            idx[j][it[j]++] = idx_j;
          }
          // Process the remaining elements
          for (int64_t i = 1; i < nnz_local; i++) {
            for (int j = order - 1; j >= 1; j--) {
              int64_t idx_j = (pairs[i].k / ldas[j]) % lens[j];
              idx_j = idx_j / phys_phase[j];
              if (pairs[i].k >= (prev - (prev % ldas[j])) + ldas[j]) {
                ptr[j][it[j]] = it[j-1];
                idx[j][it[j]++] = idx_j;
              }
            }
            int64_t idx_j = (pairs[i].k / ldas[0]) % lens[0];
            idx_j = idx_j / phys_phase[0];
            dt[it[0]] = pairs[i].d;
            idx[0][it[0]++] = idx_j;
            prev = pairs[i].k;
          }
          for (int j = order - 1; j >= 1; j--) {
            ptr[j][it[j]] = it[j-1];
          }
        }

        int64_t get_child_ptr(int level,
            int64_t pt)
        {
          return ptr[level][pt];
        }

        int64_t get_idx(int level,
            int64_t pt)
        {
          return idx[level][pt];
        }

        int64_t num_children(int level,
            int64_t pt)
        {
          return (ptr[level][pt+1] - ptr[level][pt]);
        }

        dtype get_data(int64_t pt)
        {
          return dt[pt];
        }


        void traverse_CSF(int64_t st_ptr,
            int64_t en_ptr,
            int level)
        {
          if (level == 0) {
            std::cout << "level 0: ";
            for (int64_t i = st_ptr; i < en_ptr; i++) {
              std::cout << idx[level][i] << " " << dt[i] << " "; 
            }
            std::cout << std::endl;
            return;
          }
          std::cout << "level " << level << " st_ptr: " << st_ptr << " en_ptr: " << en_ptr << std::endl;
          for (int64_t i = st_ptr; i < en_ptr; i++) {
            std::cout << "idx[" << level << "][" << i << "]: " << idx[level][i] << std::endl;
            traverse_CSF(ptr[level][i], ptr[level][i+1], level-1);
          }
        }

        void print()
        {
          for (int i = order - 1; i >= 0; i--) {
            std::cout << "level " << i << " nnz_level[" << i << "]: " << nnz_level[i] << std::endl;
            std::cout << "idx[" << i << "]: ";
            for (int64_t j = 0; j < nnz_level[i]; j++) {
              std::cout << idx[i][j] << " ";
            }
            std::cout << std::endl;
            if (i != 0) {
              std::cout << "ptr[" << i << "]: ";
              for (int64_t j = 0; j < nnz_level[i] + 1; j++) {
                std::cout << ptr[i][j] << " ";
              }
              std::cout << std::endl;
            }
          }
        }

        ~CSF()
        {
          delete [] nnz_level;
          if (nnz_local == 0) {
            return;
          }
          for (int i  = 1; i < order; i++) {
            delete [] idx[i];
            delete [] ptr[i];
          }
          delete [] idx[0];
          delete idx;
          delete ptr;
          delete dt;
          delete [] ldas;
          delete [] phys_phase;
        }
    };
}

#endif
