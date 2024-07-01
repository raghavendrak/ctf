#ifndef __CSF_H__
#define __CSF_H__

#include "../tensor/algstrct.h"
#include "../interface/set.h"

namespace CTF_int {
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
      char * cidx;

      CSF() {}

      CSF(const int64_t& npair, CTF::Pair<dtype> * pairs, 
          int order_, int64_t * lens, int * phys_phase_,
          const std::string & cindices = std::string())
      {
        order = order_;
        phys_phase = phys_phase_;
        nnz_local = npair;
        if (nnz_local == 0) {
          nnz_level = new int64_t[order]();
          return;
        }
        cidx = new char[order];
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

        // All modes except mode-0 have idx and the corresponding pointer to the idx of the next level
        // mode-0 has idx and dt stores the data
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
          printf("level 0: ");
          for (int64_t i = st_ptr; i < en_ptr; i++) {
            std::cout << idx[level][i] << " " << dt[i] << " ";
          }
          std::cout << endl;
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
          int64_t end_j = nnz_level[i] + (i != 0 ? 1 : 0);
          std::cout << "ptr[" << i << "]: ";
          for (int64_t j = 0; j < end_j; j++) {
            std::cout << ptr[i][j] << " ";
          }
          std::cout << std::endl;
        }
      }

      ~CSF()
      {
        delete [] nnz_level;
        if (nnz_local == 0) {
          return;
        }
        delete [] cidx;
        for (int i  = 1; i < order; i++) {
          delete [] idx[i];
          delete [] ptr[i];
        }
        delete [] idx[0];
        delete idx;
        delete ptr;
        delete dt;
        delete [] ldas;
      }
  };
}
#endif