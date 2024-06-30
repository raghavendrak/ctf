#ifndef __CONTRACTION_PATH_INDEX_ORDER_H__
#define __CONTRACTION_PATH_INDEX_ORDER_H__

#include <cstdint>
#include <sys/types.h>

namespace CTF_int {
  struct CPCache {
    uint16_t inds;
    int8_t cost;
    std::vector<std::pair<uint16_t, uint16_t> > orders;

    CPCache() : inds(0), cost(-1) {}

    void print() 
    {
      std::cout << " cost: " << (int)cost << std::endl;
      int x = 0;
      for (int i = 0; i < 16; i++) {
        if (inds & (1 << i)) {
          std::cout << x << " ";
        }
        x++;
      }
      std::cout << std::endl;
      for (auto & p : orders) {
        std::cout << "  " << p.first << " " << p.second << std::endl;
      }
    }
  };

  struct CTerm {
    uint16_t ta;
    uint16_t tb;
    uint16_t tab;
    uint16_t inds;

    void print()
    {
      std::cout << " ta: " << ta << " tb: " << tb << " tab: " << tab << " inds: " << inds << std::endl;
    }
  };


  class contraction_path {
    public:
      CPCache * cp_cache;
      uint16_t op_inds;
      int ntensors;
      uint8_t numones[65536];
      int64_t num_pushes;

      contraction_path()
      {
        cp_cache = nullptr;
        op_inds = 0;
        ntensors = 0;
        num_pushes = 0;
        popcount_init(numones);
      }

      contraction_path(CPCache * cp_cache_,
          uint16_t op_inds_,
          int ntensors_)
      {
        cp_cache = cp_cache_;
        op_inds = op_inds_;
        ntensors = ntensors_;
        num_pushes = 0;
        popcount_init(numones);
      }

      void contract(uint16_t ta, uint16_t tb,
          uint16_t & cinds, 
          int8_t & flops)
      {
        uint16_t nt = ta | tb;
        uint16_t rinds = 0;
        assert(cp_cache[ta].inds != 0);
        assert(cp_cache[tb].inds != 0);
        cinds = cp_cache[ta].inds | cp_cache[tb].inds;
        uint16_t tid = 1;
        for (int i = 1; i <= ntensors; i++) {
          if ((ta & tid) == 0 && (tb & tid) == 0) {
            rinds |= cp_cache[tid].inds; 
          }
          tid = 1 << i;
        }
        rinds |= op_inds;
        flops = numones[cinds];
        cinds = cinds & rinds;
      }

      int8_t optimal_contraction_paths(uint16_t * lt, int sz)
      {
        uint16_t nt = 0;
        for (int i = 0; i < sz; i++) nt |= lt[i];
        if (cp_cache[nt].cost != -1) return cp_cache[nt].cost;
        // NOTE: just an optimization, removing the below check will not affect correctness
        if (sz == 2) {
          // contract these two tensors, create indices and note their order
          assert ((lt[0] & lt[1]) == 0);
          if (cp_cache[nt].cost == -1) {
            uint16_t cinds;
            int8_t flops;
            contract(lt[0], lt[1], cinds, flops);
            cp_cache[nt].inds = cinds;
            cp_cache[nt].cost = flops;
            cp_cache[nt].orders.push_back(std::make_pair(lt[0], lt[1]));
            num_pushes++;
          }
          return cp_cache[nt].cost;
        }
        int onetoR[sz+1];
        bool picked[sz];
        for (int i = 0; i < sz; i++)  picked[i] = false;
        uint16_t ll1[sz];
        uint16_t ll2[sz];
        int8_t c1, c2, ncost;
        uint16_t cinds;
        int8_t flops;
        for (int r = 1; r <= sz/2; r++) {
          for (int i = 1; i <= r; i++) onetoR[i] = i;
          uint16_t nt1 = 0;
          uint16_t nt2 = 0;

          int * st = onetoR + 1;
          int * en = onetoR + r + 1;
          // first combination
          for (int * x = st; x != en; x++) {
            int id = *x - 1;
            ll1[id] = lt[id];
            picked[id] = true;
            nt1 |= lt[id];
          }
          int k = 0;
          for (int i = 0; i < sz; i++) {
            if (!picked[i]) {
              ll2[k++] = lt[i];
              nt2 |= lt[i];
            }
            picked[i] = false;
          }

          c1 = optimal_contraction_paths(ll1, r);
          c2 = optimal_contraction_paths(ll2, sz-r);
          contract(nt1, nt2, cinds, flops);
          ncost = c1 + c2 + flops;
          if (cp_cache[nt].cost == -1) {
            cp_cache[nt].cost = ncost;
            cp_cache[nt].inds = cinds;
            cp_cache[nt].orders.push_back(std::make_pair(nt1, nt2));
            num_pushes++;
          }
          else {
            assert (cp_cache[nt].inds == cinds);
            if (ncost < cp_cache[nt].cost) {
              // found a lower cost
              cp_cache[nt].cost = ncost;
              cp_cache[nt].orders.push_back(std::make_pair(nt1, nt2));
              num_pushes++;
            }
            else if (ncost == cp_cache[nt].cost) {
              cp_cache[nt].orders.push_back(std::make_pair(nt1, nt2));
              num_pushes++;
            }
            else {
              // found a higher cost
              cp_cache[nt].orders.push_back(std::make_pair(nt1, nt2));
              num_pushes++;
            }
          }

          // remaining combinations for a specific r
          while((*st) != sz-r+1) {
            nt1 = 0;
            nt2 = 0;
            int * mt = en;
            do {
              mt--;
            } while (*mt == sz-(en-mt)+1);
            (*mt)++;
            while (++mt != en) *mt = *(mt-1)+1;

            int k = 0; 
            for (int * x = st; x != en; x++) {
              int id = *x - 1;
              ll1[k++] = lt[id];
              nt1 |= lt[id];
              picked[id] = true;
            }
            k = 0;
            for (int i = 0; i < sz; i++) {
              if (!picked[i]) {
                ll2[k++] = lt[i];
                nt2 |= lt[i];
              }
              picked[i] = false;
            }

            c1 = optimal_contraction_paths(ll1, r);
            c2 = optimal_contraction_paths(ll2, sz-r);
            contract(nt1, nt2, cinds, flops);
            ncost = c1 + c2 + flops;
            if (cp_cache[nt].cost == -1) {
              cp_cache[nt].cost = ncost;
              cp_cache[nt].inds = cinds;
              cp_cache[nt].orders.push_back(std::make_pair(nt1, nt2));
              num_pushes++;
            }
            else {
              assert (cp_cache[nt].inds == cinds);
              if (ncost < cp_cache[nt].cost) {
                // found a lower cost
                cp_cache[nt].cost = ncost;
                cp_cache[nt].orders.push_back(std::make_pair(nt1, nt2));
                num_pushes++;
              }
              else if (ncost == cp_cache[nt].cost) {
                cp_cache[nt].orders.push_back(std::make_pair(nt1, nt2));
                num_pushes++;
              }
              else {
                cp_cache[nt].orders.push_back(std::make_pair(nt1, nt2));
                num_pushes++;
              }
            }
          }
        }
        return cp_cache[nt].cost;
      }

      std::vector<std::vector<CTerm> > enumerate_all_paths(uint16_t tid)
      {
        std::vector<std::vector<CTerm> > paths;
        std::vector<CTerm> path;
        // if this tensor is the result of contraction of two input tensors
        if (numones[tid] == 2) {
          assert(cp_cache[tid].orders.size() == 1);
          CTerm term;
          term.ta = cp_cache[tid].orders[0].first;
          term.tb = cp_cache[tid].orders[0].second;
          term.tab = tid;
          term.inds = cp_cache[tid].inds;
          path.push_back(term);
          paths.push_back(path);
          return paths;
        }
        for (int i = 0; i < cp_cache[tid].orders.size(); i++) {
          CTerm term;
          term.ta = cp_cache[tid].orders[i].first;
          term.tb = cp_cache[tid].orders[i].second;
          term.tab = tid;
          term.inds = cp_cache[tid].inds;
          assert(!(cp_cache[tid].orders[i].first == 1 && cp_cache[tid].orders[i].second == 1));
          // if it is made up of only one tensor
          if (numones[cp_cache[tid].orders[i].first] == 1) {
            std::vector<std::vector<CTerm> > rpaths = enumerate_all_paths(cp_cache[tid].orders[i].second);
            for (int r = 0; r < rpaths.size(); r++) {
              rpaths[r].push_back(term);
              paths.push_back(rpaths[r]);
            }
          }
          else if (numones[cp_cache[tid].orders[i].second] == 1) {
            std::vector<std::vector<CTerm> > lpaths = enumerate_all_paths(cp_cache[tid].orders[i].first);
            for (int l = 0; l < lpaths.size(); l++) {
              lpaths[l].push_back(term);
              paths.push_back(lpaths[l]);
            }
          }
          else {
            std::vector<std::vector<CTerm> > lpaths = enumerate_all_paths(cp_cache[tid].orders[i].first);
            std::vector<std::vector<CTerm> > rpaths = enumerate_all_paths(cp_cache[tid].orders[i].second);
            for (int l = 0; l < lpaths.size(); l++) {
              for (int r = 0; r < rpaths.size(); r++) {
                path = lpaths[l];
                path.insert(path.end(), rpaths[r].begin(), rpaths[r].end());
                path.push_back(term);
                paths.push_back(path);
              }
            }
          }
        }
        return paths;
      }
  };

  struct ICache {
    std::vector<std::vector<uint16_t> > inds_order[2];
    int8_t niloops[2];
    int8_t max_buf_sz[2];
    bool computed;

    ICache() {
      niloops[0] = niloops[1] = -1;
      computed = false;
      max_buf_sz[0] = max_buf_sz[1] = INT8_MAX;
    }

    void print_element_in_icache(uint16_t S, uint8_t sT, uint8_t eT) 
    {
      uint8_t nterms_interval = eT - sT + 1;
      for (int i = 0; i < nterms_interval; i++) {
        std::cout << "icache[" << S << "][" << (int)sT+i << "]: ";
        for (int k = 0; k < inds_order[0][i].size(); k++) {
          std::cout << inds_order[0][i][k] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "niloops: " << (int)niloops[0] << " max_buf_sz: " << (int)max_buf_sz[0] << std::endl;
    }

    void init_element_in_icache(uint16_t S, uint8_t sT, uint8_t eT, uint16_t * term_inds, uint8_t * numones) 
    {
      uint8_t nterms_interval = eT - sT + 1;
      for (int j = 0; j < 2; j++) {
        inds_order[j].reserve(nterms_interval);
      }
      for (int i = 0; i < nterms_interval; i++) {
        uint16_t rem_inds = term_inds[sT+i] & ~S;
        for (int j = 0; j < 2; j++) {
          inds_order[j].push_back(std::vector<uint16_t>());
          inds_order[j][i].reserve(numones[rem_inds]);
        }
      }
    }

    void reset_element_in_icache(uint16_t S, uint8_t sT, uint8_t eT, uint16_t * term_inds) 
    {
      uint8_t nterms_interval = eT - sT + 1;
      for (int i = 0; i < nterms_interval; i++) {
        uint16_t rem_inds = term_inds[sT+i] & ~S;
        for (int j = 0; j < 2; j++) {
          inds_order[j][i].clear();
        }
      }
      for (int j = 0; j < 2; j++) {
        inds_order[j].clear();
        niloops[j] = -1;
        max_buf_sz[j] = INT8_MAX;
      }
      computed = false;
    }
    void sanity_check(uint16_t S, uint8_t sT, uint8_t eT)
    {
      uint8_t nterms_interval = eT - sT + 1;
      for (int i = 0; i < nterms_interval; i++) {
        for (int j = 0; j < 2; j++) {
          for (int k = 0; k < inds_order[j][i].size(); k++) {
            std::cout << "S: " << std::bitset<8>(S) << " inds_order: " << inds_order[j][i][k] << std::endl;
            assert(!(S & (1 << (int)log2(inds_order[j][i][k]))));
          }
        }
      }
    }
  };


  class local_index_order {
    public:
      ICache *** icache;
      int nterms;
      uint16_t all_inds;
      uint16_t sp_inds;
      uint8_t * numones;
      std::vector<CTerm> & path;
      CPCache * cp_cache;
      int8_t thres_buf_sz;
      bool sp_buffer;
      uint16_t nindices;
      uint8_t * gc;
      uint16_t * term_inds;
      int64_t icache_sz;

      local_index_order(int nterms_, 
          uint16_t all_inds_,
          uint16_t sp_inds_,
          uint8_t * numones_, 
          std::vector<CTF_int::CTerm>& path_, 
          CPCache * cp_cache_,
          int8_t thres_buf_sz_,
          bool sp_buffer_)
        : nterms(nterms_), all_inds(all_inds_), sp_inds(sp_inds_), numones(numones_), path(path_), cp_cache(cp_cache_), thres_buf_sz(thres_buf_sz_), sp_buffer(sp_buffer_)
      {
        nindices = numones[all_inds];
        term_inds = new uint16_t[nterms];
        icache_sz = 1 << nindices;
        gc = new uint8_t[nterms];

        // init index_order cache
        icache = new ICache**[icache_sz];
        for (int i = 0; i < icache_sz; i++) {
          icache[i] = new ICache*[nterms];
          for (int j = 0; j < nterms; j++) {
            icache[i][j] = new ICache[nterms];
          }
        }
        // populate gc[] and term_inds[]
        IASSERT(nterms == path.size());
        int j = 0;
        for (; j < (path.size()-1); j++) {
          uint16_t generator = path[j].tab;
          int k = j+1;
          for (; k < path.size(); k++) {
            if (!(path[k].ta & generator) || !(path[k].tb & generator)) {
              gc[j] = k;
              break;
            }
          }
          assert (k < path.size());
          term_inds[j] = cp_cache[path[j].ta].inds | cp_cache[path[j].tb].inds;
        }
        //term_inds[j] = paths[i][j].inds;
        term_inds[j] = cp_cache[path[j].ta].inds | cp_cache[path[j].tb].inds;
      }

      bool apply_constraints(uint16_t S, uint8_t sT, uint8_t eT, uint16_t q, uint8_t constraints_to_apply)
      {
        // Constraint 1: is there any sparse index with higher index_val in the rem_inds?
        // NOTE: this assumes that the fastest moving index in the sparse tensor has index value 0 
        // i.e. T["ijk"] is interpreted as T[012]
        if (constraints_to_apply & 1) {
          if (q & sp_inds) {
            // collect all the remaining indices in this list of terms
            uint16_t rem_inds = 0;
            for (int i = sT; i <= eT; i++) {
              rem_inds |= term_inds[i];
            }
            rem_inds &= ~S;
            while (rem_inds) {
              uint16_t sp_i = 0;
              while (!(rem_inds & (1 << sp_i))) sp_i++;
              uint16_t sp = 1 << sp_i;
              rem_inds &= ~sp;
              if ((sp & sp_inds) && sp > q) {
                return true;
              }
            }
          }
        }
        // Constraint 2: are sparse indices traversed in the order of CSF?
        if (constraints_to_apply & 2) {
          uint16_t root_q = q << 1;
          if (q & sp_inds) {
            // iterate over the sparse indices and any index with higher index_val should already be in S
            while (root_q & sp_inds) {
              if (!(root_q & S)) {
                return true;
              }
              root_q = root_q << 1;
            }
          }
        }
        return false;
      }

      int8_t max_buf_cost_to_split_interval(uint16_t S, uint8_t sT, uint8_t eT, uint8_t k)
      {
        int8_t max_buf_cost = 0;
        for (int i = sT; i <= k; i++) {
          if (gc[i] > k && gc[i] <= eT) {
            uint16_t common_inds = term_inds[i] & term_inds[gc[i]];
            uint16_t buf_inds = common_inds & ~S;
            max_buf_cost = std::max(max_buf_cost, (int8_t)numones[buf_inds]);
          }
        }
        return max_buf_cost;
      }

      bool is_sp_buffer(uint16_t S, uint8_t sT, uint8_t eT, uint8_t k)
      {
        for (int i = sT; i <= k; i++) {
          if (gc[i] > k && gc[i] <= eT) {
            uint16_t common_inds = term_inds[i] & term_inds[gc[i]];
            uint16_t buf_inds = common_inds & ~S;
            if (buf_inds & sp_inds) {
              return true;
            }
          }
        }
        return false;
      }

      // (sT, eT) interval of terms
      void io_cost (uint16_t S,
          uint8_t sT, uint8_t eT)
      {
        if (icache[S][sT][eT].computed == true) {
          return;
        }

        if (eT-sT+1 == 2) {
          // check if the first two terms have the same indices remaining
          uint16_t rem_inds = term_inds[sT] & ~S;
          //assert((term_inds[sT+1] & ~S) != rem_inds);
        }

        // initialize cache for this interval and S
        icache[S][sT][eT].init_element_in_icache(S, sT, eT, term_inds, numones);

        uint16_t rem_inds = term_inds[sT] & ~S;
        if (sT == eT) {
          // TODO: check if there are any sparse indices that fall within the apply_constraints
          int num_rem_inds = numones[rem_inds];
          uint16_t cp_rem_inds = rem_inds;
          uint16_t cp_S = S;
          for (int i = 0; i < num_rem_inds; i++) {
            bool ind_picked = false;
            for (int j = 0; j < nindices; j++) {
              if (cp_rem_inds & (1 << j)) {
                if (apply_constraints(cp_S, sT, eT, (1<<j), 2)) {
                }
                else {
                  cp_rem_inds &= ~(1 << j);
                  cp_S |= (1 << j);
                  ind_picked = true;
                  break;
                }
              }
            }
            if (!ind_picked) {
              // unable to pick an index
              return;
            }
          }

          icache[S][sT][eT].max_buf_sz[0] = icache[S][sT][eT].max_buf_sz[1] = 0;
          assert(icache[S][sT][eT].inds_order[0][0].size() == 0);
          // populate the first index order
          for (int i = 0; i < nindices; i++) {
            if (rem_inds & (1 << i)) {
              icache[S][sT][eT].inds_order[0][0].push_back((1<<i));
            }
          }
          if (icache[S][sT][eT].inds_order[0][0].size() == 0) {
            assert(icache[S][sT][eT].inds_order[1][0].size() == 0);
          }
          else if (icache[S][sT][eT].inds_order[0][0].size() == 1) {
            // populate the second index order
            icache[S][sT][eT].inds_order[1][0].push_back(icache[S][sT][eT].inds_order[0][0][0]);
            assert(icache[S][sT][eT].inds_order[1][0].size() == icache[S][sT][eT].inds_order[0][0].size());
          }
          else {
            // populate the second index order
            // interchange the first two indices in the first two index order and record it as the second index order
            assert(icache[S][sT][eT].inds_order[0][0][0] != icache[S][sT][eT].inds_order[0][0][1]);
            icache[S][sT][eT].inds_order[1][0].push_back(icache[S][sT][eT].inds_order[0][0][1]);
            icache[S][sT][eT].inds_order[1][0].push_back(icache[S][sT][eT].inds_order[0][0][0]);
            icache[S][sT][eT].inds_order[1][0].insert(icache[S][sT][eT].inds_order[1][0].end(), icache[S][sT][eT].inds_order[0][0].begin()+2, icache[S][sT][eT].inds_order[0][0].end());
            assert(icache[S][sT][eT].inds_order[1][0].size() == icache[S][sT][eT].inds_order[0][0].size());
          }
          // find dense indices after all sparse indices are removed and populate independent dense loops
          for (int j = 0; j < 2; j++) {
            int dense_inds = 0;
            for (int i = 0; i < icache[S][sT][eT].inds_order[j][0].size(); i++) {
              if (numones[icache[S][sT][eT].inds_order[j][0][i] & sp_inds] == 0) {
                dense_inds++;
              }
              else {
                dense_inds = 0;
              }
            }
            icache[S][sT][eT].niloops[j] = dense_inds;
          }
          icache[S][sT][eT].computed = true;
          return;
        }

        int8_t niloopss[2];
        niloopss[0] = -1; niloopss[1] = -1;
        int8_t max_buf_szs[2];
        max_buf_szs[0] = -1; max_buf_szs[1] = -1;
        int8_t max_buf_sz = -1;
        std::vector<std::vector<uint16_t> > Ts[2];
        for (int i = 0; i < (eT-sT+1); i++) {
          for (int j = 0; j < 2; j++) {
            Ts[j].push_back(std::vector<uint16_t>());
            uint16_t rem_inds = term_inds[sT+i] & ~S;
            Ts[j][i].reserve(numones[rem_inds]);
          }
        }

        uint16_t iterate_inds = rem_inds;
        uint16_t srem_inds = rem_inds;
        int ninds_sT = numones[iterate_inds];

        if (ninds_sT == 0) {
          assert((eT-sT+1) > 1);
          // TODO
          // all indices have been common to all the terms in the interval till here and the first term has been fully iterated
          io_cost(S, sT+1, eT);
          assert (icache[S][sT+1][eT].computed == true);
          // no split in the tree
          int8_t max_buf_sz_child = icache[S][sT+1][eT].max_buf_sz[0];
          int8_t max_buf_sz_LR = 0; // no cost since no split
          max_buf_sz = std::max(max_buf_sz_LR, max_buf_sz_child);
          if (max_buf_sz <= thres_buf_sz) {
            int which_R = 0;
            int8_t niloops_LR = icache[S][sT+1][eT].niloops[0];
            assert(niloopss[0] == -1);
            niloopss[1] = niloopss[0] = niloops_LR;
            max_buf_szs[1] = max_buf_szs[0] = 0; // scalar
            Ts[0][0].clear();
            Ts[1][0].clear();
            for (int ii = 1; ii < (eT-sT+1); ii++) {
              Ts[0][ii] = icache[S][sT+1][eT].inds_order[which_R][ii-1];
              Ts[1][ii] = icache[S][sT+1][eT].inds_order[which_R][ii-1];
            }
            assert(Ts[0][0].size() == 0);
          }
        }
        for (int i = 0; i < ninds_sT; i++) {
          // get the next index
          uint16_t q_i = 0;
          while (!(iterate_inds & (1 << q_i))) q_i++;
          uint16_t q = 1 << q_i;
          iterate_inds &= ~q;
          rem_inds = srem_inds & ~q;
          uint8_t ii = sT;
          uint8_t k = UINT8_MAX;
          while (ii <= eT) {
            if (term_inds[ii] & q) {
              k = ii;
              ii++;
            }
            else break;
          }
          assert(k != UINT8_MAX);
          for (uint8_t s = sT; s <= k; s++) {
            // can I pick q for the following group of terms?
            if (apply_constraints(S, sT, s, q, UINT8_MAX)) {
              continue;
            }
            if (s == eT) {
              // there is only one tree (no L and R) 
              io_cost(S | q, sT, s);
              if (icache[(S|q)][sT][s].computed == false) {
                // could not find a loop nest within the specified cost
                continue;
              }
              assert (icache[(S|q)][sT][s].computed == true);
              // no split in the tree
              int8_t max_buf_sz_child = icache[(S|q)][sT][s].max_buf_sz[0];
              int8_t max_buf_sz_LR = 0; // no cost since no split
              max_buf_sz = std::max(max_buf_sz_LR, max_buf_sz_child);
              int which_R = 0;
              int8_t niloops_LR = icache[(S|q)][sT][s].niloops[0];
              if (niloops_LR > niloopss[0]) {
                if (Ts[0][0].size() > 0) {
                  if (Ts[0][0][0] != q) {
                    niloopss[1] = niloopss[0];
                    max_buf_szs[1] = max_buf_szs[0];
                    Ts[1] = Ts[0];
                  }
                }
                niloopss[0] = niloops_LR;
                max_buf_szs[0] = max_buf_sz;
                for (int ii = 0; ii < (s-sT+1); ii++) {
                  Ts[0][ii].clear();
                  Ts[0][ii].push_back(q);
                  assert(icache[(S|q)][sT][s].computed == true);
                  Ts[0][ii].insert(Ts[0][ii].end(), icache[(S|q)][sT][s].inds_order[0][ii].begin(), icache[(S|q)][sT][s].inds_order[0][ii].end());
                }
              }
              else if (niloops_LR > niloopss[1]) {
                assert(Ts[0][0].size() > 0);
                if (Ts[0][0][0] != q) {
                  niloopss[1] = niloops_LR;
                  max_buf_szs[1] = max_buf_sz;
                  for (int ii = 0; ii < (s-sT+1); ii++) {
                    Ts[1][ii].clear();
                    Ts[1][ii].push_back(q);
                    assert(icache[(S|q)][sT][s].computed == true);
                    Ts[1][ii].insert(Ts[1][ii].end(), icache[(S|q)][sT][s].inds_order[0][ii].begin(), icache[(S|q)][sT][s].inds_order[0][ii].end());
                  }
                }
              }
              continue;
            }

            uint16_t rem_inds_s = term_inds[s] & ~(S|q);
            if (numones[rem_inds_s] == 0 && s == sT) {
              // only one term in the tree with q as the remaining index
              icache[(S|q)][sT][s].init_element_in_icache((S|q), sT, s, term_inds, numones);
              icache[(S|q)][sT][s].niloops[0] = icache[(S|q)][sT][s].niloops[1] = 0; // number of independent loop indices; q is counted later
              icache[(S|q)][sT][s].max_buf_sz[0] = icache[(S|q)][sT][s].max_buf_sz[1] = 0;
              icache[(S|q)][sT][s].computed = true;
            }
            else {
              io_cost(S | q, sT, s);
              if (icache[(S|q)][sT][s].computed == false) {
                // could not find a loop nest within the specified cost
                continue;
              }
              for (int ii = 0; ii < (s-sT+1); ii++) {
                assert(icache[(S|q)][sT][s].inds_order[0][ii].size() == numones[term_inds[sT+ii] & ~(S|q)]);
              }
            }

            io_cost(S, s+1, eT);
            int which_R = 0;
            if (icache[S][s+1][eT].inds_order[0][0].size() == 0) {
              // nothing to do; the term has already been iterated over at this level
            }
            else if (q == icache[S][s+1][eT].inds_order[0][0][0]) {
              // term in the R branch has the same indices as the term in the L branch
              if (icache[S][s+1][eT].inds_order[1][0].size() == 0) {
                niloopss[0] = niloopss[1] = -1;
                max_buf_szs[0] = max_buf_szs[1] = -1;
                //Ts[0].clear();
                //Ts[1].clear();
                continue;
              }
              assert(icache[S][s+1][eT].inds_order[1][0].size() > 0);
              if (icache[S][s+1][eT].inds_order[1][0].size() == 1) {
                assert(icache[S][s+1][eT].inds_order[1][0][0] == q);
                continue;
              }
              which_R = 1;
            }

            // check for max_buf_sz
            int8_t max_buf_sz_L = icache[(S|q)][sT][s].max_buf_sz[0];
            int8_t max_buf_sz_R = icache[S][s+1][eT].max_buf_sz[which_R];
            int8_t max_buf_sz_LR = max_buf_cost_to_split_interval(S, sT, eT, s);
            max_buf_sz = std::max(max_buf_sz_L,max_buf_sz_R);
            max_buf_sz = std::max(max_buf_sz, max_buf_sz_LR);
            if (max_buf_sz > thres_buf_sz) {
              // could not find a loop nest within the specified cost
              continue;
            }
            if (!sp_buffer && is_sp_buffer(S, sT, s, k)) {
              continue;
            }

            // check for niloops
            int8_t niloops_L = icache[(S|q)][sT][s].niloops[0];
            if (sT == s) {
              // split in the tree, and tree L has only one term, so index q is independent to L
              if (!(q & sp_inds)) {
                // check if there are no sparse indices after q?
                bool sp_inds_after_q = false;
                for (int ii = 0; ii < icache[(S|q)][sT][s].inds_order[0][0].size(); ii++) {
                  if (icache[(S|q)][sT][s].inds_order[0][0][ii] & sp_inds) {
                    // there is a sparse index after q
                    sp_inds_after_q = true;
                    break;
                  }
                }
                if (!sp_inds_after_q) {
                  niloops_L += 1;
                }
              }
            }
            int8_t niloops_R = icache[S][s+1][eT].niloops[which_R];
            IASSERT(niloops_L != -1);
            IASSERT(niloops_R != -1);
            int8_t niloops_LR = niloops_L + niloops_R;

            if (niloops_LR > niloopss[0]) {
              if (Ts[0][0].size() > 0) {
                if (Ts[0][0][0] != q) {
                  niloopss[1] = niloopss[0];
                  max_buf_szs[1] = max_buf_szs[0];
                  Ts[1] = Ts[0];
                }
              }
              niloopss[0] = niloops_LR;
              max_buf_szs[0] = max_buf_sz;
              for (int ii = 0; ii < (s-sT+1); ii++) {
                Ts[0][ii].clear();
                Ts[0][ii].push_back(q);
                assert(icache[(S|q)][sT][s].computed == true);
                Ts[0][ii].insert(Ts[0][ii].end(), icache[(S|q)][sT][s].inds_order[0][ii].begin(), icache[(S|q)][sT][s].inds_order[0][ii].end());
              }
              for (int ii = 0; ii < (eT-s); ii++) {
                Ts[0][s-sT+1+ii] = icache[S][s+1][eT].inds_order[which_R][ii];
              }
            }
            else if (niloops_LR > niloopss[1]) {
              assert(Ts[0].size() > 0);
              assert(Ts[0][0].size() > 0);
              if (Ts[0][0][0] != q) {
                niloopss[1] = niloops_LR;
                max_buf_szs[1] = max_buf_sz;
                for (int ii = 0; ii < (s-sT+1); ii++) {
                  Ts[1][ii].clear();
                  Ts[1][ii].push_back(q);
                  assert(icache[(S|q)][sT][s].computed == true);
                  Ts[1][ii].insert(Ts[1][ii].end(), icache[(S|q)][sT][s].inds_order[0][ii].begin(), icache[(S|q)][sT][s].inds_order[0][ii].end());
                }
                for (int ii = 0; ii < (eT-s); ii++) {
                  Ts[1][s-sT+1+ii] = icache[S][s+1][eT].inds_order[which_R][ii];
                }
              }
            }
          }
          }
          if (niloopss[0] == -1) {
            // could not find a loop nest within the specified cost
            return;
          }
          // update icache
          assert(niloopss[0] != -1);
          assert (icache[S][sT][eT].computed == false);

          for (int j = 0; j < 2; j++) {
            icache[S][sT][eT].niloops[j] = niloopss[j];
            icache[S][sT][eT].max_buf_sz[j] = max_buf_szs[j];
            icache[S][sT][eT].inds_order[j] = std::move(Ts[j]);
          }
          icache[S][sT][eT].computed = true;
        }
      };

  }
#endif
