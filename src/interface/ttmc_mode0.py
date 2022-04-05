WC = np.zeros((n,R,R))
# thread over batches
for (k_st, k_end) in k_batches:
    for (i, T_i) in T_CSF:
        for (j, T_ij) in T_i:
            for (k, t_ijk) in T_ij:
                if k >= k_st and k < k_end:
                    for r in range(R): #iR
                        for s in range(R): #jR
                            WC[k,s,r] += t_ijk * U[j,s] * V[i,r]


WC = np.zeros((n, R, R))
# thread over jR
for s in range(R): #jR
    op_ij = np.zeros(R) #iR
    for (i, T_i) in T_CSF:
        for (j, T_ij) in T_i:
            for r in range(R): #iR
                op_ij[r] = U[j,s] * V[i,r]
            for (k, t_ijk) in T_ij:
                for r in range(R):
                    WC[k,s,r] += t_ijk * op_ij[r]