#mttkrp
("ijkl,ja,ka,la->ia")
#ttmc
("ijk,jr,ks->irs")
#tttp
("ijk,ia,ja,ka->ijk")

#ttmc
WC = np.zeros((n, R, R))
for (i, T_i) in T_CSF:
  for (j, T_ij) in T_i:
    for (k, t_ijk) in T_ij:
      for r in range(R):
        for s in range(R):
          WC[k,s,r] += t_ijk * U[j,s] * V[i,r]


#ttmc
UC = np.zeros((n, R, R))
for (i, T_i) in T_CSF:
  for (j, T_ij) in T_i:
    for (k, t_ijk) in T_ij:
      for r in range(R):
        for s in range(R):
          WC[i,s,r] += t_ijk * V[j,s] * V[k,r]
# 3mR^2

#ttmc
UC = np.zeros((n, R, R))
for (i, T_i) in T_CSF:
    Z_i = []
    for (j, T_ij) in T_i:
        Z_ij = np.zeros(R)
        for r in range(R):
            for (k, t_ijk) in T_ijk:
                Z_ij[r] += t_ijk * W[k, r]
        Z_i.append((j, Z_ij))
    for (j, Z_ij) in Z_i:
        for r in range(R):
            for s in range(R):
                UC[i, s, r] += Z_ij[r] * V[j, s]
# 2mR + 2kR^2 (k = 1 to m)


#mttkrp
WC = np.zeros((n, R))
for (i, T_i) in T_CSF:
  for (j, T_ij) in T_i:
    for (k, t_ijk) in T_ij:
      for a in range(R):
        WC[i,a] += t_ijk * U[j,a] * V[k,a] * W[l,a]


("ijk,jk->ijk")
# partition jk tensor along the two modes of ijk tensor?
# Hadamard; can be ("ijkl,jk,kl,il->ijkl")
WC = np.zeros((n, Rj, Rk))
for (i, T_i) in T_CSF:
  for (j, T_ij) in T_i:
    for (k, t_ijk) in T_ij:
      WC[i, j, k] += t_ijk * U[j, k]