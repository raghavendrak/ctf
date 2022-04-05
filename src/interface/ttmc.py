# all at once kernel
# if k is split; and each GPU has the whole tensor, the partial result can be constructed
# or can fuse local TTMs midway and then replicate the mid-tensor, and split modes?


import numpy as np

n = 4
R = 3
T = np.ndarray((n,n,n))
U = np.ndarray((n,R))
V = np.ndarray((n,R))
W = np.ndarray((n,R))

UC = np.einsum("ijk,jr,ks->irs",T,V,W)
VC = np.einsum("ijk,ir,ks->rjs",T,U,W)
WC = np.einsum("ijk,ir,js->rsk",T,U,V)

T_i = []
for i in range(n):
    T_ij = []
    for j in range(n):
        T_ijk = []
        for k in range(n):
            if (i,j,k) in some_mask():
                T_ijk.append((k,T[i,j,k]))
        if T_ijk.size() > 0:
            T_ij.append((j,T_ijk))
   if T_ij.size() > 0:
       T_i.append((i,T_ij))
T_CSF = T_i

UC = np.zeros((n,R,R))
for (i,T_i) in T_CSF:
    Z_i = []
    for (j,T_ij) in T_i:
        Z_ij = np.zeros(R) #kR
        for r in range(R): #kR
            for (k,t_ijk) in T_ijk:
                Z_ij[r] += t_ijk*W[k,r]
        Z_i.append((j,Z_ij))
    for (j,Z_ij) in Z_i:
        for r in range(R): #kR
            for s in range(R): #jR
                UC[i,r,s] = Z_ij[r]*V[j,s]
                #UC[i,s,r] = Z_ij[r]*V[j,s]

UC[i,r,s] = sum_p2,p3 UC[p2,p3][i,r,s]

VC = np.zeros((n,R,R))
Z = []
for (j_st,j_end) in j_batches:
    for (i,T_i) in T_CSF:
        Z_i = []
        for (j,T_ij) in T_i:
            if j>=j_st and j<j_end:
                Z_ij = np.zeros(R)
                for r in range(R):
                    for (k,t_ijk) in T_ijk:
                        Z_ij[r] += t_ijk*W[k,r]
                Z_i.append((j,Z_ij))
        Z.append(Z_i)
    for (i,Z_i) in Z:
        for (j,Z_ij) in Z_i:
            for r in range(R):
                for s in range(R):
                    VC[j,r,s] = Z_ij[r]*U[i,s]
                    #VC[s,j,r] = Z_ij[r]*U[i,s]


WC = np.zeros((n,R,R))
Z = []
for (i,T_i) in T_CSF:
    Z_i = []
    for (j,T_ij) in T_i:
        for (k,t_ijk) in T_ijk:
            dZ_ik = t_ijk*V[j,r]
            Z_i.accumulate(k,dZ_ik)
for (i,Z_i) in Z:
    for (k,Z_ik) in Z_i:
        for r in range(R):
            for s in range(R):
                WC[k,r,s] = Z_ik[r]*U[i,s]


WC = np.zeros((n,R,R))
# thread over batches
for (k_st, k_end) in k_batches:
    for (i, T_i) in T_CSF:
        for (j, T_ij) in T_i:
            for (k, t_ijk) in T_ijk:
                if k >= k_st and k < k_end:
                    for r in range(R): #iR
                        for s in range(R): #jR
                            WC[k,s,r] += t_ijk * U[j,s] * V[i,r]

def compute_outer_U(t,v,w)
    return t*np.outer(v,w)

for i in range(n):
    for j in range(n):
        for k in range(n):
            U[i,:,:] += compute_outer_U(T[i,j,k],V[j,:],W[k,:])
