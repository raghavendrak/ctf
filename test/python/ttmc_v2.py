#!/usr/bin/env python

import unittest
import numpy as np
import ctf
import os
import sys


def allclose(a, b):
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() < 1e-14

class KnowValues(unittest.TestCase):
    def test_ttmc(self):
        B = ctf.tensor((4,4))
        inds, vals = B.read_local()
        for i in range(len(inds)):
            print(str(ctf.comm().rank()) + " " + str(inds[i]))

        n = 4 
        ur = 4 
        vr = 4 
        wr = 4

        T = np.ones((n,n,n))
        U = np.ndarray((n,ur))
        V = np.ones((n,vr))
        W = np.ones((n,wr))
        #W[0][1] = 9

        # CSF tree
        T_i = []
        for i in range(n):
            T_ij = []
            for j in range(n):
                T_ijk = []
                for k in range(n):
                    if 1: #TODO
                        T_ijk.append((k,T[i,j,k]))
                if len(T_ijk) > 0:
                    T_ij.append((j,T_ijk))
            if len(T_ij) > 0:
                T_i.append((i,T_ij))
        T_CSF = T_i

        #UC = np.einsum("ijk,jr,ks->irs",T,V,W)
        UC = np.zeros((n,vr,wr))
        for (i,T_i) in T_CSF:
            Z_i = []
            #einsum("ijk,jr,ks") -> eisum("ijr,jr") 
            for (j,T_ij) in T_i:
                Z_ij = np.zeros(wr)
                for r in range(wr):
                    for (k,t_ijk) in T_ijk:
                        Z_ij[r] += t_ijk * W[k,r]
                Z_i.append((j,Z_ij))
            # can do away without this for loop, but might lose cache hits to matrix W[]?
            for (j,Z_ij) in Z_i:
                for r in range(wr):
                    for s in range(vr):
                        UC[i,s,r] += Z_ij[r] * V[j,s]
        
        UCn = np.zeros((n,vr,wr))
        for (i,T_i) in T_CSF:
            Z_i = []
            #einsum("ijk,jr,ks") -> eisum("ijr,jr") 
            for (j,T_ij) in T_i:
                Z_ij = np.zeros(wr)
                for r in range(wr):
                    for (k,t_ijk) in T_ijk:
                        Z_ij[r] += t_ijk * W[k,r]
                        #print(i, j, k, r)
                        UCn[i][j][r] += t_ijk * W[k,r]
                Z_i.append((j,Z_ij))
                print("appending to Z_i i j:", i, j)
                print(Z_i)
                print(UCn)
            for r in range(wr):
                #for (j,Z_ij) in Z_i:
                Z_j = np.zeros(n) 
                for j in range(n):
                    Z_j[j] = UCn[i][j][r]
                    UCn[i][j][r] = 0
                for j in range(n):
                #print(Z_ij)
                    #temp = 0
                    for s in range(vr):
                        #UC[i,r,s] += Z_ij[r] * V[j,s]
                        #UC[i,s,r] += Z_ij[r] * V[j,s]
                        #UCn[i,s,r] = UCn[i,j,r] * V[j,s]
                        UCn[i,j,r] += Z_j[j] * V[j,s]
                #print(UC)

        print("UC")
        print(UC)
        UCe = np.einsum("ijk,jr,ks->irs",T,V,W)
        print("UCe")
        print(UCe)
        print("UCn")
        print(UCn)


def run_tests():
    np.random.seed(5330);
    wrld = ctf.comm()
    if ctf.comm().rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for dot")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    return result

if __name__ == "__main__":
    result = run_tests()
    ctf.MPI_Stop()
    sys.exit(not result)
