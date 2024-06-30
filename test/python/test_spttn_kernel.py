#!/usr/bin/env python

import unittest
import numpy as np
import ctf
import os
import sys
import numpy.linalg as la

def allclose(a, b):
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() <= 1e-5

class KnowValues(unittest.TestCase):
    def test_ttmc(self):
        """
        Test spttn kernel for order 3 TTMc
        and all-mode order 3 TTMc
        """
        einsum_expr = "ijk,ri,sj->rsk"
        lens = [10,10,10]
        R = 4
        A = ctf.tensor(lens,sp=True)
        A.fill_sp_random(-1.,1.,.5)
        tsrs = []
        for i in range(2):
            fac_lens = [lens[i],R]
            tsrs.append(ctf.random.random(fac_lens))
        op_lens = [lens[2],R,R]
        tsrs.append(ctf.zeros(op_lens))
        ctf.spttn_kernel(A, tsrs, 3, einsum_expr.encode())
        ctr = A.i("ijk")*tsrs[0].i("kr")*tsrs[1].i("js")
        ans = ctf.zeros(op_lens)
        ans.i("isr") << ctr
        self.assertTrue(allclose(ans, tsrs[2]))

        einsum_expr = "ijk,ri,sj,tk->rst"
        A = ctf.tensor(lens,sp=True)
        A.fill_sp_random(-1.,1.,.5)
        tsrs = []
        for i in range(3):
            fac_lens = [lens[i],R]
            tsrs.append(ctf.random.random(fac_lens))
        op_lens = [R,R,R]
        tsrs.append(ctf.zeros(op_lens))
        ctf.spttn_kernel(A, tsrs, 4, einsum_expr.encode())
        ctr = A.i("ijk")*tsrs[0].i("kr")*tsrs[1].i("js")*tsrs[2].i("it")
        ans = ctf.zeros(op_lens)
        ans.i("tsr") << ctr
        self.assertTrue(allclose(ans, tsrs[3])) 

def run_tests():
    np.random.seed(5330);
    wrld = ctf.comm()
    if wrld.rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Test for spttn kernel")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    return result

if __name__ == "__main__":
    result = run_tests()
    ctf.MPI_Stop()
    sys.exit(not result)