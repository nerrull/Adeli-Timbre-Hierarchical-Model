# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria
Thomas Grill, 2011-2015
http://grrrr.org/nsgt
Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)

Modified by Etienne Richan, 2017
NECOTIS Group
University of Sherbrooke
"""


import numpy as np
cimport numpy as np

def nsgtf_loop(loopparams, np.ndarray ft not None,np.ndarray temp0 not None, k_t_r):
    cdef list c =[]
    N = len(loopparams)
    Pos_Chans_No = (N ) // 2

    #First and last channels don't have mirror filters
    c.append(nsgtf(ft,loopparams[0], temp0, k_t_r))
    cdef int i
    cdef list positiveParams, negativeParams
    cdef np.ndarray ft, temp0
    for i in range(1, Pos_Chans_No ):
        positiveParams = loopparams[i]
        negativeParams = loopparams[N-i]
        c.append( nsgtf(ft, positiveParams, temp0, k_t_r) +  nsgtf(ft, negativeParams,temp0, k_t_r))

    #First and last channels don't have mirrored filters
    c.append(nsgtf(ft,loopparams[Pos_Chans_No], temp0, k_t_r))
    return c

def nsgtf(np.ndarray ft not None, list params, np.ndarray temp0 not None, keep_temporal_resolution =True):
    # type: (object, object, object) -> object
    temp = temp0.copy()
    (mii,_, gi1, gi2, win_range, Lg, col) = params

    # modified version to avoid superfluous memory allocation
    t = temp[win_range]
    t1 = t[:(Lg + 1) // 2]
    t1[:] = gi2  # if mii is odd, this is of length mii-mii//2
    t2 = t[-(Lg // 2):]
    t2[:] = gi1  # if mii is odd, this is of length mii//2

    ftw = ft[win_range]
    t1 *= ftw[:Lg // 2]
    t2 *= ftw[Lg // 2:]

    if not keep_temporal_resolution:
        return t

    temp[win_range] = t

    if col >1:
        temp = np.sum(temp.reshape((mii, -1)), axis=1)
    else:
        temp = temp.copy()
    return temp
