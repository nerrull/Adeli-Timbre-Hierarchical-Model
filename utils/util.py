# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)

--

% Perfect reconstruction sliCQ

% right now, even slice length (sl_len) is required. Parameters are the
% same as NSGTF plus slice length, minimal required window length,
% Q-factor variation, and test run parameters.
"""

import numpy as np
from itertools import cycle, chain, tee

def _isseq(x):
    try:
        len(x)
    except TypeError:
        return False
    return True

def chkM(M, g):
    if M is None:
        M = np.array(list(map(len, g)))
    elif not _isseq(M):
        M = np.ones(len(g), dtype=int) * M
    return M

# one of the more expensive functions (32/400)
def arrange(cseq, M, fwd):
    cseq = iter(cseq)
    try:
        c0 = next(cseq)  # grab first stream element
    except StopIteration:
        return iter(())
    cseq = chain((c0,), cseq)  # push it back in
    M = list(map(len, c0[0]))  # read off M from the coefficients
    ixs = (
           [(slice(3*mkk//4, mkk), slice(0, 3*mkk//4)) for mkk in M],  # odd
           [(slice(mkk//4, mkk), slice(0, mkk//4)) for mkk in M]  # even
    )
    if fwd:
        ixs = cycle(ixs)
    else:
        ixs = cycle(ixs[::-1])

    return ([
                [np.concatenate((ckk[ix0],ckk[ix1]))
                   for ckk,(ix0,ix1) in zip(ci, ixi)
                ]
             for ci in cci
             ]
             for cci,ixi in zip(cseq, ixs)
            )

def starzip(iterables):
    def inner(itr, i):
        for t in itr:
            yield t[i]
    iterables = iter(iterables)
    it = next(iterables)  # we need that to determine the length of one element
    iterables = chain((it,), iterables)
    return [inner(itr, i) for i,itr in enumerate(tee(iterables, len(it)))]

def chnmap(gen, seq):
    chns = starzip(seq) # returns a list of generators (one for each channel)
    gens = list(map(gen, chns)) # generators including transformation
    return zip(*gens)  # packing channels to one generator yielding channel tuples
