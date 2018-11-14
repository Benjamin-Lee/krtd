# -*- coding: utf-8 -*-

import numpy as np
import itertools
import warnings
from collections import defaultdict

from skbio.sequence import DNA

warnings.filterwarnings("ignore", "Mean of empty slice")
warnings.filterwarnings("ignore", "Degrees of freedom")

def krtd(seq, k, overlap=True, return_full_dict=False):
    '''Calculates the :math:`k`-mer return time distribution for a sequence.

    Args:
        seq (~skbio.sequence.DNA or str): The sequence to analyze.
        k (int): The :math:`k` value to use.
        overlap (bool, optional): Whether the :math:`k`-mers should overlap.
        return_full_dict (bool, optional): Whether to return a full dictionary containing every :math:`k`-mer and its RTD. For large values of :math:`k`, as the sparsity of the space in creased, returning a full dictionary may be very slow. If false, returns a :obj:`~collections.defaultdict`. Functionally, this should be identical to a full dictionary if accessing dictionary elements. Defaults to false.

    Returns:
        ~collections.defaultdict or dict:

    Raises:
        ValueError: When the sequence is degenerate.

    '''

    if not isinstance(seq, DNA):
        seq = DNA(seq)

    if seq.has_degenerates():
        raise ValueError("RTD for sequences with degenerates is undefined.")

    return_times = defaultdict(lambda: dict(mean=0, std=0))
    seq = np.fromiter((str(k_mer) for k_mer in seq.iter_kmers(k=k, overlap=overlap)), f'<U{k}')
    uniques = np.unique(seq)
    for k_mer in uniques:
        x = np.argwhere(seq == k_mer).flatten()
        x = x - np.insert(x[:-1], [0], [0])
        x = x[1:] - 1

        if x.size:
            return_times[k_mer] = dict(mean=np.mean(x),
                                       std=np.std(x))

    if return_full_dict:
        for k_mer in itertools.product("ATGC", repeat=k):
            return_times["".join(k_mer)]
        return_times = dict(return_times)

    return return_times


def codon_rtd(seq):
    '''An alias for ``krtd(seq, 3, overlap=False, return_full_dict=True)`` which calculates the return time distribution for codons.

    Args:
        seq (~skbio.sequence.DNA or str): The sequence to analyze

    Returns:
        dict: A dict whose keys are codons and whose values are dicts of the form ``{mean: 0, std: 0}``.

    Raises:
        ValueError: When the sequence is not able to be divided into codons.
    '''
    if len(seq) % 3 != 0:
        raise ValueError("Sequence is not able to be divided into codons.")
    return krtd(seq, 3, overlap=False, return_full_dict=True)
