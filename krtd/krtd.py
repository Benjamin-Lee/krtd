# -*- coding: utf-8 -*-

import itertools
import warnings
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist
from skbio.sequence import DNA

warnings.filterwarnings("ignore", "Mean of empty slice")
warnings.filterwarnings("ignore", "Degrees of freedom")

def distance_between_occurences(seq, k_mer, overlap=True):
    """Takes a DNA sequence and a :math:`k`-mer and calcules the return times for the :math:`k`-mer.

    Args:
        seq (~numpy.ndarray, str, or ~skbio.sequence.DNA): The DNA sequence to analyze.
        k_mer (str): The :math:`k`-mer to calculate return times for.
        overlap (bool, optional): Whether the :math:`k`-mers should overlap.

    Returns:
        ~numpy.ndarray: The return times.

    Note:
        The distance between occurences is defined as the number of nucleotides between the first base of the :math:`k`-mer and first base of its next occurence.

    Examples:
        >>> distance_between_occurences("ATGATA", "A")
        array([2, 1])
        >>> distance_between_occurences("ATGATA", "AT")
        array([2])
        >>> distance_between_occurences("ATGAAATA", "AT")
        array([4])
        >>> distance_between_occurences("ATAAAATAAATA", "ATA")
        array([4, 3])
        >>> distance_between_occurences("ATAAAATAAATA", "ATA", overlap=False)
        array([8])

    """
    if not isinstance(seq, np.ndarray):
        seq = seq_to_array(seq, len(k_mer), overlap=overlap)

    # where the magic happens
    x = np.argwhere(seq == k_mer).flatten()
    x = x - np.insert(x[:-1], [0], [0])
    x = x[1:] - 1

    # not overlaping results in the distances being in the number of k-mers
    # between occurences. For example,
    # distance_between_occurences("ATAAAATAAATA", "ATA", overlap=False) would
    # result in array([2]) since there are two 3-mers between the ATAs if we
    # don't apply this correction
    if not overlap:
        x *= len(k_mer)
        x += len(k_mer) - 1

    return x

def seq_to_array(seq, k=1, overlap=True):
    """Converts a DNA sequence into a Numpy vector.

    Args:
        seq (~skbio.sequence.DNA or str): The sequence to convert.
        k (int, optional): The :math:`k` value to use. Defaults to 1.
        overlap (bool, optional): Whether the :math:`k`-mers should overlap. Defaults to True.

    Returns:
        ~numpy.ndarray: An array representing the sequence.

    Examples:
        >>> seq_to_array("ATGC")
        array(['A', 'T', 'G', 'C'], dtype='<U1')
        >>> seq_to_array("ATGC", k=2)
        array(['AT', 'TG', 'GC'], dtype='<U2')
        >>> seq_to_array("ATGC", k=2, overlap=False)
        array(['AT', 'GC'], dtype='<U2')

    """
    return np.fromiter((str(k_mer) for k_mer in DNA(seq).iter_kmers(k=k, overlap=overlap)), '<U' + str(k))

def krtd(seq, k, overlap=True, reverse_complement=False, return_full_dict=False):
    """Calculates the :math:`k`-mer return time distribution for a sequence.

    Args:
        seq (~skbio.sequence.DNA or str): The sequence to analyze.
        k (int): The :math:`k` value to use.
        overlap (bool, optional): Whether the :math:`k`-mers should overlap. Defaults to True.
        return_full_dict (bool, optional): Whether to return a full dictionary containing every :math:`k`-mer and its RTD. For large values of :math:`k`, as the sparsity of the space in creased, returning a full dictionary may be very slow. If False, returns a :obj:`~collections.defaultdict`. Functionally, this should be identical to a full dictionary if accessing dictionary elements. Defaults to False.

    Returns:
        dict:

    Raises:
        ValueError: When the sequence is degenerate.

    """

    # convert to DNA object
    if not isinstance(seq, DNA):
        seq = DNA(seq)

    if seq.has_degenerates():
        raise ValueError("RTD for sequences with degenerates is undefined.")

    seq = seq_to_array(seq, k=k, overlap=overlap)

    result = {}
    # only calculate RTDs of k-mers present in the seq, which is nice as sparsity increases
    for k_mer in np.unique(seq):

        if reverse_complement:
            revcomp = str(DNA(k_mer).reverse_complement())

            if revcomp in result:
                print(f"Revcomp {k_mer} seen!")
                continue

            k_mer_indices = np.argwhere(seq == k_mer).flatten()
            revcomp_indices = np.argwhere(seq == revcomp).flatten()

            dists = cdist(k_mer_indices.reshape(-1, 1), revcomp_indices.reshape(-1, 1)).flatten() - 1
            result[k_mer] = result[revcomp] = np.where(dists != -1)[0]

        else:
            result[k_mer] = distance_between_occurences(seq, k_mer)

    if return_full_dict:
        for k_mer in itertools.product("ATGC", repeat=k):
            result["".join(k_mer)]
        result = dict(result)

    return result


def codon_rtd(seq):
    """An alias for ``krtd(seq, 3, overlap=False, return_full_dict=True)`` which calculates the return time distribution for codons.

    Args:
        seq (~skbio.sequence.DNA or str): The sequence to analyze.

    Returns:
        dict: A dict whose keys are codons and whose values are dicts of the form ``{mean: 0, std: 0}``.

    Raises:
        ValueError: When the sequence is not able to be divided into codons.
    """
    if len(seq) % 3 != 0:
        raise ValueError("Sequence is not able to be divided into codons.")
    return krtd(seq, 3, overlap=False, return_full_dict=True)
