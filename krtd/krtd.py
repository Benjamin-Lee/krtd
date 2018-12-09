# -*- coding: utf-8 -*-

import itertools
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from skbio.sequence import DNA

# ignore some common warnings resulting from calculating metrics
warnings.filterwarnings("ignore", "Mean of empty slice")
warnings.filterwarnings("ignore", "Degrees of freedom")
warnings.filterwarnings("ignore", "invalid value encountered in double_scalars")

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
    """Converts a DNA sequence into a Numpy vector. If :math:`k>1`, then it creates a vector of the :math:`k`-mers.

    Args:
        seq (~skbio.sequence.DNA or str): The sequence to convert.
        k (int, optional): The :math:`k` value to use. Defaults to 1.
        overlap (bool, optional): Whether the :math:`k`-mers should overlap. Defaults to True.

    Returns:
        ~numpy.ndarray: An array representing the sequence.

    Examples:
        >>> seq_to_array("ATGC")
        array(['A', 'T', 'G', 'C'],
              dtype='<U1')
        >>> seq_to_array("ATGC", k=2)
        array(['AT', 'TG', 'GC'],
              dtype='<U2')
        >>> seq_to_array("ATGC", k=2, overlap=False)
        array(['AT', 'GC'],
              dtype='<U2')

    """
    return np.fromiter((str(k_mer) for k_mer in DNA(seq).iter_kmers(k=k, overlap=overlap)), '<U' + str(k))

def krtd(seq, k, overlap=True, reverse_complement=False, return_full_dict=False, metrics=None):
    """Calculates the :math:`k`-mer return time distribution for a sequence.

    Args:
        seq (~skbio.sequence.DNA or str): The sequence to analyze.
        k (int): The :math:`k` value to use.
        overlap (bool, optional): Whether the :math:`k`-mers should overlap. Defaults to True.
        reverse_complement (bool, optional): Whether to calculate distances between a :math:`k`-mer and its next occurence or the distances between :math:`k`-mers and their reverse complements.
        return_full_dict (bool, optional): Whether to return a full dictionary containing every :math:`k`-mer and its RTD. For large values of :math:`k`, as the sparsity of the space in creased, returning a full dictionary may be very slow. If False, returns a :obj:`~collections.defaultdict`. Functionally, this should be identical to a full dictionary if accessing dictionary elements. Defaults to False.
        metrics (list): A list of functions which, if passed, will be applied to each RTD array.

    Warning:
        Setting ``return_full_dict=True`` will take exponentially more time and as ``k`` increases.

    Returns:
        dict: A dictionary of the shape ``{k_mer: distances}`` in which ``k_mer`` is a str and distances is a :obj:`~numpy.ndarray`. If ``metrics`` is passed, the values of the dictionary will be dictionaries mapping each function to its value (see examples below).

    Raises:
        ValueError: When the sequence is degenerate.

    Examples:
        >>> from pprint import pprint # for prettier printing
        >>> pprint(krtd("ATGCACAGTTCAGA", 1))
        {'A': array([3, 1, 4, 1]),
         'C': array([1, 4]),
         'G': array([4, 4]),
         'T': array([6, 0])}
        >>> pprint(krtd("ATGCACAGTTCAGA", 1, metrics=[np.mean, np.std]))
        {'A': {'mean': 2.25, 'std': 1.299038105676658},
         'C': {'mean': 2.5, 'std': 1.5},
         'G': {'mean': 4.0, 'std': 0.0},
         'T': {'mean': 3.0, 'std': 3.0}}
        >>> pprint(krtd("ATGCACAGTTCAGA", 2, reverse_complement=True))
        {'AA': array([], dtype=int64),
         'AC': array([2]),
         'AG': array([], dtype=int64),
         'AT': array([], dtype=int64),
         'CA': array([1, 3, 8]),
         'CT': array([], dtype=int64),
         'GA': array([2]),
         'GC': array([], dtype=int64),
         'GT': array([2]),
         'TC': array([2]),
         'TG': array([1, 3, 8]),
         'TT': array([], dtype=int64)}
        >>> pprint(krtd("ATGATTGGATATTATGAGGA", 1)) # no value for "C" is printed since it's not in the original sequence
        {'A': array([2, 4, 1, 2, 2, 2]),
         'G': array([3, 0, 7, 1, 0]),
         'T': array([2, 0, 3, 1, 0, 1])}
        >>> pprint(krtd("ATGATTGGATATTATGAGGA", 1, return_full_dict=True)) # now it is
        {'A': array([2, 4, 1, 2, 2, 2]),
         'C': array([], dtype=int64),
         'G': array([3, 0, 7, 1, 0]),
         'T': array([2, 0, 3, 1, 0, 1])}
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

            if revcomp not in result:
                k_mer_indices = np.argwhere(seq == k_mer).flatten()
                revcomp_indices = np.argwhere(seq == revcomp).flatten()

                dists = cdist(k_mer_indices.reshape(-1, 1), revcomp_indices.reshape(-1, 1)).flatten() - 1
                dists = dists[np.where(dists != -1)[0]]

                if not overlap:
                    dists *= len(k_mer)
                    dists += len(k_mer) - 1

                dists = dists.astype("int64")

                if metrics:
                    dists = analyze_rtd(dists, metrics)

                result[k_mer] = result[revcomp] = dists

        else:
            dists = distance_between_occurences(seq, k_mer, overlap=overlap)
            if metrics:
                dists = analyze_rtd(dists, metrics)
            result[k_mer] = dists

    # fill in the result dictionary (expensive!)
    if return_full_dict:
        for k_mer in ("".join(_k_mer) for _k_mer in itertools.product("ATGC", repeat=k)):
            if k_mer not in result:
                dists = np.empty(0, dtype="int64")
                if metrics:
                    dists = analyze_rtd(dists, metrics)
                result[k_mer] = dists

    return result


def codon_rtd(seq, metrics=None):
    """An alias for ``krtd(seq, 3, overlap=False, return_full_dict=True)`` which calculates the return time distribution for codons.

    Args:
        seq (~skbio.sequence.DNA or str): The sequence to analyze.
        metrics (list): See :func:`krtd`.

    Returns:
        dict: See :func:`krtd`.

    Raises:
        ValueError: When the sequence is not able to be divided into codons.
    """
    if len(seq) % 3 != 0:
        raise ValueError("Sequence is not able to be divided into codons.")
    return krtd(seq, 3, overlap=False, return_full_dict=True, metrics=metrics)

def analyze_rtd(rtd, metrics):
    result = {}
    for metric in metrics:
        result[metric.__name__] = metric(rtd)
    return result
