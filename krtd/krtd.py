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

def distance_between_occurences(seq, k_mer1, k_mer2, overlap=True):
    """Takes a DNA sequence and a :math:`k`-mer and calcules the return times
    for the :math:`k`-mer.

    Args:
        seq (~numpy.ndarray, str, or ~skbio.sequence.DNA): The DNA sequence to analyze.
        k_mer1 (str or ~skbio.sequence.DNA): The :math:`k`-mer to calculate return times from.
        k_mer2 (str or ~skbio.sequence.DNA): The :math:`k`-mer to calculate return times to.
        overlap (bool, optional): Whether the :math:`k`-mers should overlap.

    Returns:
        ~numpy.ndarray: The return times.

    Note:
        The distance between occurences is defined as the number of nucleotides
        between the first base of the ``k_mer1`` and first base of ``k_mer2``.

    Examples:
        .. runblock:: pycon

            >>> from krtd import distance_between_occurences # ignore
            >>> distance_between_occurences("ATGATA", "A", "A")
            >>> distance_between_occurences("ATGATA", "AT", "AT")
            >>> distance_between_occurences("ATGAAATA", "AT", "AT")
            >>> distance_between_occurences("ATAAAATAAATA", "ATA", "ATA")
            >>> distance_between_occurences("ATAAAATAAATA", "ATA", "ATA", overlap=False)

    """
    if len(k_mer1) != len(k_mer2):
        raise ValueError("k-mers must be of same length")

    if not isinstance(seq, np.ndarray):
        seq = seq_to_array(seq, len(k_mer1), overlap=overlap)

    # convert to strings
    k_mer1, k_mer2 = str(k_mer1), str(k_mer2)
    print(k_mer1, k_mer2)

    # where the magic happens
    if k_mer1 == k_mer2:
        x = np.argwhere(seq == k_mer1).flatten()
        x = x - np.insert(x[:-1], [0], [0])
        x = x[1:] - 1

    else:
        a = np.argwhere(seq == k_mer1).flatten()
        b = np.argwhere(seq == k_mer2).flatten()
        try:
            c = a - b[np.argmax(a[:,None]<b,axis=1)]
            x = c[c < 0]*-1 -1
        except ValueError:
            x = np.array([])

    # not overlaping results in the distances being in the number of k-mers
    # between occurences. For example,
    # distance_between_occurences("ATAAAATAAATA", "ATA", "ATA", overlap=False) would
    # result in array([2]) since there are two 3-mers between the ATAs if we
    # don't apply this correction
    if not overlap:
        x *= len(k_mer1)
        x += len(k_mer1) - 1

    return x

def seq_to_array(seq, k=1, overlap=True):
    """Converts a DNA sequence into a Numpy vector. If :math:`k>1`, then it
    creates a vector of the :math:`k`-mers.

    Args:
        seq (~skbio.sequence.DNA or str): The sequence to convert.
        k (int, optional): The :math:`k` value to use. Defaults to 1.
        overlap (bool, optional): Whether the :math:`k`-mers should overlap.
            Defaults to True.

    Returns:
        ~numpy.ndarray: An array representing the sequence.

    Examples:
        .. runblock:: pycon

            >>> from krtd import seq_to_array # ignore
            >>> seq_to_array("ATGC")
            >>> seq_to_array("ATGC", k=2)
            >>> seq_to_array("ATGC", k=2, overlap=False)

    """
    # convert to DNA object
    if not isinstance(seq, DNA):
        seq = DNA(seq)
    return np.fromiter((str(k_mer) for k_mer in seq.iter_kmers(k=k, overlap=overlap)), '<U' + str(k))

def krtd(seq, k, overlap=True, reverse_complement=False, return_full_dict=False, metrics=None):
    """Calculates the :math:`k`-mer return time distribution for a sequence.

    Args:
        seq (~skbio.sequence.DNA or str): The sequence to analyze.
        k (int): The :math:`k` value to use.
        overlap (bool, optional): Whether the :math:`k`-mers should overlap.
            Defaults to True.
        reverse_complement (bool, optional): Whether to calculate distances
            between a :math:`k`-mer and its next occurence or the distances between
            :math:`k`-mers and their reverse complements.
        return_full_dict (bool, optional): Whether to return a full dictionary
            containing every :math:`k`-mer and its RTD. For large values of
            :math:`k`, as the sparsity of the space in creased, returning a full
            dictionary may be very slow. If False, returns a
            :obj:`~collections.defaultdict`. Functionally, this should be identical
            to a full dictionary if accessing dictionary elements. Defaults to
            False.
        metrics (list): A list of functions which, if passed, will be applied to
            each RTD array.

    Warning:
        Setting ``return_full_dict=True`` will take exponentially more time and as ``k`` increases.

    Returns:
        dict: A dictionary of the shape ``{k_mer: distances}`` in which ``k_mer`` is a str and distances is a :obj:`~numpy.ndarray`. If ``metrics`` is passed, the values of the dictionary will be dictionaries mapping each function to its value (see examples below).

    Raises:
        ValueError: When the sequence is degenerate.

    Examples:
        .. runblock:: pycon

            >>> from krtd import krtd # ignore
            >>> from pprint import pprint as print # for prettier printing # ignore
            >>> import numpy as np # ignore
            >>> print(krtd("ATGCACAGTTCAGA", 1))
            >>> print(krtd("ATGCACAGTTCAGA", 1, metrics=[np.mean, np.std]))
            >>> print(krtd("ATGCACAGTTCAGA", 2, reverse_complement=True))
            >>> print(krtd("ATGATTGGATATTATGAGGA", 1)) # no value for "C" is printed since it's not in the original sequence
            >>> print(krtd("ATGATTGGATATTATGAGGA", 1, return_full_dict=True)) # now it is
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
        dists = distance_between_occurences(seq, k_mer, k_mer if not reverse_complement else DNA(k_mer).reverse_complement(), overlap=overlap)
        if metrics:
            dists = _analyze_rtd(dists, metrics)
        result[k_mer] = dists

    # fill in the result dictionary (expensive!)
    if return_full_dict:
        for k_mer in ("".join(_k_mer) for _k_mer in itertools.product("ATGC", repeat=k)):
            if k_mer not in result:
                dists = np.empty(0, dtype="int64")
                if metrics:
                    dists = _analyze_rtd(dists, metrics)
                result[k_mer] = dists

    return result


def codon_rtd(seq, metrics=None):
    """An alias for ``krtd(seq, 3, overlap=False, return_full_dict=True)`` which
    calculates the return time distribution for codons.

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

def _analyze_rtd(rtd, metrics):
    """A convenience function for building a dict of metrics and their values
    for an RTD array.

    Args:
        rtd (~numpy.ndarray): The RTD array to analyze.
        metrics (list): A list of functions to call on the RTD array.

    Returns:
        dict: The string name of each metric and its value.
    """
    result = {}
    for metric in metrics:
        result[metric.__name__] = metric(rtd)
    return result

def _k_mer_to_index(k_mer):
    """Converts a k-mer to its numerical index.

    Args:
        k_mer (str): The :math:`k`-mer to convert.

    Returns:
        int: The index of the :math:`k`-mer.

    Examples:
        >>> _k_mer_to_index("A")
        0
        >>> _k_mer_to_index("T")
        3
        >>> _k_mer_to_index("TT")
        15
        >>> _k_mer_to_index("TTT")
        63

    """
    result = 0
    for base in k_mer:
        result = result * 4 + ["A", "C", "G", "T"].index(base)
    return result

def rtd_metric_dict_to_array(rtd_metric_dict):
    """A convenience function for deterministically turning RTD metric dicts
    (such as the output of :func:`krtd`) into arrays, which is useful for
    computing distances, *etc.*

    The output array is a vector with :math:`4^{k}n` elements where :math:`n` is
    the number of metrics that were analyzed. To understand the order of the
    array, first consider an RTD metric dictionary with only one metric. The
    zero-based index would correspond directly to the alphabetical index of the
    :math:`k`-mer. If :math:`k=1`, the metric for A would be in position 0, C in
    1, G, in 2, T in 3. If there is more than one metric, the metrics' values
    for the :math:`k`-mer are listed in alphabetical order before proceeding to
    the next :math:`k`-mer. See the example for a clarification.

    Args:
        rtd_metric_dict (dict): A dictionary mapping :math:`k`-mers to
            dictionaries of metrics and their float values.

    Example:
        .. runblock:: pycon

            >>> from krtd import krtd, rtd_metric_dict_to_array # ignore
            >>> from pprint import pprint as print # for prettier printing # ignore
            >>> import numpy as np # ignore
            >>> import warnings # ignores the warning caused by dividing by zero # ignore
            >>> warnings.simplefilter("ignore") # ignore
            >>> d = krtd("ATGCATGCCGTA", 1, metrics=[np.mean, np.std])
            >>> print(d)
            >>> rtd_metric_dict_to_array(d)
            >>> d = krtd("ATGCATGCCGTA", 5, metrics=[np.mean, np.std])
            >>> rtd_metric_dict_to_array(d).shape # should be (4**5)*2 or 2048
    """
    metric_names = sorted(list(rtd_metric_dict.values())[0]) # stringify the metric names
    space = np.zeros((4**len(list(rtd_metric_dict.keys())[0])) * len(metric_names)) # create an empty array to represent the rtd data for a seq
    for item in rtd_metric_dict.items(): # the iteration order doesn't matter
        for metric in metric_names:
            space[_k_mer_to_index(item[0]) * len(metric_names) + metric_names.index(metric)] = item[1][metric]
    return np.nan_to_num(space) # fill the nans with 0
