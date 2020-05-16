import numpy as np
import pytest
from hypothesis import given, strategies as st
from skbio import DNA, Sequence

from krtd import krtd


def test_1_mer_rtd_str_no_gap():
    x = krtd("AA", 1)
    assert np.array_equal(x["A"], np.array([0]))

    # by default, non represented k-mers are empty arrays
    assert "T" not in x
    assert "G" not in x
    assert "C" not in x


def test_1_mer_rtd_str_with_gap():
    x = krtd("AATTA", 1)
    assert np.array_equal(x["A"], np.array([0, 2]))
    assert np.array_equal(x["T"], np.array([0]))

    assert "G" not in x
    assert "C" not in x


def test_2_mer_rtd_str_with_gap():
    x = krtd("AATTAAT", 2)
    assert np.array_equal(x["AA"], np.array([3]))
    assert np.array_equal(x["AT"], np.array([3]))


@given(st.text(alphabet=["A", "T", "G", "C"]))
def test_verify_length(seq):
    for letter in ["A", "T", "G", "C"]:
        if letter in seq:
            assert (
                len(krtd(seq, 1)[letter]) == seq.count(letter) - 1
            )  # there are count - 1 k-mer distances


@given(
    st.text(alphabet=["A", "T", "G", "C"], min_size=3),
    st.integers(min_value=1, max_value=3),
)
def test_Sequence_DNA_and_str_equality(seq, k):
    _str = krtd(seq, k)
    _Sequence = krtd(Sequence(seq), k)
    _DNA = krtd(DNA(seq), k)

    for k_mer in _str.keys():
        assert np.array_equal(_str[k_mer], _Sequence[k_mer])
        assert np.array_equal(_str[k_mer], _DNA[k_mer])


@given(
    st.text(alphabet=["R", "Y", "S", "W", "K", "M", "B", "D", "H", "V"], min_size=3),
    st.integers(min_value=1, max_value=3),
)
def test_degenerate(seq, k):
    with pytest.raises(ValueError):
        krtd(seq, k)


def test_3_mer_revcomp_rtd():
    x = krtd("ATGCCATGCAT", 3, reverse_complement=True)
    assert np.array_equal(x["ATG"], np.array([3, 2]))
    assert np.array_equal(x["CAT"], np.array([0]))


def test_3_mer_revcomp_rtd_no_overlap():
    x = krtd("ATGCCATGCCATATGCATTAG", 3, reverse_complement=True, overlap=False)
    assert np.array_equal(x["ATG"], np.array([8, 2]))
    assert np.array_equal(x["CAT"], np.array([2]))


def test_3_mer_rtd_no_overlap():
    x = krtd("ATGCCATGCCATATGCATTAG", 3, overlap=False)
    assert np.array_equal(x["ATG"], np.array([11]))
    assert np.array_equal(x["CAT"], np.array([5]))


def test_full_dict_1_mer():
    x = krtd("AT", 1, return_full_dict=True)
    assert len(x) == 4
    assert np.array_equal(x["G"], np.array([]))
