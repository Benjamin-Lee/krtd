import pytest
import krtd
from skbio import Sequence
from skbio import DNA
from hypothesis import given
import hypothesis.strategies as st

def test_1_mer_rtd_str_no_gap():
    # the mean is 0 and the std is 0
    x = krtd.krtd("AA", 1)
    assert x["A"]["mean"] == 0
    assert x["A"]["std"] == 0

    # by default, non represented k-mers are given values of zero
    assert x["T"]["mean"] == 0
    assert x["T"]["std"] == 0

def test_1_mer_rtd_seq_no_gap():
    # the mean is 0 and the std is 0
    x = krtd.krtd(Sequence("AA"), 1)
    assert x["A"]["mean"] == 0
    assert x["A"]["std"] == 0

    # by default, non represented k-mers are given values of zero
    assert x["T"]["mean"] == 0
    assert x["T"]["std"] == 0

def test_1_mer_rtd_str_with_gap():
    x = krtd.krtd("ATTATA", 1)
    assert x["A"]["mean"] == 1.5 == (2 + 1) / 2
    assert x["A"]["std"] == (((2 - 1.5)**2 + (1 - 1.5)**2) / 2)**0.5 == 0.5

    assert x["T"]["mean"] == 0.5
    assert x["T"]["std"] == 0.5

    assert x["G"]["mean"] == 0
    assert x["G"]["std"] == 0

    assert x["C"]["mean"] == 0
    assert x["C"]["std"] == 0

def test_2_mer_rtd_str_no_gap():
    x = krtd.krtd("AAATGCAA", 2)
    assert x["AA"]["mean"] == x["AA"]["std"] == 2
    assert x["AT"]["mean"] == x["AT"]["std"] == 0

@given(st.text(alphabet=["A", "T", "G", "C"], min_size=3), st.integers(min_value=1, max_value=3))
def test_Sequence_DNA_and_str_equality(seq, k):
    assert krtd.krtd(seq, k) == krtd.krtd(Sequence(seq), k) == krtd.krtd(DNA(seq), k)

@given(st.text(alphabet=["R", "Y", "S", "W", "K", "M", "B", "D", "H", "V"], min_size=3),
       st.integers(min_value=1, max_value=3))
def test_degenerate(seq, k):
    with pytest.raises(ValueError):
        krtd.krtd(seq, k)
