import numpy as np
import pytest
from hypothesis import given, assume, strategies as st
from skbio import DNA, Sequence
from krtd import codon_rtd


@given(st.text(alphabet=["A", "T", "G", "C"]))
def test_invalid_seq(x):
    assume(len(x) % 3 != 0)
    with pytest.raises(ValueError):
        codon_rtd(x)
