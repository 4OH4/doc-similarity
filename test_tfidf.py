# -*- coding: utf-8 -*-
"""
Pytest test cases for tfidf.py

@author: 4oh4
28/03/2020

"""

from numpy.testing import assert_almost_equal

from hypothesis import given
from hypothesis.strategies import text

from tfidf import rank_documents


def test_rank_documents1():
    """
    Test similarity match - positive control
    """
    
    # given
    search_terms = 'hello world'
    documents = [search_terms]

    # when
    result = rank_documents(search_terms, documents)

    # then
    assert_almost_equal(result, [1])


@given(search_terms=text())
def test_rank_documents2(search_terms):
    """
    Test similarity match - fuzzing
    """
    
    # given
    documents = ['the quick brown fox jumped over the lazy dog', 'Joe Jackson dances in Paris']

    # when
    result = rank_documents(search_terms, documents)

    # then
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, float)


@given(search_terms=text(), document1=text(), document2=text())
def test_rank_documents3(search_terms, document1, document2):
    """
    Test similarity match - fuzzing
    """
    
    # given
    documents = [document1, document2]

    # when
    result = rank_documents(search_terms, documents)

    # then
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, float)