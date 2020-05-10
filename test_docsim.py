# -*- coding: utf-8 -*-
"""
Semantic document similarity using Gensim

@author: 4oh4
28/03/2020

Pytest test cases for docsim.py

These are quite long-running tests, due to the load time on the GloVe model

"""

import pytest

from numpy.testing import assert_almost_equal

# TODO: generative testing
# from hypothesis import given
# from hypothesis.strategies import text

from docsim import DocSim


@pytest.fixture
def fixture_DocSim(mocker):
    # Test fixture
    docsim = DocSim()

    return docsim


def test_docsim1(fixture_DocSim):
    """
    Test similarity match - positive control
    """
    
    # given
    search_terms = 'tomato'
    documents = [search_terms, 'aligator']

    # when
    results = fixture_DocSim.similarity_query(search_terms, documents)

    # then
    assert isinstance(results, list)
    for item in results:
        assert isinstance(item, float)
    assert_almost_equal(results[0], 1.)


def test_docsim2(fixture_DocSim):
    """
    Test exact match to corpus - expect an exception here as the maths breaks down
    """
    
    # given
    search_terms = 'tomato'
    documents = [search_terms]

    # when/then
    with pytest.raises(ValueError):
        fixture_DocSim.similarity_query(search_terms, documents)
