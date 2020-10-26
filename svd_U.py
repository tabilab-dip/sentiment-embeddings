#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Cem Rıfkı Aydın
"""
from scipy.sparse.linalg import svds

import PMI_matrix
import constants
import lexical_interface


class CorpusSvdU:
    """
    This class performs SVD method on the PMI matrix generated by the corpus statistics.
    Only the "U" matrix out of U*S*V is employed.
    """
    def __init__(self, revs, labels):
        self.revs = revs
        self.labels = labels

    def get_corpus_svd_u_vecs(self):
        """
        This helper method returns the embeddings generated by performing the SVD - U technique
            on the corpus.

        :return: Words and their corresponding embeddings.
        :rtype: set, np.array
        """

        pmi_dict_keys, pmi_matr = PMI_matrix.prepare_pmi(self.revs)

        return get_svd_u(pmi_dict_keys, pmi_matr)

    def get_vecs(self):
        """
        The same as the above function:
        It is a helper method that returns the embeddings generated by using the SVD - U technique.

        :return: Words and their corresponding embeddings.
        :rtype: set, np.array
        """

        return self.get_corpus_svd_u_vecs()


class LexicalSvdU:
    """
    This class performs SVD method on the dictionary matrix.
    Only the "U" matrix out of U*S*V is employed.
    """
    def __init__(self, revs, labels):
        self.revs = revs
        self.labels = labels

    def get_lexical_svd_u_vecs(self):
        """
        This helper method returns the vectors generated by performing the SVD - U technique
            on the embeddings matrix. This matrix is built by lexicon (e.g. Turkish dictionary).

        :return: Words and their corresponding embeddings.
        :rtype: set, np.array
        """

        dict_ = lexical_interface.get_lexical_dict(self.revs, self.labels)

        keys, matr = self.generate_keys_and_np_matr_from_dict(dict_)

        return get_svd_u(keys, matr)

    def get_vecs(self):
        """
        This is an interface method that is the same as the get_lexical_svd_U_vecs function:

        """

        return self.get_lexical_svd_u_vecs()

    def generate_keys_and_np_matr_from_dict(self, dict_):
        """
        This method generates keys and values from a dict variable.
        This is a basic helper function in that the conversion of types are required for some cases
            in creating the embedding model.

        :param dict_: A dictionary variable.
        :type dict_: dict
        :return: Dictionary decomposed into its keys and values.
        :rtype: list, list
        """

        return list(dict_.keys()), list(dict_.values())


def get_svd_u(words, matr):
    """
    This function obtains the embeddings of corpus words in question using the SVD - U approach.
    This also helps the conversion of types for basic operations to provide compatibility
        in the later stages.

    :param words: Words.
    :type words: set or list
    :param matr: The embedding matrix.
    :type matr: numpy.array
    :return: Words and their embeddings.
    :rtype: dict
    """

    dim_size = constants.EMBEDDING_SIZE
    if dim_size > len(matr[0]):
        raise Exception("The size of embeddings should have been lower than the number of words in the corpus."
                        "Please decrease the value of EMBEDDING_SIZE or use a bigger corpus.")

    u, _, _ = svds(matr, k=dim_size)

    res_dict = dict(zip(words, u))

    return res_dict


if __name__ == "__main__":
    pass
