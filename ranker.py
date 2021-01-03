# you can change whatever you want in this module, just make sure it doesn't 
# break the searcher module
from numpy import dot
from numpy.linalg import norm
class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def rank_relevant_docs(relevant_docs, normalized_query, inverted_documents_dict, k=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param k: number of most relevant docs to return, default to everything.
        :param relevant_docs: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """
        ranked_docs_dict = {}

        for doc in relevant_docs:
            cos_sim = dot(relevant_docs[doc], normalized_query) / (norm(relevant_docs[doc]) * norm(normalized_query))
            ranked_docs_dict[doc] = cos_sim

        sorted_ranked_docs_dict = {k: v for k, v in
                                   sorted(ranked_docs_dict.items(), key=lambda item: (item[1],
                                                                                      1 / inverted_documents_dict[
                                                                                      str(item[0])][1]), reverse=True)}
        docs_to_retrieve = []
        for idx, (key, value) in enumerate(sorted_ranked_docs_dict.items()):
            if k is not None and idx == k:
                break

            docs_to_retrieve.append(key)

        return docs_to_retrieve

