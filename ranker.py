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
        calculates cosine similarity over doc-query pair ranked by tf-idf.
        then, sorts by highest doc score and returns k most relevant docs.
        :param relevant_docs:
        :param normalized_query:
        :param inverted_documents_dict:
        :param k:
        :return:
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

