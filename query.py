class query_object:

    def __init__(self, query_dict, query_length, max_freq_term, tokenized_text, location_dict=None):
        """
        :param query_dict: query_dict --> keeps the parsed query
        :param query_length: query_length
        :param max_freq_term: the highest frequency in the query
        :param location_dict: dictionary of term locations in the query.
        """
        self.query_dict = query_dict
        self.query_length = query_length
        self.max_freq_term = max_freq_term
        self.tokenized_text = tokenized_text
        self.location_dict = location_dict
