import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from document import Document
import json
from nltk.stem.snowball import SnowballStemmer

from query import query_object


class Parse:
    CAPITAL_LETTER_DICT = {}
    idx = 0
    ENTITY_DICT = {}
    Parsing_a_word = False
    AMOUNT_OF_NUMBERS_IN_CORPUS = 0

    def __init__(self):
        self.problem_terms_to_check = []
        self.max_freq_term = 0
        self.term_dict = {}
        self.stop_words = stopwords.words('english')
        self.our_stop_words = ["RT", "http", "https", r"", r"'", r"''", r'"', '``', '’', r'', r"", '...', '…', '', r'"',
                               "twitter.com", "web", "status", "i", r'i', "n't", "--", "'re", "..", "'it", "'m"
                               , "......", ".....", "//", "'ve", "N'T", "'ll", "S", "s", r' ', r'', r"", r"''"
                               ,r"’", r"‘", r"``", r"'", r"`", '"', r'""', r'"', r"“", r"”",
                               'rt', r'!', r'?', r',', r':', r';', r'(', r')', r'...', r'[', ']', r'{', '}' "'&'", '$',
                               '.', r'\'s', '\'s', '\'d', r'\'d', r'n\'t','1️⃣.1️⃣2️⃣', '~~', '...']

        self.additional = {"twitter.com", "web", "status", "i", r'i'}
        self.stop_words.extend(self.our_stop_words)
        self.stop_words_dict = dict.fromkeys(self.stop_words)
        self.tokens = None
        self.is_num_after_num = False
        self.dict_punctuation = dict.fromkeys(string.punctuation)
        self.location_dict = {}
        self.snow_stemmer = SnowballStemmer(language='english')

        self.number_pattern = re.compile("[-+]?[\d]+(?:\.\d+)?/[-+]?[\d]+(?:\.\d+)?\w?[k|K|m|M|b|B]?"
                                         "|[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?[k|K|m|M|b|B]?")
        self.date_pattern = re.compile("\d{1,4}[-\.\/]\d{1,4}([-\.\/]\d{1,4})?")
        self.hashtag_pattern = re.compile("([A-Z]*[a-z]*)([\d]*)?""|([A-Z]*[a-z]*)([\d]*)?[_-]([A-Z]*[a-z]*)([\d]*)?")
        self.url_puctuation_pattern = re.compile("[:/=?#]")
        self.str_no_commas_pattern = re.compile("[^-?\d\./]")
        self.url_pattern = re.compile("(?P<url>https?://[^\s]+)")
        self.url_pattern_query = re.compile(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.split_url_pattern = re.compile(r"[\w'|.|-]+")
        self.non_latin_pattern = re.compile(
            pattern=r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF\u2019]')
        self.emojis_pattern = re.compile(
            pattern="["u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"u"\U0001F680-\U0001F6FF"u"\u3030"u"\U00002702-\U000027B0"
                    u"\ufe0f"u"\U0001F1E0-\U0001F1FF"u"\u2640-\u2642"u"\u200d"u"\U00002500-\U00002BEF"u"\U00010000-\U0010ffff"u"\U0001f926-\U0001f937"u"\U000024C2-\U0001F251"u"\u23cf"
                    u"\u23e9"u"\u231a"u"\u2600-\u2B55""]+", flags=re.UNICODE)

    def parse_sentence(self, text, query_tokenized=None):
        """
        This function tokenize, remove stop words and apply lower case for every word within the text
        :param text:
        :return:
        """

        # TODO: how to split the urls and what is need to do different
        text_tokens = word_tokenize(text)
        text_tokens_without_stopwords = [w for w in text_tokens if w not in self.stop_words_dict]
        self.tokens = text_tokens_without_stopwords

        last_number_parsed = None
        count_num_in_a_row = 0
        entity_counter = 1
        is_date = False
        term_dict = {}
        if not Parse.Parsing_a_word:
            term_dict = self.term_dict

        if Parse.Parsing_a_word:
            broken_urls = self.url_pattern_query.findall(text)
            broken_urls = self.parse_url_text(broken_urls)
            for term in broken_urls:
                if "http" not in term and term not in self.additional:
                    if term.isalpha():
                        if term[0].isupper():
                            term = term.upper()
                        else:
                            term = term.lower()
                    if term not in self.term_dict:
                        term_dict[term] = 1
                    else:
                        term_dict[term] += 1
                    if term_dict[term] > self.max_freq_term:
                        self.max_freq_term = term_dict[term]

        for i, token in enumerate(self.tokens):
            if token in self.additional:
                continue
            if entity_counter > 1:
                entity_counter -= 1

            parsed_token_list = []
            number_as_list = []

            if not any(c.isalpha() for c in token):
                number_as_list = self.number_pattern.findall(token)

                is_date = self.parse_date(token)  #: date format

            if "-" in token and not token.startswith("-"):  # not a number starts with "-". example covid-19
                token_before = ""
                if i > 0:
                    token_before = self.tokens[i - 1]
                parsed_token_list = self.parse_hyphen(token, token_before)
                count_num_in_a_row = 0

            elif self.emojis_pattern.match(token):
                continue

            elif token.startswith("'") or token.startswith("-"):
                token = token[1:]

            elif token.endswith("."):
                token = token[:-1]

            elif token.isalpha():  #: capital letters and entities
                entity_str = ""
                if entity_counter == 1:
                    while token.istitle() and i + entity_counter < len(self.tokens) and (
                            self.tokens[i + entity_counter].istitle() or self.istitle_with_hyphen(
                            self.tokens[i + entity_counter])):
                        entity_str += " " + self.tokens[i + entity_counter]
                        entity_counter += 1
                    entity_counter += 1

                parsed_token_list.append(self.check_if_capital(token))
                token += entity_str
                if entity_str != "":
                    parsed_token_list.append(token)
                    if not Parse.Parsing_a_word:
                        if token not in Parse.CAPITAL_LETTER_DICT.keys():
                            Parse.ENTITY_DICT[token] = 1
                        else:
                            Parse.ENTITY_DICT[token] += 1

                count_num_in_a_row = 0

            if token.startswith('@'):  #: @ sign
                if i < len(self.tokens) - 1:
                    parsed_token_list = []
                    count_num_in_a_row = 0
                    self.tokens.pop(i + 1)

            elif token.startswith('#'):  #: # sign
                if i < len(self.tokens) - 1:
                    parsed_token_list = self.parse_hashtag(token + self.tokens[i + 1])
                    count_num_in_a_row = 0
                    self.tokens.pop(i + 1)


            elif is_date:  # date format
                parsed_token_list = [token]
                is_date = False

            elif len(number_as_list) != 0 and len(parsed_token_list) == 0:  #: numbers
                if len(number_as_list) > 1:
                    number_as_list = ["".join(number_as_list)]
                if '-' not in number_as_list[0]:  # if a representation of phone numbers, do nothing
                    count_num_in_a_row += 1
                    if i == 0 and i < len(self.tokens) - 1:
                        parsed_token_list = list(self.parse_numbers(number_as_list[0], None, self.tokens[i + 1]))
                    elif i < len(self.tokens) - 1:
                        parsed_token_list = list(
                            self.parse_numbers(number_as_list[0], self.tokens[i - 1], self.tokens[i + 1]))
                    else:
                        parsed_token_list = list(self.parse_numbers(number_as_list[0], self.tokens[i - 1], None))

                    if count_num_in_a_row == 2 and len(
                            parsed_token_list) == 2:  # for numbers like 25 3/4 that appear together
                        parsed_token_list = [last_number_parsed + " " + parsed_token_list[1]]
                        count_num_in_a_row = 0
                        del term_dict[last_number_parsed]
                    else:
                        last_number_parsed = parsed_token_list[0]
                else:
                    parsed_token_list = number_as_list

            elif "/" in token and "//t" not in token:
                split_slash = token.split("/")
                for word in split_slash:
                    if len(word) > 1:
                        parsed_token_list.append(word)

            if len(parsed_token_list) > 0:

                if self.STEMMER:
                    parsed_token_list_stemmer = []
                    for word in parsed_token_list:
                        if word.isalpha():
                            parsed_token_list_stemmer.append(self.snow_stemmer.stem(word))
                        else:
                            parsed_token_list_stemmer.append(word)
                    parsed_token_list = parsed_token_list_stemmer

                for term in parsed_token_list:
                    if term in self.additional:
                        continue
                    if term not in self.location_dict:
                        self.location_dict[term] = [i]
                    else:
                        self.location_dict[term].append(i)

                    if term not in term_dict:
                        term_dict[term] = 1
                    else:
                        term_dict[term] += 1
                    if term_dict[term] > self.max_freq_term:
                        self.max_freq_term = term_dict[term]
            #############################
            else:
                # garbage = token.isalnum()
                if "//t" not in token and token not in self.dict_punctuation and token not in self.stop_words_dict:
                    is_ascii = self.check_ascii(token)
                    if is_ascii:
                        token = token.lower()

                        if self.STEMMER:
                            token = self.snow_stemmer.stem(token)

                        if token not in self.location_dict:
                            self.location_dict[token] = [i]
                        else:
                            self.location_dict[token].append(i)

                        if token not in term_dict:
                            term_dict[token] = 1
                        else:
                            term_dict[token] += 1
                        if term_dict[token] > self.max_freq_term:
                            self.max_freq_term = term_dict[token]

        return term_dict

    def check_ascii(self, token):
        # if type(token) == int:
        #     return True
        # if len(token) == 0:
        #     return False
        return all((ord(char) > 32) and (ord(char) < 128) for char in token)

    def parse_query(self, query):
        """
        This function takes a query as a string and break it into different fields
        :param query: string representation of the query
        :return: query object with corresponding fields.
        """
        Parse.Parsing_a_word = True
        query_tokenized = word_tokenize(query)
        query_tokenized = [w for w in query_tokenized if w not in self.stop_words_dict]
        query_dict = self.parse_sentence(query)

        location_dict = self.location_dict
        max_freq = self.max_freq_term
        query_length = len(query_dict)
        query = query_object(query_dict, query_length, max_freq, query_tokenized, location_dict)
        self.max_freq_term = 0
        self.term_dict = {}
        self.location_dict = {}

        return query

    def parse_doc(self, doc_as_list):
        """
        This function takes a tweet document as list and break it into different fields
        :param doc_as_list: list re-preseting the tweet.
        :return: Document object with corresponding fields.
        """
        tweet_id = doc_as_list[0]
        tweet_date = doc_as_list[1]
        full_text = doc_as_list[2]
        urls = doc_as_list[3]
        indices = doc_as_list[4]
        retweet_text = doc_as_list[5]
        retweet_urls = doc_as_list[6]
        retweet_indices = doc_as_list[7]
        quoted_text = doc_as_list[8]
        quote_urls = doc_as_list[9]
        quoted_indices = doc_as_list[10]
        retweet_quoted_text = doc_as_list[11]
        retweet_quoted_urls = doc_as_list[12]
        retweet_quoted_indices = doc_as_list[13]

        ########################################
        # TODO: check if indices needed
        # indices_as_list = self.indices_as_list(indices)
        # indices_retweet_as_list = self.indices_as_list(retweet_indices)
        # indices_quoted_as_list = self.indices_as_list(quoted_indices)
        # indices_retweet_quoted_as_list = self.indices_as_list(retweet_quoted_indices)

        raw_urls = self.parse_raw_url(urls, retweet_urls, quote_urls, retweet_quoted_urls, full_text)
        broken_urls = self.parse_url_text(raw_urls)

        for term in broken_urls:
            if "http" not in term and term not in self.additional:
                if term.isalpha():
                    if term[0].isupper():
                        term = term.upper()
                    else:
                        term = term.lower()
                if term not in self.term_dict:
                    self.term_dict[term] = 1
                else:
                    self.term_dict[term] += 1

                if self.term_dict[term] > self.max_freq_term:
                    self.max_freq_term = self.term_dict[term]

        concatenated_text = self.concatenate_tweets(full_text, retweet_text, retweet_quoted_text, quoted_text)
        concatenated_text = self.non_latin_pattern.sub('', concatenated_text)
        term_dict = self.parse_sentence(concatenated_text)

        doc_length = len(self.tokens)  # after text operations.
        Parse.idx += 1
        document = Document(tweet_id, tweet_date, full_text, urls, retweet_text, retweet_urls, quoted_text,
                            quote_urls, term_dict, self.location_dict, doc_length, self.max_freq_term)

        self.max_freq_term = 0
        self.term_dict = {}
        self.location_dict = {}
        return document

    def parse_date(self, token):
        date_list = self.date_pattern.findall(token)
        if len(date_list) > 0:
            return True

        return False

    def parse_hyphen(self, token, token_before):
        """
        :param token: example --> covid-19
        :param token_before: example --> @,%,# to verify how to split
        :return: list of a parsed phrase split by a hyphen --> [covid-19,covid,19]
        """

        to_return = []
        split_hyphen = token.split("-")
        is_alpha = False

        for i in split_hyphen:
            if i != "":
                if i.isalpha():
                    is_alpha = True
                    if i.istitle():
                        self.check_if_capital(i)
            to_return.append(i.lower())

        if not is_alpha:
            return [token]

        return to_return

    def check_if_capital(self, token):
        """
        :param token: example --> Obama,obama
        :return: the token as Upper or Lower and add them to the Global dict
        """
        # for ent in token:
        ent = token
        rest_of_token = ent[1:].upper()
        ent = ent[0] + rest_of_token
        if ent.isupper():
            if ent not in Parse.CAPITAL_LETTER_DICT:
                Parse.CAPITAL_LETTER_DICT[ent] = True
            return ent

        else:
            new_word = ent.upper()  # title
            if new_word in Parse.CAPITAL_LETTER_DICT:
                Parse.CAPITAL_LETTER_DICT[new_word] = False

            lower = ent.lower()
            return lower

    def parse_hashtag(self, token):
        """
        hashtags parsing
        :param token: example --> #stay_at_home
        :return: list of a decomposed hashtag --> [stay,at,home,#stayathome]
        """
        tokens_with_hashtag = [token.lower()]
        token = token.split("#")[1]
        tokens_with_hashtag.extend(([a.lower() for a in self.hashtag_pattern.split(token) if a]))
        tokens_with_hashtag.pop(0)
        return tokens_with_hashtag

    def parse_url_text(self, urls):
        """
        :param urls: example --> https://www.instagram.com/p/CD7fAPWs3WM/?igshid=o9kf0ugp1l8x
        :return: list of a parsed phrase split by set of rules --> [https, www, instagram.com, p, CD7fAPWs3WM , igshid , o9kf0ugp1l8x]
        """
        to_return = []
        for token in urls:

            url = self.split_url_pattern.findall(token)
            for i, elem in enumerate(url):
                if 'www.' in elem:
                    address = url[i].split('.', 2)
                    url[i] = address[1]
                    to_return.extend([address[1]])
        return to_return

    def parse_numbers(self, number_as_str, word_before, word_after):

        """
        :param number_as_str: example -->  a number to split
        :param word_before: example --> can be a sign
        :param word_after: example --> can be a sign or quantity
        :return: list of a parsed phrase split by set of rules
        """
        str_no_commas = self.str_no_commas_pattern.sub("", number_as_str)
        signs = {'usd': '$', 'aud': '$', 'eur': '€', '$': '$', '€': '€', '£': '£', 'percent': '%',
                 'percentage': '%',
                 '%': '%'}
        quantities = {"thousands", "thousand", "millions", "million", "billions", "billion"}
        quantity = ""
        result = ""
        alpha = ''
        division_as_is = ""
        is_division = False

        if number_as_str[-1].isalpha():
            alpha = number_as_str[-1]

        if word_before is not None:
            word_before = word_before.lower()

        if word_after is not None:
            word_after = word_after.lower()

        if "/" in str_no_commas:
            amount_of_sleshes = str_no_commas.count("/")
            division_as_is = str_no_commas
            if amount_of_sleshes > 1:
                return [str_no_commas]
            is_division = True
            num, denum = str_no_commas.split('/')
            if denum == "0":
                return [str_no_commas]
            as_number = float(num) / float(denum)
        elif "." in str_no_commas:
            amount_of_dots = str_no_commas.count(".")
            if amount_of_dots > 1:
                return [str_no_commas]
            as_number = float("{:.3f}".format(float(str_no_commas)))
        else:
            as_number = int(str_no_commas)

        numbers_signs_list = [""] * 3

        if word_before in signs:  # looks for signs like $ %
            numbers_signs_list[0] = signs[word_before]

        if word_after is None or (word_after not in signs and word_after not in quantities):
            if as_number < 1000:
                numbers_signs_list[1] = str(as_number)
            elif as_number < 1000000:
                quantity = 'K'
                numbers_signs_list[1] = str(as_number / 1000)
            elif 1000000 < as_number < 1000000000:
                quantity = 'M'
                numbers_signs_list[1] = str(as_number / 1000000)
            elif as_number > 1000000000:
                quantity = 'B'
                numbers_signs_list[1] = str(as_number / 1000000000)

        else:
            if word_after in signs:  # looks for signs like $ %
                numbers_signs_list[1] = str(as_number)
                numbers_signs_list[2] = signs[word_after]

            elif word_after in quantities:  # thousand, million etc.
                if word_after == "thousands" or word_after == "thousand":
                    if as_number < 1000:
                        quantity = 'K'
                        numbers_signs_list[1] = str(as_number)
                    elif as_number < 1000000:
                        quantity = 'M'
                        numbers_signs_list[1] = str(as_number / 1000)
                    else:
                        quantity = 'B'
                        numbers_signs_list[1] = str(as_number / 1000000)

                elif word_after == "millions" or word_after == "million":
                    if as_number < 1000:
                        quantity = 'M'
                        numbers_signs_list[1] = str(as_number)
                    elif as_number < 1000000:
                        quantity = 'B'
                        numbers_signs_list[1] = str(as_number / 1000)

                elif word_after == "billions" or word_after == "billion":
                    quantity = 'B'
                    numbers_signs_list[1] = str(as_number)

        if "." in numbers_signs_list[1]:
            numbers_signs_list[1] = str(float("{:.3f}".format(float(numbers_signs_list[1]))))
        if quantity == "":
            numbers_signs_list[1] = numbers_signs_list[1] + alpha
            if is_division:
                division_as_is = division_as_is + alpha

        if is_division:
            division_as_is = numbers_signs_list[0] + division_as_is + quantity + numbers_signs_list[2]
        numbers_signs_list[1] = numbers_signs_list[1] + quantity
        ret = result.join(numbers_signs_list)
        returnlist = [ret, division_as_is]

        if returnlist[1] != "":
            Parse.AMOUNT_OF_NUMBERS_IN_CORPUS += 2
            return returnlist
        Parse.AMOUNT_OF_NUMBERS_IN_CORPUS += 1
        return [ret]

    def parse_raw_url(self, url, retweet_url, quote_url, retweet_quoted_urls, full_text):
        """
          :param url:
          :param retweet_url:
          :param quote_url:
          :param retweet_quoted_urls:
          :param full_text: original tweet text
          :return: return a set of the related url to the original tweet
          """
        parsed_token_list = set()
        input_list = [url, retweet_url, quote_url, retweet_quoted_urls]
        if url == "{}":
            url_from_text = self.url_pattern.findall(full_text)
            if len(url_from_text) > 0:
                parsed_token_list.add(url_from_text[0])

        for urls in input_list:
            if urls is not None and urls != "{}":

                url_as_dict = (json.loads(urls))
                for key in url_as_dict.keys():
                    if url_as_dict[key] is None:
                        parsed_token_list.add(key)
                    else:
                        parsed_token_list.add(url_as_dict[key])

        return parsed_token_list

    def indices_as_list(self, indices):
        """
        :param indices: example --> '[174,203]'
        :return: list of the indices as integers
        """
        indices_as_list = []
        if (indices is not None) and (indices != ""):
            indices_as_list = (list(filter(''.__ne__, re.findall("\d*", indices))))
        for i in range(len(indices_as_list)):
            indices_as_list[i] = int(indices_as_list[i])

        return indices_as_list

    def concatenate_tweets(self, tweet, retweet_text, retweet_quoted_text, quoted_text):
        """
          :param tweet:
          :param retweet_text:
          :param retweet_quoted_text:
          :param quoted_text:
          :return: connect the text together depends on their existences
          """
        tweet_to_return = tweet
        if retweet_quoted_text is not None:
            tweet_to_return += " "
            tweet_to_return += retweet_quoted_text
        if quoted_text is not None and retweet_quoted_text != quoted_text:
            tweet_to_return += " "
            tweet_to_return += quoted_text

        return tweet_to_return

    def istitle_with_hyphen(self, token):

        if "-" in token:
            hyphen_index = token.find("-")
            before_hyphen = token[:hyphen_index]
            after_hyphen = token[hyphen_index:]
            if before_hyphen.istitle() and after_hyphen.istitle():
                return True

        return False


# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from document import Document
#
#
# class Parse:
#
#     def __init__(self):
#         self.stop_words = frozenset(stopwords.words('english'))
#
#     def parse_sentence(self, text):
#         """
#         This function tokenize, remove stop words and apply lower case for every word within the text
#         :param text:
#         :return:
#         """
#         text_tokens = word_tokenize(text)
#         text_tokens_without_stopwords = [w.lower() for w in text_tokens if w not in self.stop_words]
#         return text_tokens_without_stopwords
#
#     def parse_doc(self, doc_as_list):
#         """
#         This function takes a tweet document as list and break it into different fields
#         :param doc_as_list: list re-presenting the tweet.
#         :return: Document object with corresponding fields.
#         """
#         tweet_id = doc_as_list[0]
#         tweet_date = doc_as_list[1]
#         full_text = doc_as_list[2]
#         url = doc_as_list[3]
#         retweet_text = doc_as_list[4]
#         retweet_url = doc_as_list[5]
#         quote_text = doc_as_list[6]
#         quote_url = doc_as_list[7]
#         term_dict = {}
#         tokenized_text = self.parse_sentence(full_text)
#
#         doc_length = len(tokenized_text)  # after text operations.
#
#         for term in tokenized_text:
#             if term not in term_dict.keys():
#                 term_dict[term] = 1
#             else:
#                 term_dict[term] += 1
#
#         document = Document(tweet_id, tweet_date, full_text, url, retweet_text, retweet_url, quote_text,
#                             quote_url, term_dict, doc_length)
#         return document
