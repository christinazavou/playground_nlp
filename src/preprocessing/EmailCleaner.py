# -*- coding: utf-8 -*-
import re
import tldextract  # separates the generic or country code top-level domain from the registered domain and subdomains of a URL.
import queue
import multiprocessing
from collections import defaultdict


def get_flags(flag=None):
    if flag is not None:
        return re.U | re.IGNORECASE | flag
    else:
        return re.U | re.IGNORECASE


class EmailCleaner(object):

    MAX_PATH_LENGTH = 100
    code_by_normalized_code_map = {}  # {normalized_code : {code_1: counts1, code_2: counts2}}

    @staticmethod
    def remove_headers_footers(text):
        text = re.sub(r'(e{0,1}-{0,1}mail:{0,1}\s*)([^@\s]+@[^@\s]+(\.[^@\s]+){1,2})', u'\n',  text, flags=get_flags())
        text = re.sub(r'(^[\s>]*(tel|mob|phone|fax|fon)[\w\-]{0,10}:[0-9\s\+\-\)\(]+$)',
                      u'\n',
                      text,
                      flags=get_flags(re.MULTILINE))
        text = re.sub(r'(^[\s>]*(post|adres)[^\n]+$)', u'\n', text, flags=get_flags(re.MULTILINE))
        text = re.sub(r'(^[\s>]*(datum|date):[^\n]+$)', u'\n', text, flags=get_flags(re.MULTILINE))
        text = re.sub(r'(^[\s>]*(verzonden|sent):[ \t\w0-9\-:,\(\)\+;<>\[\]\{\}]+$)',
                      u'\n',
                      text,
                      flags=get_flags(re.MULTILINE)) # prosoxi mipos svinei kai kati allo pou einai sent: ...
        text = re.sub(r'(^[\s>]*(met ){0,1}(vriendelijke ){0,1}groet.*$)', u'\n', text, flags=get_flags(re.MULTILINE))
        text = re.sub(r'(^[\s>]*(thank|best|geachte|goede|dear|hello|hi|hallo|hoi|bedankt|((met\s){0,1}(vriendelijke\s){0,1}(groet)))).{0,20}$',
                      u'\n',
                      text,
                      flags=get_flags(re.MULTILINE))
        text = re.sub(r'(^[\s>]*[^\w]*(original message)|forwarded message[^\w]*$)',
                      u'\n',
                      text,
                      flags=get_flags(re.MULTILINE))
        text = re.sub(r'(^[\s>]*[^\w]*(oorspronkelijk bericht)[^\w]*$)', u'\n', text, flags=get_flags(re.MULTILINE))
        return text

    @staticmethod
    def remove_english_disclaimers(text):
        """
        English Disclaimers – only supported if they are found in one line.
        If the disclaimer text is split into multiple lines these regex expressions can’t identify it.
        """
        # Phrases/words “information or contents”, “message, mail or communication”, and “protected or confidential”
        # are found in this order in one line along with other things.
        text = re.sub(u'(^([^\n]*)(information|contents)([^\n]+)(this)([^\n]+)(message|mail|communication)([^\n]+)'
                      u'(protected|confidential)([^\n]+)$)+', u'', text, flags=get_flags(re.MULTILINE))
        # A line starts with “disclaimer:”
        text = re.sub(r'(^[\s>]*(disclaimer:)[^\n]+$)+', u'', text, flags=get_flags(re.MULTILINE))
        # Phrases/words “this”, “message, mail or communication”, and
        # “disclaimer followed by a link” are found in this order in one line along with other things.
        text = re.sub(u'(^([^\n]*)(this)([^\n]+)(message|mail|communication)([^\n]+)(disclaimer)([^\n]+)'
                      u'(http:|www.)([^\n]+)$)+', u'', text, flags=get_flags(re.MULTILINE))
        # Phrases/words “this”, “message, mail or communication”, “confidential or privileged” and “intended recipient
        # or designated recipient or adresse or individual” are found in this order in one line along with other things.
        text = re.sub(u'(^[^\n]*(this)[^\n]+(message|mail|communication)[^\n]+(confidential|privileged)[^\n]+'
                      u'(intended recipient|designated recipient|addresse|individual)[^\n]+$)', u'', text,
                      flags=get_flags(re.MULTILINE))
        # Phrases/words “this”, “message, mail or communication”, “confidential or privileged”, “received”, “error” and
        # “notify the sender” are found in this order in one line along with other things.
        text = re.sub(u'(^[^\n]*(this)[^\n]+(message|mail|communication)[^\n]+(confidential|privileged)[^\n]+'
                      u'(received)[^\n]*(error)[^\n]*(notify the sender)[^\n]+$)', u'', text, flags=get_flags(re.MULTILINE))
        # Phrases/words “this”, “message, mail or communication”, “information” and “not the intended recipient or
        # addresse or individual” are found in this order in one line along with other things.
        text = re.sub(u'(^[^\n]*(this)[^\n]+(message|mail|communication)[^\n]+(information)[^\n]+'
                      u'(not the intended recipient|addresse|individual)[^\n]+$)', u'', text, flags=get_flags(re.MULTILINE))
        # Phrases/words “this”, “message, mail or communication”, “confidential or privileged”, “unauthorized use” and
        # “legal restriction” are found in this order in one line along with other things.
        text = re.sub(u'(^[^\n]*(this)[^\n]+(message|mail|communication)[^\n]+(confidential|privileged)[^\n]+'
                      u'(unauthorized use)[^\n]+(legal restriction)[^\n]+$)', u'', text, flags=get_flags(re.MULTILINE))
        # Phrases/words “distribution or forwarding or copy”, “communication or information or mail” and “strictly
        # prohibited” are found in this order in one line along with other things.
        text = re.sub(u'(^[^\n]*(distribution|forwarding|copy)[^\n]+(communication|information|mail)[^\n]+'
                      u'(strictly prohibited)[^\n]+$)', u'', text, flags=get_flags(re.MULTILINE))
        return text

    @staticmethod
    def remove_finnish_disclaimers(text):
        return text

    @staticmethod
    def remove_dutch_disclaimers(text):
        """
        Dutch disclaimers – only supported if they are found in one line.
        If the disclaimer text is split into multiple lines these regex expressions can’t identify it.
        """
        text = re.sub(u'(^[^\n]*(dit|deze)[^\n]+(bericht|mail)[^\n]+(onrechte|onterecht)[^\n]+((direct contact)|'
                      u'(afzender direct))[^\n]+$)', u'', text, flags=get_flags(re.MULTILINE))
        text = re.sub(u'(^[^\n]*(gebruik)[^\n]+(zonder toestemming van de afzender)[^\n]+(onrechtmatig)[^\n]+$)', u'',
                      text, flags=get_flags(re.MULTILINE))
        text = re.sub(u'(^[^\n]*(geen aansprakelijkheid)[^\n]+(organisatie)[^\n]+(bijlagen|mail)[^\n]+$)', u'',
                      text, flags=get_flags(re.MULTILINE))
        text = re.sub(u'(^[^\n]*(deze aanvraag is gesloten en samengevoegd met aanvraag)[^\n]+$)', u'', text,
                      flags=get_flags(re.MULTILINE))
        text = re.sub(u'(^[^\n]*(dit|deze)[^\n]+(bericht|mail)[^\n]+(disclaimer)[^\n]+$)', u'', text,
                      flags=get_flags(re.MULTILINE))
        text = re.sub(u'(^[^\n]*(dit|deze)[^\n]+(bericht|mail)[^\n]+(vertrouwelijk)[^\n]+(bestemd voor de geadresseerde)'
                      u'[^\n]+$)', u'', text, flags=get_flags(re.MULTILINE))
        text = re.sub(u'(^[^\n]*(dit|deze)[^\n]+(bericht|mail)[^\n]+(inhoud niet te gebruiken)[^\n]+(afzender)[^\n]+$)',
                      u'', text, flags=get_flags(re.MULTILINE))
        text = re.sub(u'(^[^\n]*(verspreiding|vermenigvuldiging)[^\n]+(verboden)[^\n]+(dit|deze)[^\n]+(bericht|mail)'
                      u'^[^\n]+(afzender|verwijderen|sender)[^\n]+$)', u'', text, flags=get_flags(re.MULTILINE))
        text = re.sub(u'(^[^\n]*(dit|deze)[^\n]+(bericht|mail)[^\n]+(bestemd voor de geadresseerde)[^\n]+'
                      u'(openbaarmaking|gebruik|verstrekking)[^\n]+(niet toegestaan|onrechtmatig)[^\n]+$)', u'',
                      text, flags=get_flags(re.MULTILINE))
        return text

    @staticmethod
    def remove_disclaimers(text, languages):
        for language in languages:
            if language == 'finnish':
                text = EmailCleaner.remove_finnish_disclaimers(text)
            elif language == 'english':
                text = EmailCleaner.remove_english_disclaimers(text)
            elif language == "dutch":
                text = EmailCleaner.remove_dutch_disclaimers(text)
        return text

    @staticmethod
    def remove_emails(text, tag=None):
        """
        Regular Expression for emails identification: r'(([\w\d_\.-]+)@([\d\w\.-]+)\.([\w\.]{2,6}))'

        It tag is None, then emails are kept and text is returned as is.
        """
        if not tag:
            return text
        return re.sub(r'(([\w\d_\.-]+)@([\d\w\.-]+)\.([\w\.]{2,6}))', tag, text, flags=get_flags())

    def replace_text_and_update_map(self, pattern, remove_pattern, text, url=False):
        matches = re.findall(pattern, text, flags=get_flags())
        if matches:
            matches = [match[0] for match in matches]  # since we have an inner group i.e. parenthesis in the pattern
            matches = list(set(matches))
            matches = sorted(matches, key=lambda tup: len(tup), reverse=True)
            for match in matches:

                replace_match = match
                if url:
                    replace_match = u"".join(tldextract.extract(match))

                replace_match = re.sub(remove_pattern, '', replace_match, flags=get_flags())
                text = text.replace(match, replace_match)
                if 0 < len(replace_match) <= self.MAX_PATH_LENGTH:
                    self.code_by_normalized_code_map.setdefault(replace_match, defaultdict(int))

                    match_to_replace = match
                    if url:
                        match_to_replace = u".".join(tldextract.extract(match))

                    self.code_by_normalized_code_map[replace_match][match_to_replace] += 1
        return text

    def remove_code_text(self, text, tag=None):
        """
        Regular expression for codes identification:
        r'([\w0-9_/\\-]+\.[\w0-9_/\\-]+\.[\w0-9_/\\-]+(\.[\w0-9_/\\-]+)*)'

        NOTE: should first remove urls and emails, paths

        If tag is remove_punctuation then it removes the codes. If tag is None, then it keeps the codes and returns the
        text as it. Otherwise, it replaces the code occurrences with the given tag.
        :param tag string
        """
        if tag == u'remove_punctuation':
            return self.replace_text_and_update_map(
                r'([\w0-9_/\\-]+\.[\w0-9_/\\-]+\.[\w0-9_/\\-]+(\.[\w0-9_/\\-]+)*)',
                r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]\^_`{|\}~0-9]',
                text)
        elif not tag:
            return text
        else:
            return re.sub(r'([\w0-9_/\\-]+\.[\w0-9_/\\-]+\.[\w0-9_/\\-]+(\.[\w0-9_/\\-]+)*)', tag, text, flags=get_flags())

    def remove_path_text(self, text, tag=None):
        """
        Regular Expression for paths identification:
        r'([\w0-9_.-:]*/[\w0-9_.-]+/[\w0-9_.-]+(/[\w0-9_.-]+)*)' and
        r'([\w0-9_.-:]*\\[\w0-9_.-]+(\\[\w0-9_.-]+)*)'

        NOTE: should first remove urls and emails
        NOTE: finds false positives the dates, e.g. 10/03/2017

        If tag is remove_punctuation then it removes the paths. If tag is None, then it keeps the paths and returns the
        text as it. Otherwise, it replaces the path occurrences with the given tag.
        :param tag string
        """

        path_pattern1 = r'([\w0-9_.-:]*/[\w0-9_.-]+/[\w0-9_.-]+(/[\w0-9_.-]+)*)'# at least two '/' to ignore 'and/or'
        path_pattern2 = r'([\w0-9_.-:]*\\[\w0-9_.-]+(\\[\w0-9_.-]+)*)'

        if tag == u'remove_punctuation':
            text = self.replace_text_and_update_map(
                path_pattern1, r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]\^_`{|\}~0-9]', text)

            return self.replace_text_and_update_map(
                path_pattern2, r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]\^_`{|\}~0-9]', text)

        elif not tag:
            return text

        else:
            text = re.sub(path_pattern1, tag, text, flags=get_flags())
            return re.sub(path_pattern2, tag, text, flags=get_flags())

    def remove_urls(self, text, tag=None):
        """
        Regular Expression for URLs identification:
        r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'
        r'(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\)){0,3}(?:[^\s`!()\[\]{};:\'".,<>?«»“”‘’])*)'

        If tag is keep_domain then it replaces each occurrence of url with its domain. If tag is None, then it keeps
        the urls and returns the text as it. Otherwise, it replaces the url occurrences with the given tag.
        :param tag string
        """
        if not tag:
            return text

        url_pattern = r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'\
                      r'(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\)){0,3}'\
                      r'(?:[^\s`!()\[\]{};:\'".,<>?«»“”‘’])*'\
                      r')'

        if tag == u'keep_domain':

            return self.replace_text_and_update_map(url_pattern, r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]\^_`{|\}~0-9]', text, True)

        return re.sub(url_pattern, tag, text, flags=get_flags())

    @staticmethod
    def remove_small_words(sentence_words, min_len=3):
        words = []
        for word in sentence_words:
            if len(word) - word.count('_') >= min_len:
                words.append(word)
        return words

    @staticmethod
    def remove_long_words(sentence_words, max_len=20):
        words = []
        for word in sentence_words:
            if len(word) - word.count('_') <= max_len:
                words.append(word)
        return words

    # def rmv_url_process(self, in_queue, out_queue):
    #
    #     kwargs = in_queue.get()
    #     if 'text' in kwargs.keys() and 'tag' in kwargs.keys():
    #         text, tag = kwargs['text'], kwargs['tag']
    #     else:
    #         raise Exception('no text or tag given')
    #
    #     if not tag:
    #         out_queue.put(text)
    #         return
    #     if tag == u'keep_domain':
    #         # any url
    #
    #         text = self.replace_text_and_update_map(
    #             r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'
    #                              r'(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|'
    #                              r'(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
    #             r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]\^_`{|\}~0-9]',
    #             text,
    #             True)
    #
    #         out_queue.put(text)
    #         return
    #     text = re.sub(r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)'
    #                   r'(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|'
    #                   r'[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', tag, text, flags=re.U)
    #     out_queue.put(text)
    #     return
    #
    # # todo: call an alternative function if this was terminated from time limit
    # def call_function_with_regex_with_time_limit(self, timed_function, seconds, **kwargs):
    #
    #     in_queue, out_queue = multiprocessing.Manager().Queue(), multiprocessing.Manager().Queue()
    #
    #     in_queue.put(kwargs)
    #     proc = multiprocessing.Process(target=timed_function, args=(in_queue, out_queue,))
    #     proc.start()
    #
    #     try:
    #         result = out_queue.get(timeout=seconds)
    #
    #     except queue.Empty:  # timeout expired
    #         proc.terminate()  # kill the subprocess
    #         print('process terminated from time out')
    #         result = kwargs['text']  # SHOULD EXIST !
    #     return result


# multiprocessing.freeze_support()

# print call_function_with_regex_with_time_limit(rmv_url_process, 10, text=text_1, tag='keep_domain')
