import unittest

from src.preprocessing.EmailCleaner import EmailCleaner


class TestEmailCleanerMethods(unittest.TestCase):

    email_cleaner = EmailCleaner()
    text_1 = u"""You are being notified of 1 alarm(s).
            Alarm URL : http://172.31.136.222:8080/kati/katiallo.jnlp?alarm=323424331-rr2e-1387-030c
            Severity : Critical
            Date/Time : 06-Dec-1993 03:16:00 CET
            Event : ...No such file or directory:bss_file.c:352:fopen('/etc/httpd/conf/cert/kati.kati','r')
            ... 172.16.85.48
            Actions : This largely depends on the actual failing service.
            It may be needed to contact the defined call-out for this service.
            """

    def test_path(self):
        text = self.email_cleaner.remove_path_text(self.text_1, tag='remove_punctuation')
        # print("without path:", text)
        self.assertTrue(self.text_1 != text)
        text = self.email_cleaner.remove_path_text(self.text_1, tag='PATH_TAG')
        # print("with PATH_TAG:", text)
        self.assertTrue('PATH_TAG' in text)
        text = self.email_cleaner.remove_path_text(self.text_1)
        self.assertTrue(self.text_1 == text)

    def test_url(self):
        text = self.email_cleaner.remove_urls(self.text_1, tag='keep_domain')
        # print("without url:", text)
        self.assertTrue(self.text_1 != text)
        text = self.email_cleaner.remove_urls(self.text_1, tag='URL_TAG')
        # print("with URL_TAG:", text)
        self.assertTrue('URL_TAG' in text)
        text = self.email_cleaner.remove_urls(self.text_1)
        self.assertTrue(self.text_1 == text)

    def test_something(self):
        text = self.email_cleaner.remove_urls(self.text_1, tag='keep_domain')
        text = self.email_cleaner.remove_path_text(text, tag='remove_punctuation')
        self.assertTrue(self.email_cleaner.code_by_normalized_code_map != {})


if __name__ == '__main__':

    unittest.main()



