import unittest
import os
from tempfile import TemporaryDirectory

from src.visualization.topic import topic_cloud


class TestTopic(unittest.TestCase):

    def test(self):
        topic_representation = [
            (u'call', 0.069),
            (u'membership', 0.052),
            (u'customer', 0.035),
            (u'status', 0.035),
            (u'change', 0.034),
            (u'membership call', 0.034),
            (u'URLTAG', 0.018)
        ]

        with TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'topic_cloud.png')
            topic_cloud(topic_representation, filename=temp_file_name)
            self.assertTrue(os.path.exists(temp_file_name))


if __name__ == '__main__':
    unittest.main()


