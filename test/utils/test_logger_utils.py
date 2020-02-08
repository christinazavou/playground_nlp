import unittest
import tempfile
import shutil
import os


class TestLoggerUtilsMethods(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_manage_logger(self):
        from src.utils.logger_utils import manage_logger

        logger = manage_logger(__name__, 'INFO', self.test_dir, os.path.join(self.test_dir, 'output.log'))
        logger.info('My unit test is fine.')

        f = open(os.path.join(self.test_dir, 'output.log'))
        self.assertTrue("My unit test is fine." in f.read())


if __name__ == '__main__':
    unittest.main()
