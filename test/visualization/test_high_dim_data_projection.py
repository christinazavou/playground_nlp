import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from src.visualization.high_dim_data_projection import run_tsne_projection


class TestTopic(unittest.TestCase):
    def test(self):
        data = [
            [-0.0025216, - 0.00745017, - 0.0086065, -0.00756991, 0.00173239, - 0.00226903],
            [-0.13599935, - 0.04337048, - 0.12416705, -0.02677078, - 0.09994805, - 0.00100875],
            [-0.16225441, - 0.06430624, - 0.12654391, -0.07450635, - 0.076726, 0.01296266],
            [-0.16825441, - 0.06430624, - 0.12554391, -0.07452635, - 0.077726, 0.01291266],
            [-0.16225441, - 0.06460624, - 0.13654391, -0.07410635, - 0.076726, 0.01226266],
            [-0.16245441, - 0.06730624, - 0.12657391, -0.07460635, - 0.074726, 0.01396266]
        ]
        labels = np.array([342.0, 12.0, 368.0, 12.0, 368.0, 342.0])

        with TemporaryDirectory() as d:
            temp_file_name = os.path.join(d, 'tsne.png')

            run_tsne_projection(data,
                                temp_file_name,
                                None,
                                labels,
                                verbose=1,
                                early_exaggeration=4,
                                n_iter=500,
                                metric='cosine',
                                init='pca',
                                random_state=200)

            self.assertTrue(os.path.exists(temp_file_name))


if __name__ == '__main__':
    unittest.main()
