import unittest
import tensorflow as tf
import matplotlib.pyplot as plt
"""
This file contains tests for general execution related functionality. Such as:
- GPU availability
- Graphical backend
- Importing
- 
"""

class test_exec(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    
    def test_gpus_avail(self):
        """
        Test that the number of GPUs available is greater than 0."""
        avail_gpus = len(tf.config.list_physical_devices('GPU'))
        self.assertTrue(avail_gpus >= 1)

    def test_plotting_works(self):
        """
        Test that plotting works with matplotlib.pyplot
        Plots a figure and then closes it.
        """
        plt.scatter([1,2,3],[1,2,3])
        plt.show()
        plt.close("all") # TODO
    
    def test_importing_is_clear(self):
        import oparch
        oparch.optimize_utils.check_types((1,int),("str",str))


if __name__ == "__main__":
    unittest.main()
