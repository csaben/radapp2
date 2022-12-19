#!/usr/bin/env python

import unittest
import dataset
from dataset import ds_helper
import os
import numpy

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.path1 = "../input/DCM/DCM/Control"
        self.path2 = "../input/DCM/DCM/Control/P1/S1/MR-BRAIN-AXIAL/"


    def test_ds_helper(self):
        imgs, labels = ds_helper(self.path1)
        self.dataset  = GliomaDataset(imgs, labels)
        #make sure we make a np array
        self.assertIsInstance(imgs,np.ndarray)
        #make sure we have as many imgs as processed S_type dirs
        self.assertEqual(len(self.dataset), len(imgs))

if __name__ == "__main__":
    unittest.main()
