import unittest
from unittest.mock import patch
import numpy as np
from .ignore_regions import validate, rectangles, DEFAULTS
from .localizer import scene_mask


class IgnoreRegionTests(unittest.TestCase):
    def test_multiple_regions_and_no_compulsory_minimap(self):
        with patch('no_minimap_lab.ignore_regions.load',return_value=((.4,.2,.1,.1),(.7,.5,.1,.2))):
            mask=scene_mask((100,200,3))
            self.assertEqual(mask[25,90],0)
            self.assertEqual(mask[60,150],0)
            self.assertEqual(mask[20,20],255)
            self.assertEqual(mask[0,0],255)
        with patch('no_minimap_lab.ignore_regions.load',return_value=()):
            self.assertTrue(np.all(scene_mask((100,200))==255))

    def test_normalized_regions_scale_and_defaults_omit_minimap(self):
        with patch('no_minimap_lab.ignore_regions.load',return_value=((.1,.2,.3,.4),)):
            self.assertEqual(rectangles((100,200)),[(20,20,60,40)])
            self.assertEqual(rectangles((200,400)),[(40,40,120,80)])
        with patch('no_minimap_lab.ignore_regions.load',return_value=DEFAULTS):
            self.assertEqual(scene_mask((100,200))[20,20],255)

    def test_invalid_regions_rejected(self):
        for region in ((0,0,-1,.2),(0,0,float('nan'),.2),(.9,0,.2,.1)):
            with self.assertRaises(ValueError):validate([region])


if __name__=='__main__':
    unittest.main()
