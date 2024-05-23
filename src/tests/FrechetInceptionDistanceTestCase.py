import unittest
import torch
from ..style_transfer.metrics import FrechetInceptionDistance

class FrechetInceptionDistanceTestCase(unittest.TestCase):
    
    def test_get_frechet_inception_distance(self):
        images_1 = torch.rand(10,3,512,512)
        images_2 = torch.rand(10,3,512,512)
        
        frechet_inception_distance = FrechetInceptionDistance(0)
        frechet_inception_distance.process_generated_images(images_1)
        frechet_inception_distance.process_real_images(images_1)
        
        self.assertAlmostEqual(frechet_inception_distance.get_frechet_inception_distance(0, 1), 0, places=3)
        
        frechet_inception_distance = FrechetInceptionDistance(0)
        frechet_inception_distance.process_generated_images(images_1)
        frechet_inception_distance.process_real_images(images_2)
        
        self.assertLess(frechet_inception_distance.get_frechet_inception_distance(0, 1), 100)
        

if __name__ == '__main__':
    unittest.main()