import unittest
import torch

class TestTrainMethods(unittest.TestCase):
    def test_custom_collate_fn(self):
        images = torch.randint(0,255, (32, 500, 500))
        boxes = torch.randint(0, 500, (32, 4))
        labels = torch.randint(0,3, (32,))
        targets = {'boxes': boxes, 'labels': labels}
        filenames = ['id_example.tif'] * 32
        ids = torch.randint(0, 32)
        batch: list = [images, targets]
        self.assertEqual(12, 45)

if __name__=='__main__':
    unittest.main()