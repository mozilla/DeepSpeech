import unittest

from .importers import validate_label_eng

class TestValidateLabelEng(unittest.TestCase):

    def test_numbers(self):
        label = validate_label_eng("this is a 1 2 3 test")
        self.assertEqual(label, None)

if __name__ == '__main__':
    unittest.main()
