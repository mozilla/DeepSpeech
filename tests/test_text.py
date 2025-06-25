import unittest
import os

from ds_ctcdecoder import Alphabet

class TestAlphabetParsing(unittest.TestCase):

    def _ending_tester(self, file, expected):
        alphabet_file_path = os.path.join(os.path.dirname(__file__), 'test_data', file)
        with open(alphabet_file_path, 'r') as f:
            alphabet_data = f.read().splitlines()
        alphabet = Alphabet(alphabet_data)
        for expected_label, expected_label_id in expected:
            try:
                label_id = alphabet.Encode(expected_label)
                self.assertEqual(label_id, [expected_label_id])
            except KeyError:
                self.fail(f"Failed to encode label '{expected_label}'")
            try:
                label = alphabet.Decode([expected_label_id])
                self.assertEqual(label, expected_label)
            except KeyError:
                self.fail(f"Failed to decode label '{expected_label_id}'")

    def test_macos_ending(self):
        self._ending_tester('alphabet_macos.txt', [('a', 0), ('b', 1), ('c', 2)])

    def test_unix_ending(self):
        self._ending_tester('alphabet_unix.txt', [('a', 0), ('b', 1), ('c', 2)])

    def test_windows_ending(self):
        self._ending_tester('alphabet_windows.txt', [('a', 0), ('b', 1), ('c', 2)])

if __name__ == '__main__':
    unittest.main()
