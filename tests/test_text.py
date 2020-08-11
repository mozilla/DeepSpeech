import unittest
import os

from mvs_ctcdecoder import Alphabet

class TestAlphabetParsing(unittest.TestCase):

    def _ending_tester(self, file, expected):
        alphabet = Alphabet(os.path.join(os.path.dirname(__file__), 'test_data', file))
        label = ''
        label_id = -1
        for expected_label, expected_label_id in expected:
            try:
                label_id = alphabet.Encode(expected_label)
            except KeyError:
                pass
            self.assertEqual(label_id, [expected_label_id])
            try:
                label = alphabet.Decode([expected_label_id])
            except KeyError:
                pass
            self.assertEqual(label, expected_label)

    def test_macos_ending(self):
        self._ending_tester('alphabet_macos.txt', [('a', 0), ('b', 1), ('c', 2)])

    def test_unix_ending(self):
        self._ending_tester('alphabet_unix.txt', [('a', 0), ('b', 1), ('c', 2)])

    def test_windows_ending(self):
        self._ending_tester('alphabet_windows.txt', [('a', 0), ('b', 1), ('c', 2)])

if __name__ == '__main__':
    unittest.main()
