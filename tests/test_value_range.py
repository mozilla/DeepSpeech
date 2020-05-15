import unittest

from deepspeech_training.util.helpers import ValueRange, get_value_range, pick_value_from_range


class TestValueRange(unittest.TestCase):

    def _ending_tester(self, value, value_type, expected):
        result = get_value_range(value, value_type)
        self.assertEqual(result, expected)

    def test_int_str_scalar(self):
        self._ending_tester('1', int, ValueRange(1, 1, 0))

    def test_int_str_scalar_radius(self):
        self._ending_tester('1~3', int, ValueRange(1, 1, 3))

    def test_int_str_range(self):
        self._ending_tester('1:2', int, ValueRange(1, 2, 0))

    def test_int_str_range_radius(self):
        self._ending_tester('1:2~3', int, ValueRange(1, 2, 3))

    def test_int_scalar(self):
        self._ending_tester(1, int, ValueRange(1, 1, 0))

    def test_int_2tuple(self):
        self._ending_tester((1, 2), int, ValueRange(1, 2, 0))

    def test_int_3tuple(self):
        self._ending_tester((1, 2, 3), int, ValueRange(1, 2, 3))

    def test_float_str_scalar(self):
        self._ending_tester('1.0', float, ValueRange(1.0, 1.0, 0.0))

    def test_float_str_scalar_radius(self):
        self._ending_tester('1.0~3.0', float, ValueRange(1.0, 1.0, 3.0))

    def test_float_str_range(self):
        self._ending_tester('1.0:2.0', float, ValueRange(1.0, 2.0, 0.0))

    def test_float_str_range_radius(self):
        self._ending_tester('1.0:2.0~3.0', float, ValueRange(1.0, 2.0, 3.0))

    def test_float_scalar(self):
        self._ending_tester(1.0, float, ValueRange(1.0, 1.0, 0.0))

    def test_float_2tuple(self):
        self._ending_tester((1.0, 2.0), float, ValueRange(1.0, 2.0, 0.0))

    def test_float_3tuple(self):
        self._ending_tester((1.0, 2.0, 3.0), float, ValueRange(1.0, 2.0, 3.0))

    def test_float_int_3tuple(self):
        self._ending_tester((1, 2, 3), float, ValueRange(1.0, 2.0, 3.0))


class TestPickValueFromFixedRange(unittest.TestCase):

    def _ending_tester(self, value_range, clock, expected):
        is_int = isinstance(value_range.start, int)
        result = pick_value_from_range(value_range, clock)
        self.assertEqual(result, expected)
        self.assertTrue(isinstance(result, int if is_int else float))

    def test_int_0(self):
        self._ending_tester(ValueRange(1, 3, 0), 0.0, 1)

    def test_int_half(self):
        self._ending_tester(ValueRange(1, 3, 0), 0.5, 2)

    def test_int_1(self):
        self._ending_tester(ValueRange(1, 3, 0), 1.0, 3)

    def test_float_0(self):
        self._ending_tester(ValueRange(1.0, 2.0, 0.0), 0.0, 1.0)

    def test_float_half(self):
        self._ending_tester(ValueRange(1.0, 2.0, 0.0), 0.5, 1.5)

    def test_float_1(self):
        self._ending_tester(ValueRange(1.0, 2.0, 0.0), 1.0, 2.0)


class TestPickValueFromRandomizedRange(unittest.TestCase):

    def _ending_tester(self, value_range, clock, expected_min, expected_max):
        is_int = isinstance(value_range.start, int)
        results = list(map(lambda x: pick_value_from_range(value_range, clock), range(100)))
        self.assertGreater(len(set(results)), 80)
        self.assertTrue(all(map(lambda x: expected_min <= x <= expected_max, results)))
        self.assertTrue(all(map(lambda x: isinstance(x, int if is_int else float), results)))

    def test_int_0(self):
        self._ending_tester(ValueRange(10000, 30000, 10000), 0.0, 0, 20000)

    def test_int_half(self):
        self._ending_tester(ValueRange(10000, 30000, 10000), 0.5, 10000, 30000)

    def test_int_1(self):
        self._ending_tester(ValueRange(10000, 30000, 10000), 1.0, 20000, 40000)

    def test_float_0(self):
        self._ending_tester(ValueRange(10000.0, 30000.0, 10000.0), 0.0, 0.0, 20000.0)

    def test_float_half(self):
        self._ending_tester(ValueRange(10000.0, 30000.0, 10000.0), 0.5, 10000.0, 30000.0)

    def test_float_1(self):
        self._ending_tester(ValueRange(10000.0, 30000.0, 10000.0), 1.0, 20000.0, 40000.0)


if __name__ == '__main__':
    unittest.main()
