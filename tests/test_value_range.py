import unittest

import numpy as np
import tensorflow as tf
from deepspeech_training.util.helpers import ValueRange, get_value_range, pick_value_from_range, tf_pick_value_from_range


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
    def __init__(self, *args, **kwargs):
        super(TestPickValueFromFixedRange, self).__init__(*args, **kwargs)
        self.session = tf.Session()
        self.clock_ph = tf.placeholder(dtype=tf.float64, name='clock')

    def _ending_tester(self, value_range, clock, expected):
        with tf.Session() as session:
            tf_pick = tf_pick_value_from_range(value_range, clock=self.clock_ph)

            def run_pick(_, c):
                return session.run(tf_pick, feed_dict={self.clock_ph: c})

            is_int = isinstance(value_range.start, int)
            for pick, int_type, float_type in [(pick_value_from_range, int, float), (run_pick, np.int32, np.float32)]:
                result = pick(value_range, clock)
                self.assertEqual(result, expected)
                self.assertTrue(isinstance(result, int_type if is_int else float_type))

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
    def __init__(self, *args, **kwargs):
        super(TestPickValueFromRandomizedRange, self).__init__(*args, **kwargs)
        self.session = tf.Session()
        self.clock_ph = tf.placeholder(dtype=tf.float64, name='clock')

    def _ending_tester(self, value_range, clock_min, clock_max, expected_min, expected_max):
        with self.session as session:
            tf_pick = tf_pick_value_from_range(value_range, clock=self.clock_ph)

            def run_pick(_, c):
                return session.run(tf_pick, feed_dict={self.clock_ph: c})

            is_int = isinstance(value_range.start, int)
            clock_range = np.arange(clock_min, clock_max, (clock_max - clock_min) / 100.0)
            for pick, int_type, float_type in [(pick_value_from_range, int, float), (run_pick, np.int32, np.float32)]:
                results = [pick(value_range, c) for c in clock_range]
                self.assertGreater(len(set(results)), 80)
                self.assertTrue(all(map(lambda x: expected_min <= x <= expected_max, results)))
                self.assertTrue(all(map(lambda x: isinstance(x, int_type if is_int else float_type), results)))

    def test_int_0(self):
        self._ending_tester(ValueRange(10000, 30000, 10000), 0.0, 0.1, 0, 22000)

    def test_int_half(self):
        self._ending_tester(ValueRange(10000, 30000, 10000), 0.4, 0.6, 8000, 32000)

    def test_int_1(self):
        self._ending_tester(ValueRange(10000, 30000, 10000), 0.8, 1.0, 16000, 40000)

    def test_float_0(self):
        self._ending_tester(ValueRange(10000.0, 30000.0, 10000.0), 0.0, 0.1, 0.0, 22000.0)

    def test_float_half(self):
        self._ending_tester(ValueRange(10000.0, 30000.0, 10000.0), 0.4, 0.6, 8000.0, 32000.0)

    def test_float_1(self):
        self._ending_tester(ValueRange(10000.0, 30000.0, 10000.0), 0.8, 1.0, 16000.0, 40000.0)


if __name__ == '__main__':
    unittest.main()
