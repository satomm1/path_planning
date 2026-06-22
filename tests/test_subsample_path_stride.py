"""Tests for subsample_path_by_stride."""

import unittest

from social_path_planning.path_subsample import subsample_path_by_stride


class TestSubsamplePathByStride(unittest.TestCase):
    def test_stride_one_unchanged(self):
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        out = subsample_path_by_stride(path, 1)
        self.assertEqual(out, path)
        self.assertIsNot(out, path)

    def test_stride_two_keeps_endpoints(self):
        path = [(float(i), 0.0) for i in range(6)]
        out = subsample_path_by_stride(path, 2)
        self.assertEqual(out, [path[0], path[2], path[4], path[5]])

    def test_stride_three(self):
        path = [(float(i), 0.0) for i in range(10)]
        out = subsample_path_by_stride(path, 3)
        self.assertEqual(out[0], path[0])
        self.assertEqual(out[-1], path[-1])
        self.assertEqual(out, [path[0], path[3], path[6], path[9]])

    def test_single_point(self):
        path = [(1.0, 2.0)]
        self.assertEqual(subsample_path_by_stride(path, 5), path)

    def test_stride_larger_than_path(self):
        path = [(0.0, 0.0), (1.0, 1.0)]
        self.assertEqual(subsample_path_by_stride(path, 10), path)

    def test_invalid_stride(self):
        with self.assertRaises(ValueError):
            subsample_path_by_stride([(0.0, 0.0)], 0)


if __name__ == "__main__":
    unittest.main()
