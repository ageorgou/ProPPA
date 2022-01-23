"""Tests for importing ProPPA."""

import unittest


class TestTrivial(unittest.TestCase):
    """A trivial test case."""

    def test_import(self):
        """Test that proppa can be imported."""
        import proppa

        self.assertIsNotNone(proppa)
