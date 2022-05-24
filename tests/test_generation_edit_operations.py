import unittest

from m2_generator.edit_operation.edit_operation_generation import get_edit_operations


class TestGeneration(unittest.TestCase):
    def test_generation_yakindu(self):
        path_metamodel = "data/yakindu_simplified.ecore"
        all_patterns = get_edit_operations(path_metamodel)
        for patterns in all_patterns:
            print(len(patterns))
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
