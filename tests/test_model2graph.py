import unittest

from m2_generator.model2graph.model2graph import get_graph_from_model
from tests.test_generation_edit_operations import plot_graph


class TestM2G(unittest.TestCase):
    def test_m2g(self):
        path_metamodel = "data/yakindu_simplified.ecore"
        path_model = "data/yakindu_example.xmi"
        G = get_graph_from_model(path_model, [path_metamodel])
        plot_graph('model2graph test', G)


if __name__ == '__main__':
    unittest.main()
