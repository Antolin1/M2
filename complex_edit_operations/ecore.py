import networkx as nx

from m2_generator.edit_operation.edit_operation import EditOperation


def get_complex_reference():
    pattern_ar = nx.MultiDiGraph()
    pattern_ar.add_node(0, type=['EClass'], ids={0})
    pattern_ar.add_node(1, type=['EClass'], ids={1})
    pattern_ar.add_node(2, type='EReference')
    pattern_ar.add_edge(0, 2, type='eStructuralFeatures')
    pattern_ar.add_edge(2, 0, type='eContainingClass')
    pattern_ar.add_edge(2, 1, type='eType')

    pattern_ar_it = nx.MultiDiGraph()
    pattern_ar_it.add_node(0, type=['EClass'], ids={0, 1})
    pattern_ar_it.add_node(1, type='EReference')
    pattern_ar_it.add_edge(0, 1, type='eStructuralFeatures')
    pattern_ar_it.add_edge(1, 0, type='eContainingClass')
    pattern_ar_it.add_edge(1, 0, type='eType')

    patterns = [pattern_ar, pattern_ar_it]

    return EditOperation(patterns, ids=[0, 1], name='Add Reference Complex')


def get_complex_reference_eopposite():
    pattern_areo = nx.MultiDiGraph()
    pattern_areo.add_node(0, type=['EClass'], ids={0})
    pattern_areo.add_node(1, type=['EClass'], ids={1})
    pattern_areo.add_node(2, type='EReference')
    pattern_areo.add_node(3, type='EReference')
    pattern_areo.add_edge(0, 2, type='eStructuralFeatures')
    pattern_areo.add_edge(2, 0, type='eContainingClass')
    pattern_areo.add_edge(2, 1, type='eType')
    pattern_areo.add_edge(1, 3, type='eStructuralFeatures')
    pattern_areo.add_edge(3, 1, type='eContainingClass')
    pattern_areo.add_edge(3, 0, type='eType')
    pattern_areo.add_edge(3, 2, type='eOpposite')
    pattern_areo.add_edge(2, 3, type='eOpposite')

    pattern_areo_it = nx.MultiDiGraph()
    pattern_areo_it.add_node(0, type=['EClass'], ids={0, 1})
    pattern_areo_it.add_node(1, type='EReference')
    pattern_areo_it.add_node(2, type='EReference')
    pattern_areo_it.add_edge(0, 1, type='eStructuralFeatures')
    pattern_areo_it.add_edge(1, 0, type='eContainingClass')
    pattern_areo_it.add_edge(1, 0, type='eType')
    pattern_areo_it.add_edge(0, 2, type='eStructuralFeatures')
    pattern_areo_it.add_edge(2, 0, type='eContainingClass')
    pattern_areo_it.add_edge(2, 0, type='eType')
    pattern_areo_it.add_edge(2, 1, type='eOpposite')
    pattern_areo_it.add_edge(1, 2, type='eOpposite')

    patterns = [pattern_areo, pattern_areo_it]

    return EditOperation(patterns, ids=[0, 1], name='Add Reference eOpposite Complex')


def get_complex_eattribute():
    pattern_aea = nx.MultiDiGraph()
    pattern_aea.add_node(0, type=['EClass'], ids={0})
    pattern_aea.add_node(1, type=['EDataType', 'EEnum'], ids={1})
    pattern_aea.add_node(2, type='EAttribute')
    pattern_aea.add_edge(0, 2, type='eStructuralFeatures')
    pattern_aea.add_edge(2, 0, type='eContainingClass')
    pattern_aea.add_edge(2, 1, type='eType')

    patterns = [pattern_aea]

    return EditOperation(patterns, ids=[0, 1], name='Add EAttribute Complex')
