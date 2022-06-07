import networkx as nx

from m2_generator.edit_operation.edit_operation import EditOperation


def get_complex_add_transition_edit_operation():
    pattern_ati = nx.MultiDiGraph()
    pattern_ati.add_node(0, type=['State', 'Choice', 'Exit', 'FinalState',
                                  'Synchronization', 'Entry'], ids={0, 1})
    pattern_ati.add_node(1, type='Transition')
    pattern_ati.add_edge(0, 1, type='outgoingTransitions')
    pattern_ati.add_edge(0, 1, type='incomingTransitions')
    # useless edges, they must be deleted by the algorithm
    pattern_ati.add_edge(1, 0, type='source')
    pattern_ati.add_edge(1, 0, type='target')

    pattern_at = nx.MultiDiGraph()
    pattern_at.add_node(0, type=['State', 'Choice', 'Exit', 'FinalState',
                                 'Synchronization', 'Entry'], ids={0})
    pattern_at.add_node(1, type='Transition')
    pattern_at.add_node(2, type=['State', 'Choice', 'Exit', 'FinalState',
                                 'Synchronization', 'Entry'], ids={1})
    pattern_at.add_edge(0, 1, type='outgoingTransitions')
    pattern_at.add_edge(2, 1, type='incomingTransitions')
    pattern_at.add_edge(1, 0, type='source')
    pattern_at.add_edge(1, 2, type='target')

    patterns = [pattern_at, pattern_ati]

    return EditOperation(patterns, ids=[0, 1], name='Add Transition Complex')


def get_complex_add_region_with_entry_operation():
    pattern_arwe = nx.MultiDiGraph()
    pattern_arwe.add_node(0, type=['State', 'Statechart'], ids={0})
    pattern_arwe.add_node(1, type='Region')
    pattern_arwe.add_edge(0, 1, type='regions')
    pattern_arwe.add_node(2, type='Entry')
    pattern_arwe.add_edge(1, 2, type='vertices')
    return EditOperation([pattern_arwe], ids=[0], name='Add Region with Entry')
