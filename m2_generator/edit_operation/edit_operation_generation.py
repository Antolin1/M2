import networkx as nx
from pyecore.ecore import EReference, EClass
from pyecore.resources import ResourceSet, URI


def get_source_target(reference, classes):
    source = reference.eContainingClass
    target = reference.eType
    targets = [c for c in classes if target in c.eAllSuperTypes() and not c.abstract]
    sources = [c for c in classes if source in c.eAllSuperTypes() and not c.abstract]
    if not target.abstract:
        targets += [target]
    if not source.abstract:
        sources += [source]
    return sources, targets


def generate_atomic_first_type(reference, classes):
    """
    It obtains the atomic patterns associated to the containment reference. One pattern for each different target.
    :param reference: Containment reference
    :param classes: List of meta-classes of the meta-model
    :return: List of patterns
    """
    # TODO: containment to the same meta-class
    sources, targets = get_source_target(reference, classes)
    patterns = []
    for c in targets:
        pattern = nx.MultiDiGraph()
        pattern.add_node(0, type=[s.name for s in sources], ids={0})
        pattern.add_node(1, type=c.name)
        pattern.add_edge(0, 1, type=reference.name)
        patterns.append(pattern)
        print(f'Add first type {c.name}')
    return patterns


def generate_atomic_second_type(reference, classes):
    """
    It obtains the atomic patterns associated to the non-containment references.
    One or two patterns depending on the intersection.
    :param reference: Containment reference
    :param classes: List of meta-classes of the meta-model
    :return: List of patterns
    """
    print(f'Add second type {reference.name}')
    sources, targets = get_source_target(reference, classes)
    patterns = []
    pattern = nx.MultiDiGraph()
    pattern.add_node(0, type=[s.name for s in sources], ids={0})
    pattern.add_node(1, type=[s.name for s in targets], ids={1})
    pattern.add_edge(0, 1, type=reference.name)
    patterns.append(pattern)
    intersection = [x for x in sources if x in targets]
    if intersection:
        pattern_it = nx.MultiDiGraph()
        pattern_it.add_node(0, type=[s.name for s in intersection], ids={0, 1})
        pattern_it.add_edge(0, 1, type=reference.name)
        pattern.append(pattern_it)
    return patterns


def get_edit_operations(path_metamodel):
    rset = ResourceSet()
    # load model
    resource = rset.get_resource(URI(path_metamodel))
    # get list of all elements of the model
    list_elements = []
    for root in resource.contents:
        list_elements.append(root)
        list_elements = list_elements + list(root.eAllContents())

    ereferences = [e for e in list_elements if isinstance(e, EReference)]
    classes = [c for c in list_elements if isinstance(c, EClass)]

    visited = []
    all_patterns = []
    for r in ereferences:
        if r.containment:
            patterns = generate_atomic_first_type(r, classes)
            all_patterns.append(patterns)
            print('First type', len(patterns))
        elif r.eOpposite and not r.eOpposite.containment and r.eOpposite.name not in visited:
            patterns = generate_atomic_second_type(r, classes)
            all_patterns.append(patterns)
            print('Second type', len(patterns))
        elif not r.eOpposite:
            patterns = generate_atomic_second_type(r, classes)
            all_patterns.append(patterns)
            print('Second type', len(patterns))
        visited.append(r.name)
    return all_patterns
