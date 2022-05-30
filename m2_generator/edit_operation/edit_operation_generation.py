import networkx as nx
from pyecore.ecore import EReference, EClass
from pyecore.resources import ResourceSet, URI

from m2_generator.edit_operation.edit_operation import EditOperation


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
    It obtains the atomic edit operations of the first type.
    :param reference: Containment reference
    :param classes: List of meta-classes of the meta-model
    :return: List of edit operations
    """
    # TODO: containment to the same meta-class
    sources, targets = get_source_target(reference, classes)
    edit_operations = []
    for c in targets:
        pattern = nx.MultiDiGraph()
        pattern.add_node(0, type=[s.name for s in sources], ids={0})
        pattern.add_node(1, type=c.name)
        pattern.add_edge(0, 1, type=reference.name)
        edit_operation = EditOperation([pattern], ids=[0], name=f'Add {c.name}')
        edit_operations.append(edit_operation)
    return edit_operations


def generate_atomic_second_type(reference, classes):
    """
    It obtains the atomic edit operation of the first type.
    :param reference: Containment reference
    :param classes: List of meta-classes of the meta-model
    :return: an edit operation
    """
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
        patterns.append(pattern_it)
    return [EditOperation(patterns, ids=[0, 1], name=f'Add {reference.name}', type='second')]


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
    edit_operations = []
    for r in ereferences:
        edit_ops = []
        if r.containment:
            edit_ops = generate_atomic_first_type(r, classes)
        elif r.eOpposite and not r.eOpposite.containment and r.eOpposite.name not in visited:
            edit_ops = generate_atomic_second_type(r, classes)
        elif not r.eOpposite:
            edit_ops = generate_atomic_second_type(r, classes)
        visited.append(r.name)
        edit_operations += edit_ops
    return edit_operations
