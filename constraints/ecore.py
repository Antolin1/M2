import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from m2_generator.edit_operation.edit_operation import edge_match


def node_match(n1, n2):
    return n1['type'] == n2['type']


def has_cycles(G):
    eclasses = []
    for n in G:
        if G.nodes[n]['type'] == 'EClass':
            eclasses.append(n)

    remove = []
    G_eclasses = G.subgraph(eclasses).copy()
    for e in G_eclasses.edges:
        if G[e[0]][e[1]][e[2]]['type'] != 'eSuperTypes':
            remove.append((e[0], e[1], e[2]))
    for s, t, k in remove:
        G_eclasses.remove_edge(s, t, k)

    try:
        nx.find_cycle(G_eclasses, orientation="original")
    except nx.exception.NetworkXNoCycle:
        return False
    return True


def does_not_have_type(G, n):
    for n2 in G[n]:
        for e in G[n][n2]:
            if G[n][n2][e]['type'] == 'eType':
                return False
    return True


def wrong_attribute_type(G, n):
    for n2 in G[n]:
        for e in G[n][n2]:
            if G[n][n2][e]['type'] == 'eType' and G.nodes[n2]['type'] == 'EClass':
                return True
    return False


def wrong_ereference_type(G, n):
    for n2 in G[n]:
        for e in G[n][n2]:
            if G[n][n2][e]['type'] == 'eType' and G.nodes[n2]['type'] != 'EClass':
                return True
    return False


def reference_does_not_have_type(G):
    for n in G:
        if G.nodes[n]['type'] == 'EReference' and (does_not_have_type(G, n) or
                                                   wrong_ereference_type(G, n)):
            return True
    return False


def attribute_does_not_have_type(G):
    for n in G:
        if G.nodes[n]['type'] == 'EAttribute' and (does_not_have_type(G, n) or
                                                   wrong_attribute_type(G, n)):
            return True
    return False


# def oppositeOfItself(G):
oppItself = nx.MultiDiGraph()
oppItself.add_node(0, type='EReference')
oppItself.add_edge(0, 0, type='eOpposite')


def oposite_of_itself(G):
    GM = GraphMatcher(G, oppItself, node_match=node_match,
                      edge_match=edge_match)
    for subgraph in GM.subgraph_isomorphisms_iter():
        return True
    return False


## end opposites oppositeRestriction, oppositeRestrictionSameClasses
def are_they_opposite(G, n1, n2):
    try:
        for e in G[n1][n2]:
            if G[n1][n2][e]['type'] == 'eOpposite':
                return True
        return False
    except:
        return False


def restriction_opposite(G):
    for n1 in G:
        for n2 in G:
            if (n1 != n2 and G.nodes[n1]['type'] == 'EReference'
                    and G.nodes[n2]['type'] == 'EReference' and are_they_opposite(G, n1, n2)):
                if not are_they_opposite(G, n2, n1):
                    return True
    return False


def getType(G, n):
    for n2 in G[n]:
        for e in G[n][n2]:
            if G[n][n2][e]['type'] == 'eType':
                return n2


def get_containing(G, n):
    for n2 in G[n]:
        for e in G[n][n2]:
            if G[n][n2][e]['type'] == 'eContainingClass':
                return n2


def restriction_same_classes(G):
    for n1 in G:
        for n2 in G:
            if (n1 != n2 and G.nodes[n1]['type'] == 'EReference'
                    and G.nodes[n2]['type'] == 'EReference' and are_they_opposite(G, n1, n2)
                    and are_they_opposite(G, n2, n1)):
                if getType(G, n1) != get_containing(G, n2):
                    return True
    return False


def inconsistent(G):
    return (has_cycles(G) or
            reference_does_not_have_type(G) or
            attribute_does_not_have_type(G) or
            oposite_of_itself(G) or restriction_opposite(G)
            or restriction_same_classes(G))
