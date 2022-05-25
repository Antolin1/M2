import random
from itertools import combinations

import networkx as nx
import numpy as np
from multiset import Multiset
from networkx.algorithms.isomorphism import DiGraphMatcher


def edge_match(e1, e2):
    t1 = []
    t2 = []
    for e in e1:
        t1.append(e1[e]['type'])
    for e in e2:
        t2.append(e2[e]['type'])
    return Multiset(t1) == Multiset(t2)


def node_match(n1, n2):
    if ('ids' in n2):
        return n1['type'] in n2['type']
    else:
        return ((n1['type'] == n2['type']) and
                n1['out'] == n2['out'] and
                n1['in'] == n2['in'])


def relabel_randomly(G):
    new_map = {}
    j = 0
    new_ids = list(range(0, len(G)))
    random.shuffle(new_ids)
    for n in G:
        new_map[n] = new_ids[j]
        j = j + 1
    return nx.relabel_nodes(G, new_map)


# add in/out degree for each node. Useful to compute the isomorfism
def add_in_out(G):
    for n in G:
        G.nodes[n]['in'] = G.in_degree(n)
        G.nodes[n]['out'] = G.out_degree(n)
    return G


# given a pattern of the edit operation and a graph. It returns a new graph
# that corresponds to the action of removing an edit operation.
# this graph has nodes with 'ids' indicating the border nodes.
def remove_edit_pattern(pat, G):
    new_G = add_in_out(nx.MultiDiGraph(G))
    pat_en = add_in_out(nx.MultiDiGraph(pat))

    GM = DiGraphMatcher(new_G, pat_en, node_match=node_match,
                        edge_match=edge_match)
    dics = []
    for subgraph in GM.subgraph_isomorphisms_iter():
        dics.append(subgraph)
        # break
    ##No match, return none
    if len(dics) == 0:
        return None
    chosen = random.sample(dics, 1)[0]

    ## remove not border nodes
    border_nodes = []
    for n1, n2 in chosen.items():
        if not ('ids' in pat_en.nodes[n2]):
            new_G.remove_node(n1)
        else:
            border_nodes.append(n1)

    remove_edges = []
    for b1 in border_nodes:
        for b2 in border_nodes:
            c1 = chosen[b1]
            c2 = chosen[b2]
            if (b2 in new_G[b1]) and (c2 in pat_en[c1]):
                list_type_edges = [pat_en[c1][c2][ee]['type']
                                   for ee in pat_en[c1][c2]]
                for e in new_G[b1][b2]:
                    ty = new_G[b1][b2][e]['type']
                    if ty in list_type_edges:
                        remove_edges.append((b1, b2, e))
                        list_type_edges.remove(ty)

    for a, b, e in remove_edges:
        new_G.remove_edge(a, b, e)
    # remove useless atts
    for n in new_G:
        del new_G.nodes[n]['in']
        del new_G.nodes[n]['out']

    for b in border_nodes:
        new_G.nodes[b]['ids'] = pat_en.nodes[chosen[b]]['ids']

    return new_G


class EditOperation:
    def __init__(self, patterns, ids, name=''):
        self.patterns = patterns
        self.ids = ids
        self.name = name
        # TODO: consistency of the patterns

    def can_apply(self, G) -> bool:
        return self.select_pattern(G) is not None

    # there should be nodes with attribute 'ids' in G
    # return the result of the edit. It removes the 'ids' attrs.
    # apply the edit operation if it is possible (if not, None is returned).
    # The nodes of the resultant graph does not contain 'ids'
    # the nodes are relabed to 0,1,...,len(g)
    def apply_edit(self, G):
        pat = self.select_pattern(G)
        if pat is None:
            return None
        all_ids = [n for n in G.nodes()] + [n for n in pat.nodes()]
        max_id_add = np.max(all_ids) + 1

        map_G = {}
        map_edit = {}

        for n in G:
            if 'ids' in G.nodes[n]:
                key = G.nodes[n]['ids']
                map_G[n] = max_id_add
                for m in pat:
                    if 'ids' in pat.nodes[m] and pat.nodes[m]['ids'] == key:
                        map_edit[m] = max_id_add
                        break
                max_id_add = max_id_add + 1

        for m in pat:
            if not 'ids' in pat.nodes[m]:
                map_edit[m] = max_id_add
                max_id_add = max_id_add + 1

        new_pat = nx.MultiDiGraph(pat)
        new_G = nx.MultiDiGraph(G)

        new_pat = nx.relabel_nodes(new_pat, map_edit)
        new_G = nx.relabel_nodes(new_G, map_G)

        G_compose = nx.compose(new_pat, new_G)
        # fix ids
        new_map = {}
        j = 0
        for n in G_compose:
            new_map[n] = j
            j = j + 1
            if ('ids' in G_compose.nodes[n]):
                del G_compose.nodes[n]['ids']

        G_compose_final = nx.relabel_nodes(G_compose, new_map)
        return G_compose_final

    # there should be nodes with attribute 'ids' in G
    # select one of the possible patters
    # of the editOperation that can be applied
    # return none if no one can be applied
    def select_pattern(self, G):
        dic_spe_nodes_G = {}
        for n in G.nodes():
            if 'ids' in G.nodes[n]:
                for idd in G.nodes[n]['ids']:
                    dic_spe_nodes_G[idd] = n

        dic_patterns = self.get_dict_nodes()
        for j, dp in enumerate(dic_patterns):
            # check keys
            if set(dp.keys()) != set(dic_spe_nodes_G.keys()):
                continue

            valid = True
            # check combinations
            for k1, k2 in combinations(dp.keys(), 2):
                if (dp[k1] == dp[k2]) != (dic_spe_nodes_G[k1]
                                          == dic_spe_nodes_G[k2]):
                    valid = False
                    break
            if not valid:
                continue
            # check types
            valid = True
            for k, v in dic_spe_nodes_G.items():
                node_G = G.nodes[v]
                node_p = self.patterns[j].nodes[dp[k]]
                if not node_G['type'] in node_p['type']:
                    valid = False
            if valid:
                return self.patterns[j]
        return None

    # return a list of dics. In each dic, the key is the 'ids'
    # and the value is the node that has this id.
    def get_dict_nodes(self):
        result = []
        for G in self.patterns:
            dic_spe = {}
            for n in G.nodes():
                if 'ids' in G.nodes[n]:
                    for idd in G.nodes[n]['ids']:
                        dic_spe[idd] = n
            result.append(dic_spe)
        return result

    # given a graph it returns a new graph and the pattern used (that belongs
    # to the editoperation object). The graph
    # corresponds to the action of removing an edit operation.
    # this graph has nodes with 'ids' indicating the border nodes.
    # the nodes are relabed to 0,1,...,len(g)
    def remove_edit(self, G):
        pats = self.patterns.copy()
        random.shuffle(pats)
        for p in pats:
            re = remove_edit_pattern(p, G)
            if re is not None:
                new_map = {}
                j = 0
                for n in re:
                    new_map[n] = j
                    j = j + 1
                re = nx.relabel_nodes(re, new_map)
                return re, p
        return None
