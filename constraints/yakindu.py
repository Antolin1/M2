def number_inconmig_transitions(G, n):
    count = 0
    for n2 in G[n]:
        for e in G[n][n2]:
            if G[n][n2][e]['type'] == 'incomingTransitions':
                count = count + 1
    return count


def number_outgoing_transitions(G, n):
    count = 0
    for n2 in G[n]:
        for e in G[n][n2]:
            if G[n][n2][e]['type'] == 'outgoingTransitions':
                count = count + 1
    return count


def contains_entry(G, n):
    for n2 in G[n]:
        if G.nodes[n2]['type'] == 'Entry':
            return True
    return False


## No entry in region
def no_entry_region(G):
    for n in G.nodes():
        if G.nodes[n]['type'] == 'Region' and (not contains_entry(G, n)):
            return True
    return False


def entries(G, n):
    e = 0
    for n2 in G[n]:
        if G.nodes[n2]['type'] == 'Entry':
            e = e + 1
    return e


## Multiple entry in region
def multiple_entry_region(G):
    for n in G.nodes():
        if G.nodes[n]['type'] == 'Region' and (entries(G, n) > 1):
            return True
    return False


def to_entry(G, n):
    for n2 in G[n]:
        if G.nodes[n2]['type'] == 'Entry':
            for e in G[n][n2]:
                if G[n][n2][e]['type'] == 'target':
                    return True
    return False


## Incoming to entry
def incoming_to_entry(G):
    for n in G.nodes():
        if G.nodes[n]['type'] == 'Transition' and to_entry(G, n):
            return True
    return False


def entry_out_tran(G):
    for n in G.nodes():
        if G.nodes[n]['type'] == 'Entry' and (number_outgoing_transitions(G, n) != 1):
            return True
    return False


def exit_final(G):
    for n in G.nodes():
        if (G.nodes[n]['type'] == 'Exit' or
            G.nodes[n]['type'] == 'Final') and \
                (number_outgoing_transitions(G, n) > 0):
            return True
    return False


def contains_states(G, n):
    for n2 in G[n]:
        if G.nodes[n2]['type'] == 'State':
            return True
    return False


## No state in region
def no_state_region(G):
    for n in G.nodes():
        if G.nodes[n]['type'] == 'Region' and (not contains_states(G, n)):
            return True
    return False


def choice(G):
    for n in G.nodes():
        if (G.nodes[n]['type'] == 'Choice') and (number_inconmig_transitions(G, n) == 0
                                                 or number_outgoing_transitions(G, n) == 0):
            return True
    return False


def inconsistent(G):
    return (no_entry_region(G) or multiple_entry_region(G)
            or incoming_to_entry(G) or no_state_region(G) or
            choice(G) or exit_final(G) or entry_out_tran(G))
