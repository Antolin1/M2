import networkx as nx
from pyecore.ecore import EReference, EAttribute
from pyecore.resources import ResourceSet, URI


def get_graph_from_model(path_model, path_metamodels, metafilter=None,
                         consider_atts=True):
    rset = ResourceSet()
    for pathMetamodel in path_metamodels:
        # load meta-model
        resource = rset.get_resource(URI(pathMetamodel))
        mm_root = resource.contents[0]
        rset.metamodel_registry[mm_root.nsURI] = mm_root
        # print(mm_root.nsURI, mm_root)

    # load model
    resource = rset.get_resource(URI(path_model))
    # get list of all elements of the model
    list_elements = []
    for root in resource.contents:
        list_elements.append(root)
        list_elements = list_elements + list(root.eAllContents())

    return get_graph_from_model_elements(list_elements, metafilter=metafilter,
                                         consider_atts=consider_atts)


def get_graph_from_model_elements(list_elements, metafilter=None,
                                  consider_atts=True):
    # obtain graph
    nodes = {}
    i = 0
    G = nx.MultiDiGraph()
    for o in list_elements:
        if (metafilter is not None) and (not metafilter.pass_filter_object(o)):
            continue
        # Register node
        if not o in nodes:
            nodes[o] = i
            i = i + 1
            G.add_node(nodes[o], type=o.eClass.name)
        dic_attributes = {}
        for f in o.eClass.eAllStructuralFeatures():
            if (f.derived):
                continue
            if (metafilter is not None) and (not metafilter.pass_filter_structural(f)):
                continue
            # references
            if isinstance(f, EReference):
                if f.many:
                    for o2 in o.eGet(f):
                        if o2 is None:  # or o2.eIsProxy
                            continue
                        # avoid adding elements thar are not in the model
                        if not o2 in list_elements:
                            continue
                        if ((metafilter is not None) and
                                (not metafilter.pass_filter_object(o2))):
                            continue
                        if not o2 in nodes:
                            nodes[o2] = i
                            i = i + 1
                            G.add_node(nodes[o2], type=o2.eClass.name)
                        G.add_edge(nodes[o], nodes[o2], type=f.name)
                else:
                    o2 = o.eGet(f)
                    if o2 is None:  # or o2.eIsProxy
                        continue
                    # avoid adding elements thar are not in the model
                    if not o2 in list_elements:
                        continue
                    if ((metafilter is not None) and
                            (not metafilter.pass_filter_object(o2))):
                        continue
                    if not o2 in nodes:
                        nodes[o2] = i
                        i = i + 1
                        G.add_node(nodes[o2], type=o2.eClass.name)
                    G.add_edge(nodes[o], nodes[o2], type=f.name)
            # attributes
            elif isinstance(f, EAttribute):
                if f.many:
                    list_att_val = []
                    for o2 in o.eGet(f):
                        if o2 is None:  # or o2.eIsProxy
                            list_att_val.append('<none>')
                        else:
                            list_att_val.append(o2)
                    dic_attributes[f.name] = list_att_val
                else:
                    o2 = o.eGet(f)
                    if o2 == None:  # or o2.eIsProxy
                        dic_attributes[f.name] = '<none>'
                    else:
                        dic_attributes[f.name] = o2
        if consider_atts:
            G.nodes[nodes[o]]['atts'] = dic_attributes
    return G
