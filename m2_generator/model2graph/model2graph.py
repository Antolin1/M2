import networkx as nx
from pyecore.ecore import EReference, EAttribute, EClass
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
            if f.derived:
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
                    if o2 is None:  # or o2.eIsProxy
                        dic_attributes[f.name] = '<none>'
                    else:
                        dic_attributes[f.name] = o2
        if consider_atts:
            G.nodes[nodes[o]]['atts'] = dic_attributes
    return G


def get_model_from_graph(path_metamodels, G):
    # Register metamodel
    rset = ResourceSet()
    list_elements = []
    for path_metamodel in path_metamodels:
        # load meta-model
        resource = rset.get_resource(URI(path_metamodel))
        mm_root = resource.contents[0]
        rset.metamodel_registry[mm_root.nsURI] = mm_root
        for root in resource.contents:
            list_elements.append(root)
            list_elements = list_elements + list(root.eAllContents())

    # get eclasses and references
    name_correspondence = {}
    name_ereferences = {}
    name_attributes = {}
    for e in list_elements:
        if isinstance(e, EClass):
            name_correspondence[e.name] = e
        if isinstance(e, EReference):
            name_ereferences[e.name] = e
        if isinstance(e, EAttribute):
            name_attributes[e.name] = e

    # graph to model

    nodes_objects = {}
    for n in G:
        eobj = None
        if n not in nodes_objects:
            type = G.nodes[n]['type']
            if type not in name_correspondence:
                continue
            cal = name_correspondence[type]
            eobj = cal()
            nodes_objects[n] = eobj
        else:
            eobj = nodes_objects[n]
        # TODO: the <none> token
        # if 'atts' in G.nodes[n]:
        #    atts = G.nodes[n]['atts']
        #    for att_name, value in atts.items():
        #        if name_attributes[att_name].many:
        #            for v in value:
        #                getattr(eobj, att_name).add(v)
        #        else:
        #            setattr(eobj, att_name, value)

        # references
        for n2 in G[n]:
            eobj2 = None
            if not n2 in nodes_objects:
                type2 = G.nodes[n2]['type']
                if type2 not in name_correspondence:
                    continue
                cal2 = name_correspondence[type2]
                eobj2 = cal2()
                nodes_objects[n2] = eobj2
            else:
                eobj2 = nodes_objects[n2]
            for e in G[n][n2]:
                type_edge = G[n][n2][e]['type']
                if name_ereferences[type_edge].many:
                    getattr(eobj, type_edge).add(eobj2)
                else:
                    setattr(eobj, type_edge, eobj2)
    return nodes_objects


def serialize_graph_model(path, path_metamodels, main_class, G):
    nodes_objects = get_model_from_graph(path_metamodels, G)
    for n, ob in nodes_objects.items():
        if ob.eClass.name == main_class:
            rset = ResourceSet()
            resource = rset.create_resource(URI(path))  # This will create an XMI resource
            resource.append(ob)  # we add the EPackage instance in the resource
            resource.save()  # we then serialize it
