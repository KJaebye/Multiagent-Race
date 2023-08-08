from config.config import cfg
from copy import deepcopy


from lxml import etree
import utils.xml as xu


def merge_multiagent_with_base(multiagent, ispath=True):
    base_xml = cfg.TEMPLATE
    root_b, tree_b = xu.etree_from_xml(base_xml)
    root_a, tree_a = xu.etree_from_xml(multiagent, ispath=ispath)

    worldbody = root_b.findall("./worldbody")[0]
    actuator_b = root_b.findall("./actuator")[0]

    for idx in range(cfg.EMAT.AGENT_NUM):
        prefix = 'agent' + str(idx) + '_'
        agent_body = xu.find_elem(root_a, "body", "name", prefix + "torso")[0]

        # Update agent z pos based on starting terrain
        if cfg.ENV.DIMENSION == 2:
            pos = xu.str2arr(agent_body.get("pos"))
            pos[1] += idx
            agent_body.set("pos", xu.arr2str(pos))
        else:
            pos = xu.str2arr(agent_body.get("pos"))
            agent_body.set("pos", xu.arr2str(pos))
        worldbody.append(agent_body)

        actuator_a = root_a.findall("./actuator")[idx]
        agent_motors = xu.find_elem(actuator_a, "motor")
        actuator_b.extend(agent_motors)

    return xu.etree_to_str(root_b)

def create_agent_xml(root, idx):
    prefix = 'agent' + str(idx) + '_'
    root = rename_attributes(root, prefix)

    agent = xu.find_elem(root, "body", "name", prefix + "torso")[0]
    actuator = root.findall("./actuator")[0]

    return agent, actuator

def rename_attributes(root, prefix):
    """ Recursively add the prefix. """
    for elem in root.iter():
        for k, v in sorted(elem.items()):
            if k == 'name' or k == 'site' or k == 'joint' or k == 'body':
                elem.set(k, prefix + elem.get(k))
    return root

def modify_xml_attributes(xml):
    root, tree = xu.etree_from_xml(xml, ispath=False)

    # # Enable/disable filterparent
    # flag = xu.find_elem(root, "flag")[0]
    # flag.set("filterparent", str(cfg.XML.FILTER_PARENT))

    # # Modify default geom params
    # default_geom = xu.find_elem(root, "geom")[0]
    # default_geom.set("condim", str(cfg.XML.GEOM_CONDIM))
    # default_geom.set("friction", xu.arr2str(cfg.XML.GEOM_FRICTION))

    # visual = xu.find_elem(root, "visual")[0]
    # map_ = xu.find_elem(visual, "map")[0]
    # map_.set("shadowclip", str(cfg.XML.SHADOWCLIP))

    return xu.etree_to_str(root)

def create_multiagent_xml(xml_path):
    ###### Create and combine multiple agent xml ########################        
    agent_list = []
    actuator_list = []
    root, _ = xu.etree_from_xml(xml_path)
    for idx in range(cfg.EMAT.AGENT_NUM):
        root_ = deepcopy(root)
        agent, actuator = create_agent_xml(root_, idx)
        agent_list.append(agent)
        actuator_list.append(actuator)
        
    multiagent = etree.Element("multi_agent", {"model": "multi agent"})
    for agent in agent_list:
        multiagent.append(agent)
    for actuator in actuator_list:
        multiagent.append(actuator)
        
    x = xu.etree_to_str(multiagent)

    #### Merge multiagent xml to base ###################################
    xml = merge_multiagent_with_base(x, False)

    #### Add xml attributes ###################################
    xml = modify_xml_attributes(xml)

    return xml
