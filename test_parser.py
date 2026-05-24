import re
content = open('f:/Py_Project/always/cluster/zone/inputs/starlink550_data/step_0000_2026-04-16_11-28-16.txt', encoding='utf-8').read().replace('\r\n', '\n')
m_nodes = re.search(r'\[NODES\]\nID, Name, Lat\(deg\), Lon\(deg\), Alt\(km\)\n(.*?)(?:\n\n|\Z)', content, re.S)

node_lines = m_nodes.group(1).strip().split('\n')
sat_ids, lats, lons = [] ,[], []
name_to_id = {}
for line in node_lines:
    if not line.strip(): continue
    parts = [p.strip() for p in line.split(',')]
    sat_id = int(parts[0])
    sat_name = parts[1]
    sat_ids.append(sat_id)
    lats.append(float(parts[2]))
    lons.append(float(parts[3]))
    name_to_id[sat_name] = sat_id

m_links = re.search(r'\[LINKS\]\nType, SourceName, TargetName, Latency\(ms\)\n(.*?)(?:\n\n|\Z)', content, re.S)
if not m_links:
    print('LINKS headers not found!')
else:
    print('LINKS found!')
graph = []
if m_links:
    link_lines = m_links.group(1).strip().split('\n')
    for line in link_lines:
        if not line.strip(): continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            s_name = parts[1]
            t_name = parts[2]
            if s_name in name_to_id and t_name in name_to_id:
                graph.append([name_to_id[s_name], name_to_id[t_name]])

print('Parsed nodes:', len(sat_ids))
print('Parsed links:', len(graph))
