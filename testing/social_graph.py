import json
import networkx as nx

with open("processed_meta.json", encoding="utf-8") as f:
    posts = json.load(f)

G = nx.Graph()
if posts:
    user = posts[0].get("ownerUsername") or "user"
    G.add_node(user)
    for post in posts:
        for m in post.get("mentions", []):
            G.add_node(m)
            G.add_edge(user, m)

nx.write_graphml(G, "social_graph.graphml")
print("Граф упоминаний сохранён в social_graph.graphml")
