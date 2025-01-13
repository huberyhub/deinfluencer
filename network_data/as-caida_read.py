import networkx as nx
import pickle
import os

# Directed graph: as-caida20071105.txt
# The CAIDA AS Relationships Dataset, from 11 05 2007
# Relationships:    -1 (<FromNodeId> is a customer of <ToNodeId>)
#             1 (<FromNodeId> is a provider of <ToNodeId>)
#             0 (<FromNodeId> and <ToNodeId> are peers)
#             2 (<FromNodeId> and <ToNodeId> are siblings (the same organization).)
# Nodes:26475    Edges: 106762
# FromNodeId        ToNodeId    Relationship
# Load the data from the file

file_path = '/Users/huberyhu/Desktop/SURP/im_deinfluence/network_data/as-caida.txt'
save_path = '/Users/huberyhu/Desktop/SURP/im_deinfluence/network_data/as-caida.gpickle'

# Create an empty directed graph
G = nx.DiGraph()

# Open the file and add edges to the Graph
with open(file_path, 'r') as file:
    for line in file:
        node1, node2, *_ = map(int, line.split())
        G.add_edge(node1, node2)

# Ensure the directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the graph to a file
try:
    with open(save_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved to {save_path}")
except Exception as e:
    print(f"An error occurred while saving the graph: {e}")

# Display basic information about the graph
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
