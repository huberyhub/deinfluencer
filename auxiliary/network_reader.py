import networkx as nx
import pickle
import os

# Load the data from the file
# Use facebook_combined.txt as the input file for example

file_path = '' # Path to the input file

save_path = '' # Path to save the graph

# Create an empty directed graph
G = nx.DiGraph()

# Open the file and add edges to the Graph
with open(file_path, 'r') as file:
    for line in file:
        node1, node2 = map(int, line.split())
        G.add_edge(node1, node2)

# Ensure the directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the graph to a file
# Save the graph to a file using pickle
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