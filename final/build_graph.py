import osmnx as ox
import pickle

print("Downloading Manhattan graph...")

# Download drivable road network of Manhattan
G = ox.graph_from_place(
    "Manhattan, New York City, USA",
    network_type="drive"
)

# Add edge lengths so we know distances
G = ox.distance.add_edge_lengths(G)

# Save as pickle
with open("manhattan_graph.gpickle", "wb") as f:
    pickle.dump(G, f)

print("Graph saved as manhattan_graph.gpickle")
