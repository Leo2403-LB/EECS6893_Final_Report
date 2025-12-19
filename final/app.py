import pickle
import threading
import time
from typing import List, Tuple

from flask import Flask, request, jsonify
import osmnx as ox
import networkx as nx
import requests
from shapely.geometry import LineString
from shapely.strtree import STRtree
from flask_cors import CORS


app = Flask(__name__)
CORS(app)   

# CONFIG


HERE_KEY = "goGlBf5o2ffkcI7O7BdABn0txWhD644EwB8UJUa5haI"  
UPDATE_INTERVAL = 120  # seconds between traffic updates
DEFAULT_SPEED_MPH = 25.0
MATCH_TOL = 0.0005  # roughly ~50m in lat/lon for Manhattan

# LOAD GRAPH


print("Loading Manhattan graph...")
with open("manhattan_graph.gpickle", "rb") as f:
    G = pickle.load(f)
print("Graph loaded with", len(G.nodes), "nodes and", len(G.edges), "edges.")


# EDGE GEOMETRIES + STRtree (PATCHED)


edge_geoms: List[LineString] = []
edge_keys: List[Tuple[int, int, int]] = []

for u, v, k, data in G.edges(keys=True, data=True):
    geom = data.get("geometry")

    # Ensure geometry exists and is valid, with correct (lon, lat) ordering
    if not isinstance(geom, LineString) or len(geom.coords) < 2:
        geom = LineString([
            (G.nodes[u]["x"], G.nodes[u]["y"]),  # (lon, lat)
            (G.nodes[v]["x"], G.nodes[v]["y"])
        ])
        data["geometry"] = geom

    edge_geoms.append(geom)
    edge_keys.append((u, v, k))

# Filter only valid LineStrings 
clean_geoms: List[LineString] = []
clean_keys: List[Tuple[int, int, int]] = []

for geom, key in zip(edge_geoms, edge_keys):
    if isinstance(geom, LineString) and len(geom.coords) > 1:
        clean_geoms.append(geom)
        clean_keys.append(key)
    else:
        print("[Warning] Skipped invalid geometry:", geom)

print("Building spatial index (STRtree) with", len(clean_geoms), "edges...")
edge_tree = STRtree(clean_geoms)
geom_to_key = {id(g): k for g, k in zip(clean_geoms, clean_keys)}
print("Spatial index ready.")

graph_lock = threading.Lock()


# HERE TRAFFIC v7

def get_traffic_flow():
    """
    Fetch HERE Traffic v7 flow data over Manhattan area.
    Using a circle around midtown Manhattan, r=5000m.
    """
    if not HERE_KEY:
        raise ValueError("HERE_KEY is empty. Put your HERE API key in app.py")

    url = "https://data.traffic.hereapi.com/v7/flow"
    params = {
        "locationReferencing": "shape",
        "in": "circle:40.7580,-73.9855;r=5000",  # covers Manhattan
        "apikey": HERE_KEY,
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def parse_here_segments(data):
    """
    Parse HERE v7 flow JSON into list of (LineString, speed_mph).

    Example structure (per result):
    - location.shape.links[].points[] -> {"lat":..,"lng":..}
    - currentFlow.speed -> m/s (we convert to mph)
    """
    segments = []

    for item in data.get("results", []):
        flow = item.get("currentFlow", {})
        speed = flow.get("speed")
        if speed is None:
            continue

        # m/s -> convert to mph
        speed_m_s = float(speed)
        speed_mph = speed_m_s * 2.23694

        links = item["location"]["shape"]["links"]
        for link in links:
            pts = [(p["lat"], p["lng"]) for p in link["points"]]
            if len(pts) < 2:
                continue

            line = LineString(pts)  # (lat, lon) order; fine for distance compare
            segments.append((line, speed_mph))

    return segments
# GRAPH WEIGHTS

def traffic_light_delay(u, v, data):
    """
    Estimate traffic-light delay for Manhattan intersections.
    No real-time signal data available → use industry-average approach.
    """
    # Node degrees = number of streets connected
    deg_u = G.degree[u]
    deg_v = G.degree[v]

    if deg_u <= 2 and deg_v <= 2:
        return 0

    highway_type = data.get("highway", "")

    if isinstance(highway_type, list):
        highway_type = highway_type[0]

    if highway_type in ("primary", "primary_link", "secondary"):
        return 35   # 35s average for avenues (big intersections)
    else:
        return 20   # 20–25s for typical Manhattan street crossings


def reset_default_speeds():
    """Set every edge to default speed (mph)."""
    for _, _, _, data in G.edges(keys=True, data=True):
        data["speed"] = DEFAULT_SPEED_MPH


def recompute_travel_time():
    """Recompute travel_time in seconds for every edge (includes traffic lights)."""
    for u, v, k, data in G.edges(keys=True, data=True):
        speed_mph = data.get("speed", DEFAULT_SPEED_MPH)
        speed_m_s = max(speed_mph * 0.44704, 1.0)  # prevent division by zero
        length_m = data.get("length", 1.0)

        base_time = float(length_m / speed_m_s)

        tl_penalty = traffic_light_delay(u, v, data)

        data["travel_time"] = base_time + tl_penalty



def update_traffic_once():
    """Fetch latest HERE traffic and update the graph weights."""
    print("[Traffic] Fetching HERE traffic data...")
    flow_json = get_traffic_flow()
    segments = parse_here_segments(flow_json)
    print(f"[Traffic] Parsed {len(segments)} segments.")

    with graph_lock:
        reset_default_speeds()

        for shape, speed in segments:
            # candidate edges whose bounding box intersect the segment
            candidates = edge_tree.query(shape)

            for geom in candidates:
                # Safety: ensure geom behaves like a shapely geometry
                if not hasattr(geom, "distance"):
                    print("[Traffic] Skipping non-geometry from STRtree:", type(geom))
                    continue

                if geom.distance(shape) < MATCH_TOL:
                    key = geom_to_key.get(id(geom))
                    if key is None:
                        continue
                    u, v, k = key
                    G[u][v][k]["speed"] = speed

        recompute_travel_time()

    print("[Traffic] Graph speeds and travel_time updated.")


def traffic_loop():
    """Background loop that periodically refreshes traffic data."""
    while True:
        try:
            update_traffic_once()
        except Exception as e:
            print("[Traffic] Update error:", e)
        time.sleep(UPDATE_INTERVAL)


# Initialize default travel_time
with graph_lock:
    reset_default_speeds()
    recompute_travel_time()

# Start background traffic updater
threading.Thread(target=traffic_loop, daemon=True).start()
print("Background traffic update thread started.")

# ROUTING


def compute_fastest_path(start_lat, start_lon, end_lat, end_lon):
    """Return (coords list, total_distance_m, total_time_s)."""
    with graph_lock:
        start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
        end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

        path = nx.shortest_path(G, start_node, end_node, weight="travel_time")

        total_time = 0.0
        total_dist = 0.0

        for u, v in zip(path[:-1], path[1:]):
            data_dict = G.get_edge_data(u, v)
            # For MultiDiGraph, pick edge with smallest travel_time
            best_edge = min(
                data_dict.values(),
                key=lambda d: d.get("travel_time", 1e18),
            )
            total_time += best_edge.get("travel_time", 0.0)
            total_dist += best_edge.get("length", 0.0)

        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]

    return coords, total_dist, total_time


# FLASK APP

@app.route("/")
def home():
    return (
        "Routing backend is running.<br>"
        "Use /route?start=LAT,LON&end=LAT,LON<br>"
        "Example: /route?start=40.7484,-73.9857&end=40.7308,-73.9973"
    )


@app.route("/route")
def route():
    start_raw = request.args.get("start")
    end_raw = request.args.get("end")

    if not start_raw or not end_raw:
        return jsonify({"error": "Please provide start and end parameters"}), 400

    try:
        s_lat, s_lon = map(float, start_raw.split(","))
        e_lat, e_lon = map(float, end_raw.split(","))
    except Exception:
        return jsonify({"error": "Invalid format. Use start=LAT,LON & end=LAT,LON"}), 400

    try:
        coords, dist_m, time_s = compute_fastest_path(s_lat, s_lon, e_lat, e_lon)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(
        {
            "coords": coords,
            "total_distance_m": dist_m,
            "total_time_s": time_s,
        }
    )

@app.route("/traffic")
def traffic():
    """
    Return current edges with their geometry and speed
    so the frontend can color them by congestion level.
    """
    features = []

    with graph_lock:
        for u, v, k, data in G.edges(keys=True, data=True):
            speed = float(data.get("speed", DEFAULT_SPEED_MPH))
            geom = data.get("geometry")

            # Only use valid LineStrings
            if not isinstance(geom, LineString):
                continue

            # Our geom is (lon, lat); Leaflet expects [lat, lon]
            latlon = [(lat, lon) for (lon, lat) in geom.coords]

            features.append({
                "coords": latlon,
                "speed": speed
            })

    return jsonify(features)


if __name__ == "__main__":
    # Local dev server
    app.run(port=8000, debug=True)

