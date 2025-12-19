import pickle
import threading
import time
from typing import List, Tuple
from datetime import datetime

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

HERE_KEY = ""#TODO: input here api key
UPDATE_INTERVAL = 120  # seconds between traffic updates
DEFAULT_SPEED_MPH = 25.0
MATCH_TOL = 0.0005  # roughly ~50m in lat/lon for Manhattan

# LOAD GRAPH


print("Loading Manhattan graph...")
with open("manhattan_graph.gpickle", "rb") as f:
    G = pickle.load(f)
print("Graph loaded with", len(G.nodes), "nodes and", len(G.edges), "edges.")

# EDGE GEOMETRIES + STRtree


edge_geoms: List[LineString] = []
edge_keys: List[Tuple[int, int, int]] = []

for u, v, k, data in G.edges(keys=True, data=True):
    geom = data.get("geometry")

    # Ensure geometry exists and is valid, with correct (lon, lat) ordering
    if not isinstance(geom, LineString) or len(geom.coords) < 2:
        geom = LineString(
            [
                (G.nodes[u]["x"], G.nodes[u]["y"]),  # (lon, lat)
                (G.nodes[v]["x"], G.nodes[v]["y"]),
            ]
        )
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

    url = "https://data.traffic.hereapi.com/v7/flow"
    params = {
        "locationReferencing": "shape",
        "in": "circle:40.7580,-73.9855;r=10000",  # covers Manhattan
        "apikey": HERE_KEY,
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def parse_here_segments(data):
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
    # Node degrees = number of streets connected
    deg_u = G.degree[u]
    deg_v = G.degree[v]

    if deg_u <= 2 and deg_v <= 2:
        return 0

    highway_type = data.get("highway", "")
    if isinstance(highway_type, list):
        highway_type = highway_type[0]

    if highway_type in ("primary", "primary_link", "secondary"):
        return 35  # 35s average for avenues (big intersections)
    else:
        return 20  # 20–25s for typical Manhattan street crossings


def reset_default_speeds():
    """Set every edge to default speed (mph)."""
    for _, _, _, data in G.edges(keys=True, data=True):
        data["speed"] = DEFAULT_SPEED_MPH


def recompute_travel_time():
    """Recompute travel_time in seconds for every edge."""
    for u, v, k, data in G.edges(keys=True, data=True):
        speed_mph = data.get("speed", DEFAULT_SPEED_MPH)
        speed_m_s = max(speed_mph * 0.44704, 1.0)  # prevent division by zero
        length_m = data.get("length", 1.0)

        base_time = float(length_m / speed_m_s)
        tl_penalty = traffic_light_delay(u, v, data)

        data["travel_time"] = base_time + tl_penalty


def update_traffic_once():
    """Fetch latest HERE traffic and update the graph weights."""
    print("Fetching HERE traffic data...")
    flow_json = get_traffic_flow()
    segments = parse_here_segments(flow_json)
    print(f"Parsed {len(segments)} segments.")

    with graph_lock:
        reset_default_speeds()

        for shape, speed in segments:
            # candidate edges whose bounding box intersect the segment
            candidates = edge_tree.query(shape)

            for geom in candidates:
                # ensure geom behaves like a shapely geometry
                if not hasattr(geom, "distance"):
                    print("Skipping non-geometry from STRtree:", type(geom))
                    continue

                if geom.distance(shape) < MATCH_TOL:
                    key = geom_to_key.get(id(geom))
                    if key is None:
                        continue
                    u, v, k = key
                    G[u][v][k]["speed"] = speed

        recompute_travel_time()

    print("Graph speeds and travel_time updated.")


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

# ML PREDICTION STUB

def ml_predict_future_graph(
    G_in: nx.MultiDiGraph,
    current_time: datetime,
    minutes_ahead: int = 30
) -> nx.MultiDiGraph:
    if not hasattr(ml_predict_future_graph, "_loaded"):
        import pandas as pd
        import joblib

        bundle = joblib.load("rf_speed_model_light.joblib")
        ml_predict_future_graph._rf_model = bundle["model"]
        ml_predict_future_graph._W = int(bundle["W"])
        ml_predict_future_graph._H = int(bundle["H"])

        state = pd.read_parquet("rf_last_window.parquet")
        state["item_id"] = state["item_id"].astype(str)
        ml_predict_future_graph._rf_state = state.set_index("item_id")

        map_df = pd.read_parquet("here_segment_to_osm_edge_filtered.parquet")
        map_df["segment_id"] = map_df["segment_id"].astype(str)
        ml_predict_future_graph._segment_to_edge = (
            map_df.set_index("segment_id")[["u", "v", "key"]].to_dict("index")
        )

        ml_predict_future_graph._loaded = True

    rf_model = ml_predict_future_graph._rf_model
    W = ml_predict_future_graph._W
    H = ml_predict_future_graph._H
    rf_state = ml_predict_future_graph._rf_state
    segment_to_edge = ml_predict_future_graph._segment_to_edge

    G_pred = G_in.copy()
    n_steps = int(round(float(minutes_ahead)))
    if n_steps < 1:
        n_steps = 1

    import numpy as np

    for seg_id, info in segment_to_edge.items():
        u, v, k = info["u"], info["v"], info["key"]

        if not G_pred.has_edge(u, v, k):
            continue

        data = G_pred[u][v][k]

        cur_speed_mph = data.get("speed", DEFAULT_SPEED_MPH)

        cur_speed_kmh = float(cur_speed_mph) * 1.609344
        if not np.isfinite(cur_speed_kmh) or cur_speed_kmh <= 0:
            cur_speed_kmh = 40.0 

        if seg_id in rf_state.index:
            try:
                window = rf_state.loc[seg_id, [f"lag_{i+1}" for i in range(W)]].to_numpy(dtype=np.float32)
                if window.shape[0] != W or np.any(~np.isfinite(window)):
                    raise ValueError("bad window")
                window[-1] = cur_speed_kmh
            except Exception:
                window = np.full((W,), cur_speed_kmh, dtype=np.float32)
        else:
            window = np.full((W,), cur_speed_kmh, dtype=np.float32)

        remaining = n_steps
        last_pred_kmh = cur_speed_kmh

        while remaining > 0:
            y = rf_model.predict(window.reshape(1, -1))[0]  # shape: (H,)
            y = np.asarray(y, dtype=np.float32)
            if y.ndim != 1 or y.shape[0] < 1:
                break

            take = min(H, remaining)
            for j in range(take):
                pred_kmh = float(y[j])
                if not np.isfinite(pred_kmh) or pred_kmh <= 0:
                    pred_kmh = last_pred_kmh 
                last_pred_kmh = pred_kmh
                window = np.concatenate([window[1:], np.array([pred_kmh], dtype=np.float32)])

            remaining -= take

        pred_mph = max(last_pred_kmh * 0.621371, 1.0)  # km/h -> mph

        data["speed"] = float(pred_mph)

        length_m = data.get("length", 1.0)
        speed_m_s = max(pred_mph * 0.44704, 1.0)
        base_time = float(length_m / speed_m_s)
        tl_penalty = traffic_light_delay(u, v, data)
        data["travel_time"] = base_time + tl_penalty

    return G_pred


# ROUTING



def compute_fastest_path(start_lat, start_lon, end_lat, end_lon):
    """Return (coords list, total_distance_m, total_time_s) using current real-time graph G."""
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


def compute_hybrid_fastest_path(start_lat, start_lon, end_lat, end_lon):
    """
    - Use real-time traffic (G) for the first 30 minutes of travel.
    - Use ML-predicted traffic (G_pred) for the remaining part of the trip.
    """

    cutoff_time = 30 * 60  # 30 minutes in seconds

    with graph_lock:
        start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
        end_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

        full_path = nx.shortest_path(G, start_node, end_node, weight="travel_time")

        # Walk through full_path to accumulate time and distance
        prefix_time = 0.0
        prefix_dist = 0.0
        cutoff_index = None  # index in full_path where we cross 30 min

        for i, (u, v) in enumerate(zip(full_path[:-1], full_path[1:])):
            data_dict = G.get_edge_data(u, v)
            best_edge = min(
                data_dict.values(),
                key=lambda d: d.get("travel_time", 1e18),
            )
            dt = best_edge.get("travel_time", 0.0)
            dl = best_edge.get("length", 0.0)

            prefix_time += dt
            prefix_dist += dl

            if prefix_time >= cutoff_time:
                cutoff_index = i + 1  # node index in full_path
                break

        # If the whole trip is within 30 minutes, just return the live result
        if cutoff_index is None:
            coords_live = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in full_path]
            total_time = prefix_time
            total_dist = prefix_dist
            routing_mode = "live_only"
            return coords_live, total_dist, total_time, routing_mode

        
        prefix_nodes = full_path[:cutoff_index + 1]  # include cutoff node
        mid_node = full_path[cutoff_index]

        # Coordinates for live prefix
        live_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in prefix_nodes]

    current_time = datetime.utcnow()
    G_pred = ml_predict_future_graph(G, current_time, minutes_ahead=30)

    
    # Note: we find end_node in G_pred again, based on end_lat, end_lon
    end_node_pred = ox.distance.nearest_nodes(G_pred, end_lon, end_lat)
    future_path = nx.shortest_path(G_pred, mid_node, end_node_pred, weight="travel_time")

    # accumulate future time and distance on G_pred 
    future_time = 0.0
    future_dist = 0.0

    for u, v in zip(future_path[:-1], future_path[1:]):
        data_dict = G_pred.get_edge_data(u, v)
        best_edge = min(
            data_dict.values(),
            key=lambda d: d.get("travel_time", 1e18),
        )
        future_time += best_edge.get("travel_time", 0.0)
        future_dist += best_edge.get("length", 0.0)

    # Coordinates for predicted suffix 
    future_coords = [(G_pred.nodes[n]["y"], G_pred.nodes[n]["x"]) for n in future_path[1:]]

    #merge live prefix + predicted suffix
    merged_coords = live_coords + future_coords
    total_time = prefix_time + future_time
    total_dist = prefix_dist + future_dist
    routing_mode = "hybrid_30min"

    return merged_coords, total_dist, total_time, routing_mode

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
        return jsonify(
            {"error": "Invalid format. Use start=LAT,LON & end=LAT,LON"}
        ), 400

    try:
        # Use hybrid routing: first 30 min real-time, then predicted
        coords, dist_m, time_s, mode = compute_hybrid_fastest_path(
            s_lat, s_lon, e_lat, e_lon
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(
        {
            "coords": coords,
            "total_distance_m": dist_m,
            "total_time_s": time_s,
            "routing_mode": mode,
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

            features.append(
                {
                    "coords": latlon,
                    "speed": speed,
                }
            )

    return jsonify(features)


if __name__ == "__main__":
    # Local dev server
    app.run(port=8000, debug=True)
