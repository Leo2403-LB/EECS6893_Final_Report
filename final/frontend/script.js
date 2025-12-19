//Map Initialization
let map = L.map("map").setView([40.758, -73.9855], 13);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors"
}).addTo(map);

// Markers + Route Layer
let startMarker = null;
let endMarker = null;
let routeLayer = null;

let clickCount = 0;

//  Clicks
map.on("click", async function (e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;

    clickCount++;

    
    if (clickCount === 1) {
        if (startMarker) map.removeLayer(startMarker);
        startMarker = L.marker([lat, lon], { draggable: false })
            .bindPopup("Start")
            .addTo(map);
        document.getElementById("routeInfo").innerHTML = "Start selected.<br>Click on the map to select END.";
    }

    
    else if (clickCount === 2) {
        if (endMarker) map.removeLayer(endMarker);
        endMarker = L.marker([lat, lon], { draggable: false })
            .bindPopup("End")
            .addTo(map);

        document.getElementById("routeInfo").innerHTML = "Loading route...";

        // Fetch route from backend
        getRoute(startMarker.getLatLng(), endMarker.getLatLng());

        // Reset after choosing start+end
        clickCount = 0;
    }
});

//Fetch Route from Backend 
async function getRoute(start, end) {
    const url =
        `http://127.0.0.1:8000/route?start=${start.lat},${start.lng}&end=${end.lat},${end.lng}`;

    try {
        const res = await fetch(url);
        const data = await res.json();

        if (data.error) {
            document.getElementById("routeInfo").innerHTML = "<b>Error:</b> " + data.error;
            return;
        }

        // Remove old route if exists
        if (routeLayer) map.removeLayer(routeLayer);

        // Draw route polyline
        routeLayer = L.polyline(data.coords, {
            color: "blue",
            weight: 5
        }).addTo(map);

        // Fit map to route
        map.fitBounds(routeLayer.getBounds());

        // Show ETA and distance
        let minutes = (data.total_time_s / 60).toFixed(1);
        let km = (data.total_distance_m / 1000).toFixed(2);

        document.getElementById("routeInfo").innerHTML =
            `<b>Route Loaded:</b><br>
             Distance: ${km} km<br>
             ETA: ${minutes} minutes<br>`;

    } catch (err) {
        document.getElementById("routeInfo").innerHTML =
            "<b>Error contacting backend.</b>";
        console.error(err);
    }
}
