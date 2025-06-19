import json
from geopy.geocoders import Nominatim
import folium

geolocator = Nominatim(user_agent="inst-profile")

with open("processed_meta.json", encoding="utf-8") as f:
    data = json.load(f)
locations = set()
for post in data:
    for loc in post.get("locations", []):
        locations.add(loc)

coords = {}
for loc in locations:
    try:
        location = geolocator.geocode(loc)
        if location:
            coords[loc] = (location.latitude, location.longitude)
    except Exception as e:
        print(f"Ошибка геокодирования {loc}: {e}")

if coords:
    first = list(coords.values())[0]
    m = folium.Map(location=first, zoom_start=6)
    for name, (lat, lon) in coords.items():
        folium.Marker([lat, lon], popup=name).add_to(m)
    m.save("map.html")
    print("Карта сохранена в map.html")
else:
    print("Нет координат для построения карты.")
