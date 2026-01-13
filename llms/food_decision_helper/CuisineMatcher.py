# !pip install transformers sentence-transformers faiss-cpu

import requests
from sentence_transformers import SentenceTransformer, util

# ===== CONFIG =====
LAT_LON_DEFAULT = (32.062882, 34.769206)  # Center of Tel Aviv
RADIUS = 1500  # meters


# ===== STEP 1: OSM Query =====
def get_restaurants(lat, lon, radius=RADIUS):
    query = f"""
    [out:json][timeout:60];
    node["amenity"="restaurant"](around:{radius},{lat},{lon});
    out;
    """
    try:
        r = requests.get("https://overpass-api.de/api/interpreter", params={"data": query}, timeout=30)
        data = r.json()
    except:
        print("Error fetching data from OSM")
        return []
    restaurants = []
    for n in data.get("elements", []):
        name = n.get("tags", {}).get("name")
        cuisine = n.get("tags", {}).get("cuisine")
        if name:
            restaurants.append({
                "name": name,
                "cuisine": cuisine,
                "lat": n["lat"],
                "lon": n["lon"]
            })
    return restaurants


# ===== STEP 2: Simple AI Matching =====
# Use a small multilingual embedding model to match cuisine keywords
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def match_restaurants(user_input, restaurants):
    if not restaurants:
        return []

    # Build embeddings for restaurant names + cuisine
    texts = [f"{r['name']} {r.get('cuisine', '')}" for r in restaurants]
    embeddings = model.encode(texts, convert_to_tensor=True)

    # Embed user input
    user_emb = model.encode(user_input, convert_to_tensor=True)

    # Compute similarity
    similarities = util.cos_sim(user_emb, embeddings)[0]
    scores = similarities.cpu().tolist()

    # Pick top 3 matches
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]
    top_restaurants = [restaurants[i] for i in top_idx]
    return top_restaurants


# ===== STEP 3: AI Agent =====
def ai_food_agent(user_input):
    lat, lon = LAT_LON_DEFAULT
    restaurants = get_restaurants(lat, lon)
    if not restaurants:
        print("לא נמצאו מסעדות באזור הקרוב.")
        return

    top_choices = match_restaurants(user_input, restaurants)

    print("🍽️ הנה מסעדות בשבילך:")
    for r in top_choices:
        print(f"- {r['name']} ({r.get('cuisine', 'לא ידוע')})")
        print(f"  מיקום: https://www.google.com/maps/search/?api=1&query={r['lat']},{r['lon']}")


# ===== RUN =====
if __name__ == "__main__":
    user_input = input("ספר לי איזה אוכל בא לך (עברית או אנגלית): ")
    ai_food_agent(user_input)
