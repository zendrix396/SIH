from geopy.geocoders import Nominatim

# India's approximate bounding box
INDIA_BBOX = {
    "min_lat": 8.4,
    "max_lat": 37.6,
    "min_lon": 68.7,
    "max_lon": 97.25,
}

def get_state_from_lat_lon(lat, lon):
    # Check if coordinates are inside India's bounding box
    if not (INDIA_BBOX["min_lat"] <= lat <= INDIA_BBOX["max_lat"] and
            INDIA_BBOX["min_lon"] <= lon <= INDIA_BBOX["max_lon"]):
        return None

    geolocator = Nominatim(user_agent="india_state_locator")
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language='en')
        if not location:
            return None
        address = location.raw.get('address', {})
        country = address.get('country', '')
        # Confirm country is India
        if country.lower() != "india":
            return None
        # Get state name from address dictionary keys
        state = address.get('state')
        if not state:
            # Sometimes state might be in 'state_district' or 'region'
            state = address.get('state_district') or address.get('region')
        return state
    except Exception as e:
        # In case of any error, return None
        return None

# Example usage
if __name__ == "__main__":
    lat = 26.00137  
    lon = 83.67239  
    state = get_state_from_lat_lon(lat, lon)
    print(state)  
