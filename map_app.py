import streamlit as st
import json
import pandas as pd
import requests
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh

# --- Configuration ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
CACHE_EXPIRATION_SECONDS = 5
API_URL = "https://railradar.in/api/v1/trains/live-map"
API_HEADERS = {
    'x-api-key': 'rri_eyJleHAiOjE3NTc4NzUyODEzMzEsImlhdCI6MTc1Nzc4ODg4MTMzMSwidHlwZSI6ImludGVybmFsIiwicm5kIjoiVkJrYWpCUERINlY5In0=_NDQyYjk0MzljNWZkNWEyM2Y4MTliMzQ4MjczNDUyNmFkNTZlM2NhMTg0ZTliOTQ2NWEzYTY4ZDc0MWU5ODdmMg==',
    'Cookie': 'user_id=ca319a54450a46e7929a2b79a31592c8'
}

st.set_page_config(page_title="Live Train Map", page_icon="🚆", layout="wide")

# Auto-refresh the app every 5 seconds
st_autorefresh(interval=5000, key="data_refresher")

st.title("Live Train Map")

def fetch_and_update_local_data():
    """
    Fetches live train data from the API. If successful, it updates
    livedata.json and returns the new data.
    """
    try:
        response = requests.get(API_URL, headers=API_HEADERS, timeout=10)
        response.raise_for_status()
        live_data = response.json()
        
        with open('livedata.json', 'w') as f:
            json.dump(live_data, f, indent=4)
            
        st.success("Fetched live data and updated `livedata.json`!")
        return live_data
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch live data: {e}. The API key may have expired.")
        return None

def load_local_data():
    """Loads data from the local livedata.json file."""
    st.info("Displaying data from local `livedata.json` file.")
    try:
        with open('livedata.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("`livedata.json` not found. Please make sure the file exists.")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding `livedata.json`. The file might be corrupted.")
        return None

# --- Main App Logic ---

# On each refresh, try to get live data. If it fails, load the last known local data.
train_data = fetch_and_update_local_data()
if not train_data:
    train_data = load_local_data()


if train_data:
    trains = train_data.get('data', [])

    if trains:
        map_data = []
        for train in trains:
            # Ensure the train has valid, non-null location data
            if train.get('current_lat') is not None and train.get('current_lng') is not None:
                map_data.append({
                    'lat': train['current_lat'],
                    'lon': train['current_lng'],
                    'train_name': train.get('train_name', 'N/A'),
                    'train_number': train.get('train_number', 'N/A'),
                    'current_station': train.get('current_station_name', 'N/A'),
                    'type': train.get('type', 'N/A'),
                    'next_station': train.get('next_station_name', 'N/A'),
                    'distance_from_source': train.get('distance_from_source_km', 'N/A'),
                    'mins_since_departure': train.get('mins_since_dep', 'N/A'),
                })

        if map_data:
            df = pd.DataFrame(map_data)
            
            # As a final safeguard, drop any rows that might still have nulls
            df.dropna(subset=['lat', 'lon'], inplace=True)

            # --- Filter UI ---
            st.markdown("### Filters")
            col1, col2 = st.columns(2)

            with col1:
                train_types = ['All'] + sorted(df['type'].unique().tolist())
                selected_type = st.selectbox("Filter by Train Type:", train_types)
            
            # Filter the dataframe based on type selection
            if selected_type != 'All':
                df_type_filtered = df[df['type'] == selected_type].copy()
            else:
                df_type_filtered = df.copy()

            with col2:
                train_names = ['All'] + sorted(df_type_filtered['train_name'].unique().tolist())
                selected_name = st.selectbox("Filter by Train Name (Route):", train_names)

            # Filter the dataframe based on name selection
            if selected_name != 'All':
                df_filtered = df_type_filtered[df_type_filtered['train_name'] == selected_name].copy()
            else:
                df_filtered = df_type_filtered.copy()


            if df_filtered.empty:
                st.warning(f"No trains found for the selected filters.")
            else:
                # --- Map Display ---
                tooltip = {
                    "html": """
                    <b>Train Name:</b> {train_name} <br/>
                    <b>Train Number:</b> {train_number} <br/>
                    <b>Current Station:</b> {current_station} <br/>
                    <b>Next Station:</b> {next_station} <br/>
                    <b>Type:</b> {type} <br/>
                    <b>Distance from Source:</b> {distance_from_source} km <br/>
                    <b>Minutes Since Departure:</b> {mins_since_departure}
                    """,
                    "style": {"backgroundColor": "steelblue", "color": "white"}
                }

                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=df_filtered,
                    get_position='[lon, lat]',
                    get_fill_color='[200, 30, 0, 160]',
                    get_radius=5,
                    radius_units='pixels',  # This ensures fixed dot size
                    pickable=True,
                )

                view_state = pdk.ViewState(
                    latitude=float(df_filtered['lat'].mean()),
                    longitude=float(df_filtered['lon'].mean()),
                    zoom=4.5,
                    pitch=0,
                )

                deck = pdk.Deck(
                    map_provider='carto',
                    map_style='dark',
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip=tooltip,
                )
                st.pydeck_chart(deck)

                with st.expander("Show Raw Data"):
                    st.write(df_filtered)
        else:
            st.warning("No trains with location data found.")
    else:
        st.warning("No train data found in the data source.")
else:
    st.error("Could not fetch or load any train data.")
