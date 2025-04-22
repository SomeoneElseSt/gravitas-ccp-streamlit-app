import sys
from io import StringIO
sys.modules['StringIO'] = StringIO #applies patch to StringIO for Streamlit Deployment

import ee
import streamlit as st
import json
from google.oauth2 import service_account
from PIL import Image
import base64
import io 

# Earth Engine initialization with service account
if 'gcp_service_account' in st.secrets:
    service_account_info = dict(st.secrets['gcp_service_account'])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/earthengine']
    )
    ee.Initialize(credentials)
else:
    try:
        ee.Initialize(project='gravitasuhisteveapi-450908')
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project='gravitasuhisteveapi-450908')

# Import remaining libraries after Earth Engine is initialized
import geemap.foliumap as geemap
import datetime
import google.generativeai as genai


# Create Streamlit app layout
st.title("Land Surface Temperature And Urban Heat Index Analysis 🌡️")

# Create a map
uhi_map = geemap.Map()

# Cities with their center coordinates and appropriate bounding boxes
cities = {
    "San Francisco": {
        "center": [-122.4194, 37.7749],
        "bbox": [[-122.5194, 37.8749],
                 [-122.5194, 37.6749],
                 [-122.3194, 37.6749],
                 [-122.3194, 37.8749]]
    },
    "Belgrade": {
        "center": [20.4489, 44.7866],
        "bbox": [[20.3489, 44.8866],
                 [20.3489, 44.6866],
                 [20.5489, 44.6866],
                 [20.5489, 44.8866]]
    },
    "Zagreb": {
        "center": [15.9819, 45.8150],
        "bbox": [[15.8819, 45.9150],
                 [15.8819, 45.7150],
                 [16.0819, 45.7150],
                 [16.0819, 45.9150]]
    },
    "Sarajevo": {
        "center": [18.4131, 43.8563],
        "bbox": [[18.2131, 43.9563],
                 [18.2131, 43.7563],
                 [18.5131, 43.7563],
                 [18.5131, 43.9563]]
    },
    "Podgorica": {
        "center": [19.2594, 42.4304],
        "bbox": [[19.1594, 42.5304],
                 [19.1594, 42.3304],
                 [19.3594, 42.3304],
                 [19.3594, 42.5304]]
    },
    "Skopje": {
        "center": [21.4254, 41.9981],
        "bbox": [[21.3254, 42.0981],
                 [21.3254, 41.8981],
                 [21.5254, 41.8981],
                 [21.5254, 42.0981]]
    },
    "Tirana": {
        "center": [19.8189, 41.3275],
        "bbox": [[19.7189, 41.4275],
                 [19.7189, 41.2275],
                 [19.9189, 41.2275],
                 [19.9189, 41.4275]]
    },
    "Pristina": {
        "center": [21.1655, 42.6629],
        "bbox": [[21.0655, 42.7629],
                 [21.0655, 42.5629],
                 [21.2655, 42.5629],
                 [21.2655, 42.7629]]
    },
    "Novi Sad": {
        "center": [19.8444, 45.2671],
        "bbox": [[19.7444, 45.3671],
                 [19.7444, 45.1671],
                 [19.9444, 45.1671],
                 [19.9444, 45.3671]]
    },
    "Banja Luka": {
        "center": [17.1876, 44.7750],
        "bbox": [[17.0876, 44.8750],
                 [17.0876, 44.6750],
                 [17.2876, 44.6750],
                 [17.2876, 44.8750]]
    }
}

# City selection dropdown
selected_city = st.selectbox("Select a city", list(cities.keys()))

# Set AOI based on selected city
aoi = ee.Geometry.Polygon([cities[selected_city]["bbox"]])

# Date inputs - use Python's datetime directly, not ee.Date
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", value=datetime.date(2022, 5, 1))
with col2:
    end_date = st.date_input("End date", value=datetime.date(2022, 12, 31))

# Convert to strings when using with Earth Engine
ee_start_date = start_date.strftime("%Y-%m-%d")
ee_end_date = end_date.strftime("%Y-%m-%d")

# Convert the JavaScript functions to Python
def apply_scale_factors(image):
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

def mask_l8sr(col):
    # Bits 3 and 5 are cloud shadow and cloud, respectively
    cloud_shadow_bit_mask = (1 << 3)
    clouds_bit_mask = (1 << 5)
    # Get the pixel QA band
    qa = col.select('QA_PIXEL')
    # Both flags should be set to zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(
           qa.bitwiseAnd(clouds_bit_mask).eq(0))
    return col.updateMask(mask)

# Center map on selected city
uhi_map.centerObject(aoi, 11)

st.write("The start and end date comprise the time range of satellite images and data used for analysis.")

# Process imagery when user clicks the button
if st.button("Process Imagery") or 'processed_data' in st.session_state:
    
    # Only compute anew if the city, start date, or end date has changed
    compute_new = False
    
    # Check if new data needs to be computed
    if 'processed_data' not in st.session_state:
        compute_new = True
    elif st.session_state.processed_data.get('city') != selected_city:
        compute_new = True
    elif st.session_state.processed_data.get('start_date') != ee_start_date:
        compute_new = True
    elif st.session_state.processed_data.get('end_date') != ee_end_date:
        compute_new = True
    
    # Compute new data if needed
    if compute_new:
        with st.spinner(f"Processing imagery for {selected_city}... This may take a moment."):
            # Filter the collection using string dates
            image = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterDate(ee_start_date, ee_end_date) \
                .filterBounds(aoi) \
                .map(apply_scale_factors) \
                .map(mask_l8sr) \
                .median()

            # True color visualization
            vis_params = {
                'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
                'min': 0.0,
                'max': 0.3,
            }
            
            ## NDVI calculation
            ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
            
            # Calculate NDVI min and max for normalization
            ndvi_min = ee.Number(ndvi.reduceRegion(
                reducer=ee.Reducer.min(),
                geometry=aoi,
                scale=30,
                maxPixels=1e9
            ).values().get(0))

            ndvi_max = ee.Number(ndvi.reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=aoi,
                scale=30,
                maxPixels=1e9
            ).values().get(0))

            # FV calculation 
            fv = ndvi.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min)).pow(ee.Number(2)).rename('FV')
            
            # EM calculation
            em = fv.multiply(ee.Number(0.004)).add(ee.Number(0.986)).rename('EM')
            
            # Thermal band selection
            thermal = image.select('ST_B10').rename('thermal')
            
            # LST calculation
            lst = thermal.expression(
                '(tb / (1 + ((11.5 * (tb / 14380)) * log(em)))) - 273.15',
                {
                    'tb': thermal.select('thermal'),
                    'em': em
                }
            ).rename('LST')
            
            lst_vis = {
                'min': 7,
                'max': 50,
                'palette': [
                    '040274', '040281', '0502a3', '0502b8', '0502ce', '0502e6',
                    '0602ff', '235cb1', '307ef3', '269db1', '30c8e2', '32d3ef',
                    '3be285', '3ff38f', '86e26f', '3ae237', 'b5e22e', 'd6e21f',
                    'fff705', 'ffd611', 'ffb613', 'ff8b13', 'ff6e08', 'ff500d',
                    'ff0000', 'de0101', 'c21301', 'a71001', '911003'
                ]
            }
            
            # Mean and Standard Deviation of LST
            lst_mean = ee.Number(lst.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=30,
                maxPixels=1e9
            ).values().get(0))

            lst_std = ee.Number(lst.reduceRegion(
                reducer=ee.Reducer.stdDev(),
                geometry=aoi,
                scale=30,
                maxPixels=1e9
            ).values().get(0))
            
            # UHI calculation
            uhi = lst.subtract(lst_mean).divide(lst_std).rename('UHI')
            uhi_vis = {
                'min': -4,
                'max': 4,
                'palette': ['313695', '74add1', 'fed976', 'feb24c', 'fd8d3c', 'fc4e2a', 'e31a1c', 'b10026']
            }
            
            # UTFVI calculation
            utfvi = lst.subtract(lst_mean).divide(lst).rename('UTFVI')
            
            utfvi_vis = {
                'min': -1,
                'max': 0.3,
                'palette': ['313695', '74add1', 'fed976', 'feb24c', 'fd8d3c', 'fc4e2a', 'e31a1c', 'b10026']
            }
            
            # Store results in session state
            st.session_state.processed_data = {
                'city': selected_city,
                'start_date': ee_start_date,
                'end_date': ee_end_date,
                'ndvi': ndvi,
                'lst': lst,
                'lst_mean': lst_mean.getInfo(),
                'lst_std': lst_std.getInfo(),
                'uhi': uhi,
                'utfvi': utfvi,
                'ndvi_vis': {'min': -1, 'max': 1, 'palette': ['blue', 'white', 'green']},
                'lst_vis': lst_vis,
                'uhi_vis': uhi_vis,
                'utfvi_vis': utfvi_vis
            }
    
    # Use data from session state (whether just computed or from previous computation)
    data = st.session_state.processed_data
    
    # UHI Map
    uhi_map = geemap.Map()
    uhi_map.centerObject(aoi, 11)
    uhi_map.addLayer(aoi, {}, 'Area of Interest', False)
    uhi_map.addLayer(data['uhi'], data['uhi_vis'], 'Urban Heat Index', True, 0.7)
    
    # UHI Legend
    uhi_colors = ['313695', '74add1', 'fed976', 'feb24c', 'fd8d3c', 'fc4e2a', 'e31a1c', 'b10026']
    uhi_legend_dict = {
                '> 3°C Severe': uhi_colors[7],
                '2 - 3°C Strong': uhi_colors[6],
                '1 - 2°C Moderate': uhi_colors[5],
                '0 - 1°C Slight': uhi_colors[4],
                '± 0.5°C Neutral': uhi_colors[3],
                '–1 - 0°C Slight Cool': uhi_colors[2],
                '–2 - –1°C Moderate Cool': uhi_colors[1],
                '< 2°C Strong Cool': uhi_colors[0]
            }
    uhi_legend_style = {
        'right': '230px',
        'border': '1px solid gray',
        'background': 'white',
        'padding': '10px',
        'border-radius': '5px',
        'margin-bottom': '10px'
    }
    uhi_map.add_legend(title="", legend_dict=uhi_legend_dict, style=uhi_legend_style)
    uhi_map.add_layer_control()
    
    # Display the map
    st.write("### Urban Heat Index Map")
    st.write("This index highlights areas that are artificially hotter due to urban agglomeration, with the legend showing the relative temperature increase by color.")
    uhi_map.to_streamlit(height=500)
    
    # NDVI Map
    ndvi_map = geemap.Map()
    ndvi_map.centerObject(aoi, 11)
    ndvi_map.addLayer(data['ndvi'], data['ndvi_vis'], 'NDVI', True, 0.7)
    ndvi_map.addLayer(aoi, {}, 'Area of Interest', False)
    
    # NDVI Legend
    ndvi_legend_dict = {
        'Built-up Areas/Bare Surfaces': 'white',
        'Healthy/Dense Vegetation': 'green',
        'Water Bodies': 'blue'
    }
    ndvi_legend_style = {
        'left': '10px',
        'border': '1px solid gray',
        'background': 'white',
        'padding': '10px',
        'border-radius': '5px',
        'margin-bottom': '10px'
    }
    ndvi_map.add_legend(title="", legend_dict=ndvi_legend_dict, style=ndvi_legend_style)
    ndvi_map.add_layer_control()
    st.write("#### NDVI Map")
    st.write("This index highlights the presence and health of vegetation in an area. Higher values (green) indicate healthier vegetation.")
    ndvi_map.to_streamlit(height=500)
    
    # LST Map
    lst_map = geemap.Map()
    lst_map.centerObject(aoi, 11)
    lst_map.addLayer(data['lst'], data['lst_vis'], 'Land Surface Temperature', True, 0.7)
    lst_map.addLayer(aoi, {}, 'Area of Interest', False)
    
    # LST Legend
    lst_colors = ['040274', '040281', '0502a3', '0502b8', '0502ce', '0502e6',
                '0602ff', '235cb1', '307ef3', '269db1', '30c8e2', '32d3ef',
                '3be285', '3ff38f', '86e26f', '3ae237', 'b5e22e', 'd6e21f',
                'fff705', 'ffd611', 'ffb613', 'ff8b13', 'ff6e08', 'ff500d',
                'ff0000', 'de0101', 'c21301', 'a71001', '911003']
    lst_legend_dict = {
        '7°C - 14°C (Very Cool)': lst_colors[2],   
        '15°C - 24°C (Cool)': lst_colors[9],       
        '25°C - 34°C (Moderate)': lst_colors[14],  
        '35°C - 42°C (Warm)': lst_colors[18],      
        '43°C - 50°C (Very Hot)': lst_colors[24]   
    }
    lst_legend_style = {
        'left': '230px',
        'border': '1px solid gray',
        'background': 'white',
        'padding': '10px',
        'border-radius': '5px',
        'margin-bottom': '10px'
    }
    lst_map.add_legend(title="", legend_dict=lst_legend_dict, style=lst_legend_style)
    lst_map.add_layer_control()
    st.write("#### Land Surface Temperature Map")
    st.write("Shows the temperature of the land surface. Cooler areas are blue, warmer areas are red.")
    lst_map.to_streamlit(height=500)

    # Display LST statistics
    st.write(f"Mean LST (Land Surface Temperature) in {selected_city}: {data['lst_mean']:.2f}°C")
    st.write(f"Standard Deviation LST (Land Surface Temperature) in {selected_city}: {data['lst_std']:.2f}°C")
    
    # UTFVI Map
    utfvi_map = geemap.Map()
    utfvi_map.centerObject(aoi, 11)
    utfvi_map.addLayer(data['utfvi'], data['utfvi_vis'], 'Urban Thermal Field Variance Index', True, 0.7)
    utfvi_map.addLayer(aoi, {}, 'Area of Interest', False)
    
    # UTFVI Legend
    utfvi_legend_dict = {
        'High Heat Stress': 'b10026',       
        'Moderate Heat Stress': 'e31a1c',   
        'Mild Heat Stress': 'fc4e2a',       
        'Neutral': 'fd8d3c',                
        'Cooling Effect': 'feb24c'          
    }
    utfvi_legend_style = {
        'right': '10px',
        'border': '1px solid gray',
        'background': 'white',
        'padding': '10px',
        'border-radius': '5px',
        'margin-bottom': '10px'
    }
    utfvi_map.add_legend(title="", legend_dict=utfvi_legend_dict, style=utfvi_legend_style)
    utfvi_map.add_layer_control()
    st.write("#### Urban Thermal Field Variance Index Map")
    st.write("This index classifies urban areas by temperature comfort levels for humans, showing which city areas **feel** cooler or hotter than the city/area average.")
    utfvi_map.to_streamlit(height=500)
else:
    # Create a default map with no layers when the button hasn't been clicked
    default_map = geemap.Map()
    default_map.centerObject(aoi, 11)
    default_map.add_layer_control()
    default_map.to_streamlit(height=500)

st.header("How to read")
st.write("To toggle on and off different masks, hover over the layers icon in the top right corner of the map.")
st.write("**Normalized Difference Vegetation Index (NDVI)**: This index highlights the presence and health of vegetation in an area. It is calculated using the reflectance values from the near-infrared (NIR) and red bands of satellite imagery. Healthy vegetation reflects more NIR and absorbs more red light, resulting in a higher NDVI value. The NDVI formula is shown below:")
st.latex(r'''NDVI = \frac{NIR - Red}{NIR + Red}''')
st.write("**Land Surface Temperature (LST)**: Highlights the map based on the surface temperature of each point using a red-blue scale. The bluer the area, the lesser its temperature, and vice versa with red areas. It is calculated accounting for brightness temperature, emitted radiance, and land surface emissivity. The LST formula is shown below:")
st.latex(r'''LST = \frac{BT} {(1 + (λ * BT / c2) * ln(E))}''')
st.write("**Urban Heat Index (UHI)**: This index categorizes how much hotter a point in a city is relative to the LST mean divided by the LST Standard Deviation. Red spots signal that an urban area has more concentrated heat islands relative to the LST in the rest of the AOI. This mask shows the areas that are artificially hotter due to urban agglomeration. The UHI formula is shown below:")
st.latex(r'''UHI = \frac {LST - LSTm} {SD}''')
st.write("**Urban Thermal Field Variance Index (UTFVI)**: This index further intensifies the UHI effect by quantifying the variation in LST across urban areas. It provides a measure of how temperature fluctuates across an area by comparing LST deviations relative to the LST itself. The formula is shown below:")
st.latex(r'''UTFVI = \frac {LST - LSTm} {LST}''')
st.write("**Area of Interest (AOI)**: This is an outline of the area that will be analyzed. It is important to get the average LST of the area to calculate the UHI.")

st.header("Blank City Spots Notes")
st.write("For some cities, there are spots that show up as blank or are not properly masked (e.g. Belgrade). This is due to limitations of the Landsat's 8 dataset, and is an issue we will look to resolve or circumvent in the future.")


### Beguinning of Chatbot Code

# Set Gemini API key from Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Function to load image for Gemini
def load_image_for_gemini(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Create a byte stream
        byte_stream = io.BytesIO()
        # Save the image to the byte stream in JPEG format
        image.save(byte_stream, format="JPEG")
        # Get the byte value and encode as base64
        image_bytes = base64.b64encode(byte_stream.getvalue()).decode('utf-8')
        return image_bytes
    return None

st.header("Interactive Chatbot")
st.write("Ask a chatbot about what each metric means and how to interpret the maps.")
# Set Gemini API key from Streamlit secrets

# Example format with dynamic building type
st.info(f"""
Example question format:
        
Analyze the thermal patterns in the captured screenshot to identify urban heat island effects and recommend optimal locations for new buildings, indicating specific streets or districts to choose or avoid, to minimize environmental impact.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Gemini chat with system prompt
if "gemini_chat" not in st.session_state:
    try:
        # Use Gemini 2.0 Flash model
        model = genai.GenerativeModel('gemini-2.0-flash')
        system_prompt = """
        
        You are an expert in urban heat analysis and building thermal patterns. 
        For each user query, ensure they've provided:
        1. Type of building (residential, commercial, industrial, etc.)
        2. Area of interest in the city
        3. Specific analysis request
        
        If any of these are missing, politely ask for the missing information.
        When analyzing, consider:
        - Land Surface Temperature (LST)
        - Urban Heat Island (UHI) effects
        - Urban Thermal Field Variance Index (UTFVI)
        - Surrounding vegetation (NDVI)
        
        When analyzing images:
        - Identify areas of high heat concentration
        - Note patterns in urban heat distribution
        - Comment on potential vegetation impacts
        - Suggest mitigation strategies if relevant
        
        Provide concise, practical insights based on both the thermal analysis data and visual patterns.
        
        Assume the user is interested in learning more about what the graphs mean for Urban Planning Recommendations.

        Also explain what each map shows (e.g. the neighboorhood in red must be avoided for new buildings because it has a high UHI, etc)
        
        """
        
        st.session_state.gemini_chat = model.start_chat(history=[])
        # Send system prompt
        st.session_state.gemini_chat.send_message(system_prompt)
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add image upload functionality
uploaded_image = st.file_uploader("Upload a screenshot of the map (optional)", type=['png', 'jpg', 'jpeg'])
if uploaded_image is not None:
    # Display the image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Screenshot")
    
    # Process image for Gemini
    st.session_state['current_image'] = load_image_for_gemini(uploaded_image)

# Accept user input
if prompt := st.chat_input("Ask a bot about the indexes or upload a map screenshot to have it explained."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # For messages with images
            if 'current_image' in st.session_state and st.session_state['current_image'] is not None:
                # Send both the image and the prompt
                parts = [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": st.session_state['current_image']
                        }
                    }
                ]
                response = st.session_state.gemini_chat.send_message(parts)
            else:
                # For text-only messages
                response = st.session_state.gemini_chat.send_message(prompt)
            
            for chunk in response:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            message_placeholder.markdown(error_message)
            full_response = error_message
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

