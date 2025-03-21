import sys
from io import StringIO
sys.modules['StringIO'] = StringIO #applies patch to StringIO for Streamlit Deployment

import ee
import streamlit as st
import json

# to initialize this project locally, follow these steps:

# 1. Create a virtual environment or .venv in your file directory with: python -m venv .venv and activate it w/  .venv\Scripts\activate on Windows or w/   source .venv/bin/activate on Mac

# 2. Install the required libraries above with pip install ee streamlit geemap datetime setuptools earthengine-api folium

# 3. Set up a GOOGLE_CLOUD_KEY in a .env file in the same directory and make sure that the earth engine api is enabled on google's developer console

# 4. Make sure to change the project name in ee.Initialize to your project name in Google Cloud

# 5. Upon first running this, it should execute an oAuth workflow with google to verify the key. if you face errors make sure the key credentials are set up correectly in google cloud 

# 6. Once you're set up, you can run + test the application dashborad by running streamlit run prototype_v1.py on the terminal

# For Streamlit Cloud deployment
 if 'gcp_service_account' in st.secrets:
     # Convert AttrDict to regular dict
     service_account = dict(st.secrets['gcp_service_account'])
     credentials = ee.ServiceAccountCredentials(
         service_account['client_email'],
         key_data=json.dumps(service_account)
     )
     ee.Initialize(credentials)
 else:
     # Fallback for local development
     try:
         ee.Initialize(project='gravitasuhisteveapi')
     except Exception as e:
         ee.Authenticate()
         ee.Initialize(project='gravitasuhisteveapi')


import geemap.foliumap as geemap
import datetime #libraries are imported after google earth engine is initialized 


# Create Streamlit app layout
st.title("Land Surface Temperature And Urban Heat Index Analysis 🌡️")

# Create a map
Map = geemap.Map()

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
Map.centerObject(aoi, 11)

st.write("The start and end date comprise the time range of satellite images and data used for analysis.")

# Process imagery when user clicks the button
if st.button("Process Imagery"):
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
        
        ## NDVI - Normalized Difference Vegetation Index,  NDVI = (NIR - Red) / (NIR + Red)
        # Helps identify vegetation, higher NDVI values indicate dense vegetation
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

        # FV - Fraction of Vegetation, FV = ((NDVI - NDVI_min) / (NDVI_max - NDVI_min))^2
        # Represents the proportion of vegetation in a pixel.
        fv = ndvi.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min)).pow(ee.Number(2)).rename('FV')
        
        # EM - Emissivity, EM = 0.004 * FV + 0.986
        # Accounts for the emissivity of land surface, necessary for LST calculations.
        em = fv.multiply(ee.Number(0.004)).add(ee.Number(0.986)).rename('EM')
        
        # Selecting thermal band for brightness temperature calculation
        thermal = image.select('ST_B10').rename('thermal')
        
        # LST - Land Surface Temperature, LST = BT / (1 + ((λ * BT / c2) * ln(EM))) - 273.15
        # Corrects top-of-atmosphere brightness temperature to land surface temperature.
        lst = thermal.expression(
            '(tb / (1 + ((11.5 * (tb / 14380)) * log(em)))) - 273.15',
            {
                'tb': thermal.select('thermal'),# Brightness temperature
                'em': em # Land surface emissivity
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
        
        # UHI - Urban Heat Island,  UHI = (LST - LST_mean) / LST_std
        # Standardizes LST variations to highlight urban heat islands.
        uhi = lst.subtract(lst_mean).divide(lst_std).rename('UHI')
        uhi_vis = {
            'min': -4,
            'max': 4,
            'palette': ['313695', '74add1', 'fed976', 'feb24c', 'fd8d3c', 'fc4e2a', 'e31a1c', 'b10026']
        }
        
        
        # UTFVI - Urban Thermal Field Variance Index, UTFVI = (LST - LST_mean) / LST
        # Intensifies the UHI effect by quantifying the variation in LST across urban areas.
        utfvi = lst.subtract(lst_mean).divide(lst).rename('UTFVI')
        
        utfvi_vis = {
            'min': -1,
            'max': 0.3,
            'palette': ['313695', '74add1', 'fed976', 'feb24c', 'fd8d3c', 'fc4e2a', 'e31a1c', 'b10026']
        }
        
        # Add layers to map
        Map.addLayer(image, vis_params, 'True Color (432)')
        Map.addLayer(ndvi, {'min': -1, 'max': 1, 'palette': ['blue', 'white', 'green']}, 'Normalized Difference Vegetation Index')
        Map.addLayer(lst, lst_vis, 'Land Surface Temperature')
        Map.addLayer(uhi, uhi_vis, 'Urban Heat Index')
        Map.addLayer(utfvi, utfvi_vis, 'Urban Thermal Field Variance Index')
        
        # Display LST statistics
        st.write(f"Mean LST (Land Surface Temperature) in {selected_city}: {lst_mean.getInfo():.2f}°C")
        st.write(f"Standard Deviation LST (Land Surface Temperature) in {selected_city}: {lst_std.getInfo():.2f}°C")





# Draw the bounding box on the map to show the AOI
Map.addLayer(aoi, {}, 'Area of Interest')

# Move the layer control here, after all layers are added
Map.add_layer_control()

# Display the map
Map.to_streamlit(height=600)

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

st.header("Prototype Notes")
st.write("For some cities, there are spots that show up as blank or are not properly masked (e.g., Belgrade). This is due to limitations of the Landsat's 8 dataset, and is an issue we will look to resolve or circumvent in the future.")
