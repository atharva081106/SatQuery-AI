import numpy as np
import rasterio
from rasterio.transform import from_origin

def generate_geotiff(filename, start_lon, start_lat, r_color, g_color, b_color):
    width = 256
    height = 256
    
    # Generate some basic patterns (noise + base color) to simulate terrain variance
    r = (np.random.rand(height, width) * 50 + r_color).clip(0, 255).astype(np.uint8)
    g = (np.random.rand(height, width) * 50 + g_color).clip(0, 255).astype(np.uint8)
    b = (np.random.rand(height, width) * 50 + b_color).clip(0, 255).astype(np.uint8)
    
    data = np.stack([r, g, b])
    
    # Set transform (0.0001 deg per pixel approx 11m)
    transform = from_origin(start_lon, start_lat, 0.0001, 0.0001)
    
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=data.dtype,
        crs='+proj=latlong +datum=WGS84',
        transform=transform,
    ) as dst:
        dst.write(data)

# Generate synthetic GeoTIFFs with valid geographic coordinates and CRS
generate_geotiff("test_amazon_forest.tif", -60.025, -3.119, 34, 139, 34)
generate_geotiff("test_new_york_urban.tif", -74.006, 40.712, 105, 105, 105)
generate_geotiff("test_sahara_desert.tif", 10.0, 23.0, 237, 201, 175)

print("Generated 3 GeoTIFFs successfully.")
