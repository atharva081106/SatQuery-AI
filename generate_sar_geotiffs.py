import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_geotiff(filename, data, start_lon, start_lat):
    height, width = data.shape[1], data.shape[2]
    transform = from_origin(start_lon, start_lat, 0.0001, 0.0001)
    
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='+proj=latlong +datum=WGS84',
        transform=transform,
    ) as dst:
        dst.write(data)

def generate_sar_files():
    width, height = 256, 256
    
    # Simulate SAR speckle noise (Rayleigh/Exponential-like distribution)
    def add_speckle(base_intensity):
        # Multiplicative noise common in SAR
        noise = np.random.exponential(scale=1.0, size=(height, width))
        return np.clip(base_intensity * noise, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------
    # SCENARIO 1: Ship Detection in Ocean
    # Background: Dark (water backscatters very little)
    # Target: Extremely bright pixels (ships are corner reflectors)
    # ---------------------------------------------------------
    lon_ocean, lat_ocean = -40.0, 30.0
    
    # Ocean base intensity (very low)
    ocean_base = np.full((height, width), 20.0)
    
    # Add a few ships (bright spots)
    ocean_base[100:105, 120:123] = 250.0  # Ship 1
    ocean_base[180:182, 50:52] = 200.0    # Ship 2 (smaller)
    ocean_base[40:48, 200:205] = 255.0    # Ship 3 (large vessel)
    
    sar_ships = add_speckle(ocean_base)
    # Needs a channel dimension for the writer: (1, H, W)
    create_geotiff("sar_ship_detection.tif", sar_ships[np.newaxis, ...], lon_ocean, lat_ocean)


    # ---------------------------------------------------------
    # SCENARIO 2: Flood Monitoring (Bi-polar Change)
    # T1: Rough land surface (medium intensity)
    # T2: Flooded area appears very dark (water)
    # ---------------------------------------------------------
    lon_land, lat_land = 88.36, 22.57 # Approx Kolkata region
    
    # Land base intensity (medium)
    land_base = np.full((height, width), 100.0)
    # Add some structural features (brighter ridges/buildings)
    land_base[150:170, :] = 140.0
    land_base[:, 80:90] = 130.0
    
    # T1
    sar_flood_t1 = add_speckle(land_base)
    
    # T2 (Flooded)
    flooded_base = land_base.copy()
    # Create a large flood plain (dark)
    flooded_base[100:200, 100:220] = 15.0 
    
    # Ensure the noise instance is different for T2 (temporal decorrelation)
    sar_flood_t2 = add_speckle(flooded_base)
    
    create_geotiff("sar_flood_T1.tif", sar_flood_t1[np.newaxis, ...], lon_land, lat_land)
    create_geotiff("sar_flood_T2.tif", sar_flood_t2[np.newaxis, ...], lon_land, lat_land)

if __name__ == "__main__":
    generate_sar_files()
    print("SAR testing files generated successfully!")
