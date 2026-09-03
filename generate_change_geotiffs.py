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
        count=3,
        dtype=data.dtype,
        crs='+proj=latlong +datum=WGS84',
        transform=transform,
    ) as dst:
        dst.write(data)

def generate_change_pairs():
    width, height = 256, 256
    
    # ---------------------------------------------------------
    # PAIR 1: Deforestation (Amazon)
    # T1: Dense Forest
    # T2: Forest with a large cleared patch in the center
    # ---------------------------------------------------------
    lon_amz, lat_amz = -60.025, -3.119
    
    # Base forest (Green)
    r_f = (np.random.rand(height, width) * 30 + 20).astype(np.uint8)
    g_f = (np.random.rand(height, width) * 40 + 120).astype(np.uint8)
    b_f = (np.random.rand(height, width) * 30 + 20).astype(np.uint8)
    t1_forest = np.stack([r_f, g_f, b_f])
    
    # Copy T1 to T2, but add a dirt patch
    t2_forest = t1_forest.copy()
    
    # Dirt colors (Brown)
    r_d = (np.random.rand(height, width) * 20 + 130).astype(np.uint8)
    g_d = (np.random.rand(height, width) * 20 + 90).astype(np.uint8)
    b_d = (np.random.rand(height, width) * 20 + 50).astype(np.uint8)
    
    # Create an irregular clearcut shape in the middle
    mask = np.zeros((height, width), dtype=bool)
    mask[100:180, 80:160] = True
    mask[90:120, 150:190] = True # extra path
    
    t2_forest[0, mask] = r_d[mask]
    t2_forest[1, mask] = g_d[mask]
    t2_forest[2, mask] = b_d[mask]
    
    create_geotiff("deforestation_T1.tif", t1_forest, lon_amz, lat_amz)
    create_geotiff("deforestation_T2.tif", t2_forest, lon_amz, lat_amz)


    # ---------------------------------------------------------
    # PAIR 2: Urban Sprawl (City edge)
    # T1: Small city on the left, fields on the right
    # T2: City expands to cover most of the fields
    # ---------------------------------------------------------
    lon_city, lat_city = -74.006, 40.712
    
    # Base Fields (Light Green)
    r_fld = (np.random.rand(height, width) * 20 + 80).astype(np.uint8)
    g_fld = (np.random.rand(height, width) * 20 + 150).astype(np.uint8)
    b_fld = (np.random.rand(height, width) * 20 + 60).astype(np.uint8)
    
    # Base City (Grey)
    r_cty = (np.random.rand(height, width) * 30 + 120).astype(np.uint8)
    g_cty = (np.random.rand(height, width) * 30 + 120).astype(np.uint8)
    b_cty = (np.random.rand(height, width) * 30 + 120).astype(np.uint8)
    
    # T1: City only on the left side
    t1_urban = np.stack([r_fld, g_fld, b_fld])
    mask_t1 = np.zeros((height, width), dtype=bool)
    mask_t1[:, :80] = True # city occupies left third
    t1_urban[0, mask_t1] = r_cty[mask_t1]
    t1_urban[1, mask_t1] = g_cty[mask_t1]
    t1_urban[2, mask_t1] = b_cty[mask_t1]
    
    # T2: City expands significantly
    t2_urban = t1_urban.copy()
    mask_t2 = np.zeros((height, width), dtype=bool)
    mask_t2[:, 80:200] = True # city expands
    t2_urban[0, mask_t2] = r_cty[mask_t2]
    t2_urban[1, mask_t2] = g_cty[mask_t2]
    t2_urban[2, mask_t2] = b_cty[mask_t2]
    
    create_geotiff("urban_growth_T1.tif", t1_urban, lon_city, lat_city)
    create_geotiff("urban_growth_T2.tif", t2_urban, lon_city, lat_city)

if __name__ == "__main__":
    generate_change_pairs()
    print("Change analysis pairs generated successfully!")
