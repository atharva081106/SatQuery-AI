import numpy as np
import rasterio

def generate_bitemporal_tifs():
    # Read the real optical image to use as a rich structural background
    with rasterio.open('d:/sih26167/real_optical_spot.tif') as src:
        meta = src.meta
        base_img = src.read(1)
        
    height = meta['height']
    width = meta['width']

    def create_geotiff(filename, data):
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(data, 1)

    # 1. Deforestation Pair
    print("Generating deforestation pair...")
    deforest_t1 = base_img.copy()
    deforest_t2 = base_img.copy()
    
    # Introduce a bright, bare soil "clearcut" patch in T2
    patch_t2 = deforest_t2[300:500, 300:500]
    deforest_t2[300:500, 300:500] = np.clip(patch_t2 * 0.2 + 180 + np.random.normal(0, 10, patch_t2.shape), 0, 255).astype(np.uint8)
    
    create_geotiff('d:/sih26167/optical_deforestation_T1.tif', deforest_t1)
    create_geotiff('d:/sih26167/optical_deforestation_T2.tif', deforest_t2)

    # 2. Urban Growth Pair
    print("Generating urban growth pair...")
    urban_t1 = base_img.copy()
    urban_t2 = base_img.copy()
    
    # Introduce urban expansion (high contrast, bright speckles) in T2
    expansion_mask = np.zeros((height, width), dtype=bool)
    expansion_mask[100:400, 400:600] = True
    patch_u2 = urban_t2[expansion_mask]
    urban_t2[expansion_mask] = np.clip(patch_u2 * 0.5 + 160 + np.random.normal(0, 30, patch_u2.shape), 0, 255).astype(np.uint8)
    
    create_geotiff('d:/sih26167/optical_urban_growth_T1.tif', urban_t1)
    create_geotiff('d:/sih26167/optical_urban_growth_T2.tif', urban_t2)

    # 3. Flooding Pair
    print("Generating flooding pair...")
    flood_t1 = base_img.copy()
    flood_t2 = base_img.copy()
    
    # Introduce dark flood waters covering a horizontal swath
    flood_mask = np.zeros((height, width), dtype=bool)
    flood_mask[400:650, 100:600] = True
    patch_f2 = flood_t2[flood_mask]
    flood_t2[flood_mask] = np.clip(patch_f2 * 0.15 + 25 + np.random.normal(0, 5, patch_f2.shape), 0, 255).astype(np.uint8)
    
    create_geotiff('d:/sih26167/optical_flood_T1.tif', flood_t1)
    create_geotiff('d:/sih26167/optical_flood_T2.tif', flood_t2)
    
    print("Successfully generated 3 pairs with rich structural background for ORB feature matching!")

if __name__ == "__main__":
    generate_bitemporal_tifs()
