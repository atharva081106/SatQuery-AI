import numpy as np
import rasterio

def generate_synthetic_optical_tifs():
    # Read base metadata from real_optical_spot.tif
    with rasterio.open('d:/sih26167/real_optical_spot.tif') as src:
        meta = src.meta

    height = meta['height']
    width = meta['width']

    def create_geotiff(filename, data):
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(data, 1)

    # Scenario 1: Agriculture (Checkerboard / Crop fields)
    print("Generating optical_agriculture.tif...")
    ag_data = np.zeros((height, width), dtype=np.uint8)
    # create grid of fields
    for i in range(0, height, 100):
        for j in range(0, width, 100):
            # field intensity varies
            intensity = np.random.randint(60, 220)
            ag_data[i:i+95, j:j+95] = intensity
    # Add a little noise
    ag_data = np.clip(ag_data + np.random.normal(0, 5, (height, width)), 0, 255).astype(np.uint8)
    create_geotiff('d:/sih26167/optical_agriculture.tif', ag_data)

    # Scenario 2: Cloud Cover over terrain
    print("Generating optical_cloud_cover.tif...")
    cloud_data = np.full((height, width), 80, dtype=np.float32) # base ground
    y, x = np.ogrid[:height, :width]
    # Add multiple gaussian clouds
    for _ in range(8):
        cy = np.random.randint(0, height)
        cx = np.random.randint(0, width)
        sigma = np.random.randint(40, 150)
        dist = (y - cy)**2 + (x - cx)**2
        cloud_add = np.exp(-dist / (2.0 * sigma**2)) * 175
        cloud_data += cloud_add
    
    cloud_data = np.clip(cloud_data, 0, 255).astype(np.uint8)
    create_geotiff('d:/sih26167/optical_cloud_cover.tif', cloud_data)

    # Scenario 3: Water Body / Coastline
    print("Generating optical_water_body.tif...")
    water_data = np.full((height, width), 140, dtype=np.uint8) # land
    # Create a simple diagonal coast
    for i in range(height):
        for j in range(width):
            if j < i * (width / height) - 50 + np.sin(i / 20.0) * 30:
                water_data[i, j] = 30 # dark water
            elif j < i * (width / height) + np.sin(i / 20.0) * 30:
                water_data[i, j] = 180 # bright beach/shore
    
    # Add some noise
    water_data = np.clip(water_data + np.random.normal(0, 3, (height, width)), 0, 255).astype(np.uint8)
    create_geotiff('d:/sih26167/optical_water_body.tif', water_data)

    print("Successfully generated 3 optical TIFFs for different analyses!")

if __name__ == "__main__":
    generate_synthetic_optical_tifs()
