# inference_utils.py
import os
import glob
import zipfile

import numpy as np
import torch
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape as shape_geom
import segmentation_models_pytorch as smp
import ee
import geemap
import json


# -----------------------------
# MODEL LOADING
# -----------------------------

def build_model():
    """Construct the segmentation model architecture.

    The checkpoint in ``water_unet_best.pth`` is a U-Net with a ResNet-34
    encoder, 6 input channels and 2 output classes (background, water).
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6,
        classes=2,
    )
    return model


def load_model(model_path, device="cpu"):
    """Load a model from ``model_path`` on the requested device.

    Handles both full-model checkpoints (``torch.save(model, path)``) and
    ``state_dict``-only checkpoints (``torch.save(model.state_dict(), path)``).
    """
    checkpoint = torch.load(model_path, map_location=device)

    # If a full model was saved, just use it directly.
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        # ``checkpoint`` is an OrderedDict state_dict: build the architecture
        # and load weights into it.
        model = build_model()
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        # Optional: log any mismatch information to help debugging in the UI.
        if missing or unexpected:
            print("[load_model] Missing keys:", missing)
            print("[load_model] Unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model


# -----------------------------
# GOOGLE EARTH ENGINE HELPERS (AOI MODE)
# -----------------------------

def init_ee():
    """Initialize the Earth Engine client or raise a clear error."""
    import socket
    
    # Quick network check
    def check_network():
        try:
            socket.create_connection(("www.google.com", 80), timeout=3)
            return True
        except OSError:
            return False
    
    if not check_network():
        raise RuntimeError(
            "No internet connection detected. Google Earth Engine requires internet access. "
            "Please check your network connection and try again."
        )
    
    # Get project ID from Streamlit secrets or environment variable or use default
    project_id = None
    try:
        if hasattr(st, 'secrets') and 'gee_project_id' in st.secrets:
            project_id = st.secrets['gee_project_id']
    except Exception:
        pass  # Secrets file doesn't exist, that's okay
    
    if not project_id:
        project_id = os.getenv('GEE_PROJECT_ID', 'ee-desmondkangah98')
    
    try:
        # Try to initialize with existing credentials first
        ee.Initialize(project=project_id)
    except Exception:
        # If that fails, try authenticating with service account from Streamlit secrets
        try:
            has_secrets = False
            try:
                has_secrets = hasattr(st, 'secrets') and 'gee_service_account' in st.secrets
            except Exception:
                pass  # Secrets file doesn't exist
            
            # Check for Streamlit secrets first (best for deployment)
            if has_secrets:
                service_account_info = dict(st.secrets['gee_service_account'])
                credentials = ee.ServiceAccountCredentials(
                    email=service_account_info.get('client_email'),
                    key_data=json.dumps(service_account_info)
                )
                ee.Initialize(credentials, project=project_id)
            # Fall back to environment variable
            elif os.getenv('GEE_SERVICE_ACCOUNT_FILE'):
                service_account_file = os.getenv('GEE_SERVICE_ACCOUNT_FILE')
                if os.path.exists(service_account_file):
                    credentials = ee.ServiceAccountCredentials(
                        email=None,
                        key_file=service_account_file
                    )
                    ee.Initialize(credentials, project=project_id)
            else:
                # Last resort: try default credentials
                ee.Authenticate()
                ee.Initialize(project=project_id)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not initialize Google Earth Engine. "
                "Setup: (1) Add GEE service account to Streamlit secrets (.streamlit/secrets.toml), or "
                "(2) Set GEE_SERVICE_ACCOUNT_FILE environment variable, or "
                "(3) Run `earthengine authenticate` in terminal. "
                f"Error: {str(e)}"
            ) from e


def mask_s2_clouds(image):
    """Mask clouds in Sentinel-2 imagery using QA60 band."""
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask)


def mask_landsat_clouds(image):
    """Mask clouds in Landsat 8/9 imagery using QA_PIXEL band."""
    qa = image.select('QA_PIXEL')
    # Bit 3: cloud shadow, Bit 4: snow, Bit 5: cloud
    cloud_shadow_bit_mask = 1 << 3
    cloud_bit_mask = 1 << 5
    # Both flags should be set to zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
    )
    return image.updateMask(mask)


def build_planetary_computer_image_for_aoi(aoi_geojson, satellite_type: str, months_back: int = 6):
    """Build an image from Microsoft Planetary Computer over the AOI.
    
    Uses STAC API to search and download Sentinel-2 or Landsat imagery.
    No authentication required!
    """
    import pystac_client
    import planetary_computer
    import stackstac
    from datetime import datetime, timedelta
    
    # Extract geometry
    if isinstance(aoi_geojson, dict) and "geometry" in aoi_geojson:
        geom_dict = aoi_geojson["geometry"]
    else:
        geom_dict = aoi_geojson
    
    # Get bounding box from geometry
    coords = geom_dict["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    bbox_wgs84 = [min(lons), min(lats), max(lons), max(lats)]
    
    # Transform bbox to Web Mercator for stackstac (fixes negative longitude issue)
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    min_x, min_y = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
    max_x, max_y = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])
    bbox_mercator = [min_x, min_y, max_x, max_y]
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    
    print(f"[Planetary Computer] Searching for {satellite_type} imagery...")
    print(f"  AOI bbox (WGS84 lon/lat): {bbox_wgs84}")
    print(f"  AOI bbox (Web Mercator): {bbox_mercator}")
    print(f"  AOI center: lon={sum(lons)/len(lons):.4f}, lat={sum(lats)/len(lats):.4f}")
    print(f"  Date range: {date_range}")
    
    # Connect to Planetary Computer STAC catalog
    try:
        # Try with modifier (newer versions)
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
    except TypeError:
        # Fallback for older versions without modifier support
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )
    
    # Determine collection and bands based on satellite type
    if "Sentinel-2" in satellite_type:
        collection = "sentinel-2-l2a"
        bands = ["B02", "B03", "B04", "B08", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
        scale = 10
    elif "Landsat" in satellite_type:
        if "8" in satellite_type:
            collection = "landsat-c2-l2"
        else:
            collection = "landsat-c2-l2"  # Landsat 9 also in same collection
        bands = ["coastal", "blue", "green", "red", "nir08", "swir16"]
        scale = 30
    else:
        raise ValueError(f"Satellite type '{satellite_type}' not supported for Planetary Computer")
    
    # Search for imagery (use WGS84 bbox for search)
    search = catalog.search(
        collections=[collection],
        bbox=bbox_wgs84,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 20}},  # Less than 20% cloud cover
    )
    
    # Get items (compatible with older pystac-client versions)
    try:
        items = list(search.items())
    except AttributeError:
        # Older versions use get_items() or item_collection()
        try:
            items = list(search.get_items())
        except AttributeError:
            items = search.item_collection()
    
    image_count = len(items)
    
    print(f"[Planetary Computer] Found {image_count} images")
    
    if image_count == 0:
        return None, None, scale, 0
    
    # Sign items for access (required for Planetary Computer)
    items = [planetary_computer.sign(item) for item in items]
    
    # Stack images using stackstac
    print(f"[Planetary Computer] Loading and stacking {min(10, image_count)} least cloudy images...")
    items_sorted = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))[:10]
    
    # Stack images using stackstac with Web Mercator bounds (fixes negative longitude bug)
    stack = stackstac.stack(
        items_sorted,
        assets=bands,
        bounds=bbox_mercator,  # Use Web Mercator bounds instead of bounds_latlon
        epsg=3857,  # Explicitly set to Web Mercator
        resolution=scale,
    )
    
    # Compute median composite
    print(f"[Planetary Computer] Computing median composite...")
    import dask
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        composite = stack.median(dim="time").compute()
    
    # Convert to numpy array (C, H, W) format
    image = composite.values
    
    # Get geospatial metadata from xarray attributes
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    
    # Debug: Print what stackstac actually returned
    print(f"[Planetary Computer] Composite x range: [{composite.x.values[0]:.2f}, {composite.x.values[-1]:.2f}]")
    print(f"[Planetary Computer] Composite y range: [{composite.y.values[0]:.2f}, {composite.y.values[-1]:.2f}]")
    if hasattr(composite, 'crs'):
        print(f"[Planetary Computer] Composite CRS attribute: {composite.crs}")
    
    # Try to get transform from attributes
    if hasattr(composite, 'transform'):
        transform = composite.transform
        print(f"[Planetary Computer] Using transform from composite.transform")
    else:
        # Build transform from coordinates
        x_coords = composite.x.values
        y_coords = composite.y.values
        x_res = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else scale
        y_res = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else -scale
        transform = Affine(x_res, 0, float(x_coords[0]), 0, y_res, float(y_coords[0]))
        print(f"[Planetary Computer] Built transform from coordinates")
    
    print(f"[Planetary Computer] Transform: {transform}")
    
    # Get CRS
    if hasattr(composite, 'crs'):
        crs_str = str(composite.crs)
        print(f"[Planetary Computer] Using CRS from composite: {crs_str}")
    else:
        # Default to Web Mercator
        crs_str = 'EPSG:3857'
        print(f"[Planetary Computer] No CRS in composite, defaulting to: {crs_str}")
    
    # Ensure no NaN values and convert to uint16 for consistency with training data
    image = np.nan_to_num(image, nan=0.0)
    image = np.clip(image, 0, 65535).astype(np.uint16)
    
    profile = {
        'driver': 'GTiff',
        'height': image.shape[1],
        'width': image.shape[2],
        'count': image.shape[0],
        'dtype': 'uint16',
        'crs': crs_str,
        'transform': transform,
    }
    
    # Calculate bounds as BoundingBox
    from rasterio.coords import BoundingBox
    bounds_tuple = rasterio.transform.array_bounds(image.shape[1], image.shape[2], transform)
    bounds = BoundingBox(*bounds_tuple)
    
    print(f"[Planetary Computer] Downloaded image shape: {image.shape}, dtype: {image.dtype}")
    print(f"[Planetary Computer] Value range: [{image.min()}, {image.max()}]")
    print(f"[Planetary Computer] CRS: {crs_str}")
    print(f"[Planetary Computer] Bounds (in {crs_str}): {bounds}")
    
    # Convert bounds to WGS84 to verify location
    if crs_str == 'EPSG:3857':
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
        max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
        print(f"[Planetary Computer] Bounds in WGS84: lon=[{min_lon:.4f}, {max_lon:.4f}], lat=[{min_lat:.4f}, {max_lat:.4f}]")
        print(f"[Planetary Computer] Expected around: lon=[-91.26, -91.01], lat=[31.93, 32.14]")
    
    return image, profile, transform, crs_str, bounds, image_count


def build_gee_image_for_aoi(aoi_geojson, satellite_type: str, months_back: int = 6):
    """Build an ee.Image over the AOI matching the expected 6 input bands.

    Uses recent imagery composite with cloud masking for better quality.
    Adjust bands/date/cloud filters to match how your model was trained.
    """
    init_ee()

    # aoi_geojson can be either a Feature with a 'geometry' field or a bare geometry dict
    if isinstance(aoi_geojson, dict) and "geometry" in aoi_geojson:
        geom_dict = aoi_geojson["geometry"]
    else:
        geom_dict = aoi_geojson

    geom = ee.Geometry(geom_dict)

    # Calculate date range - use recent data for faster processing
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if "Sentinel-2" in satellite_type:
        collection_id = "COPERNICUS/S2_SR_HARMONIZED"
        # Example 6-band stack: B2,B3,B4,B8,B11,B12 (blue, green, red, NIR, SWIR1, SWIR2)
        bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
        scale = 10
        col = (
            ee.ImageCollection(collection_id)
            .filterBounds(geom)
            .filterDate(start_str, end_str)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))  # Stricter cloud filter
            .map(mask_s2_clouds)  # Apply cloud masking
            .select(bands)
        )
        # Use median of cloud-masked images
        image = col.median()
        image_count = col.size().getInfo()

    elif "Landsat 8" in satellite_type or "Landsat 9" in satellite_type:
        # Landsat 8/9 surface reflectance collections
        collection_id = "LANDSAT/LC08/C02/T1_L2" if "8" in satellite_type else "LANDSAT/LC09/C02/T1_L2"
        # Example 6-band stack: coastal, blue, green, red, NIR, SWIR1
        bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"]
        scale = 30
        col = (
            ee.ImageCollection(collection_id)
            .filterBounds(geom)
            .filterDate(start_str, end_str)
            .filter(ee.Filter.lt("CLOUD_COVER", 20))  # Stricter cloud filter
            .map(mask_landsat_clouds)  # Apply cloud masking
            .select(bands)
        )
        # Use median of cloud-masked images
        image = col.median()
        image_count = col.size().getInfo()

    else:
        raise ValueError(f"Satellite type '{satellite_type}' is not yet supported for AOI mode.")

    return image, geom, scale, image_count


def export_gee_image_to_geotiff(ee_image, geom, scale, out_tif, max_pixels=2048):
    """Export an ee.Image to a local GeoTIFF and read it back as numpy + metadata.

    Args:
        max_pixels: Maximum dimension size. Larger images will be downsampled for faster processing.
    """
    import requests

    # Calculate appropriate scale to limit image size
    # Get the bounds to estimate size
    bounds = geom.bounds().getInfo()['coordinates'][0]
    width_deg = abs(bounds[2][0] - bounds[0][0])
    height_deg = abs(bounds[2][1] - bounds[0][1])

    # Estimate pixel dimensions at native scale
    meters_per_degree = 111320  # approximate at equator
    width_m = width_deg * meters_per_degree
    height_m = height_deg * meters_per_degree

    width_px = width_m / scale
    height_px = height_m / scale
    max_dim = max(width_px, height_px)

    # Adjust scale if image would be too large
    if max_dim > max_pixels:
        scale = scale * (max_dim / max_pixels)
        scale = int(scale)
    
    print(f"[export_gee_image_to_geotiff] Downloading at scale={scale}m, estimated size: {int(width_px)}x{int(height_px)} pixels")

    # Get the download URL from Earth Engine with size limits
    url = ee_image.getDownloadURL({
        'scale': scale,
        'region': geom,
        'format': 'GEO_TIFF',
        'maxPixels': 1e8  # 100 million pixel cap
    })

    # Download the image
    response = requests.get(url, stream=True, timeout=300)  # 5 min timeout
    response.raise_for_status()

    # Save to file
    with open(out_tif, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Read and verify the downloaded image
    image, profile, transform, crs, bounds = read_geotiff(out_tif)
    print(f"[export_gee_image_to_geotiff] Downloaded image shape: {image.shape}, dtype: {image.dtype}")
    print(f"[export_gee_image_to_geotiff] Value range: [{image.min()}, {image.max()}]")

    return image, profile, transform, crs, bounds


# -----------------------------
# RASTER IO
# -----------------------------
def read_geotiff(path):
    with rasterio.open(path) as src:
        image = src.read()          # (C, H, W)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    return image, profile, transform, crs, bounds


def save_mask_geotiff(mask, profile, out_path):
    """Save binary mask as a GeoTIFF with same georeferencing as input."""
    prof = profile.copy()
    prof.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(mask.astype(np.uint8), 1)
    return out_path


# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess_tile(tile):
    """
    tile: np.ndarray (C, H, W) in original dtype (uint8/uint16).
    IMPORTANT: Must match the normalization used during training.
    The geoai training code divides by 255 regardless of data type,
    so we must do the same here.
    """
    tile = tile.astype(np.float32)
    # Divide by 255 to match training normalization
    # (even though this is incorrect for Sentinel-2 uint16 data)
    tile = tile / 255.0
    return tile


def postprocess_mask(mask, min_area_pixels=100, kernel_size=3):
    """Post-process binary mask to remove noise and improve quality.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        min_area_pixels: Minimum area in pixels for a water body to keep
        kernel_size: Size of morphological kernel (3, 5, 7, etc.)
    
    Returns:
        Cleaned binary mask
    """
    from scipy import ndimage
    from skimage import morphology
    
    # Morphological closing to fill small holes
    kernel = morphology.disk(kernel_size)
    mask_closed = morphology.binary_closing(mask, kernel)
    
    # Remove small objects (noise)
    mask_cleaned = morphology.remove_small_objects(
        mask_closed.astype(bool), 
        min_size=min_area_pixels
    ).astype(np.uint8)
    
    # Fill holes in remaining water bodies
    mask_filled = ndimage.binary_fill_holes(mask_cleaned).astype(np.uint8)
    
    return mask_filled


# -----------------------------
# TILED PREDICTION FOR LARGE IMAGES
# -----------------------------
def predict_large_image(
    model,
    image,
    device="cpu",
    tile_size=512,
    overlap=64,
):
    """
    image: np.ndarray (C, H, W)
    Returns:
        mask_full: (H, W) binary mask (0/1)
        prob_full: (H, W) mean probability (0–1)
    """

    _, H, W = image.shape

    # pad to multiple of tile_size
    pad_H = (tile_size - H % tile_size) if H % tile_size != 0 else 0
    pad_W = (tile_size - W % tile_size) if W % tile_size != 0 else 0

    image_pad = np.pad(
        image,
        ((0, 0), (0, pad_H), (0, pad_W)),
        mode="constant",
        constant_values=0,
    )
    _, H_pad, W_pad = image_pad.shape

    stride = tile_size - overlap
    prob_sum = np.zeros((H_pad, W_pad), dtype=np.float32)
    count = np.zeros((H_pad, W_pad), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y0 in range(0, H_pad, stride):
            for x0 in range(0, W_pad, stride):
                y1 = min(y0 + tile_size, H_pad)
                x1 = min(x0 + tile_size, W_pad)

                tile = image_pad[:, y0:y1, x0:x1]
                tile = preprocess_tile(tile)

                # add batch dim
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(device)  # (1, C, h, w)

                # forward pass
                out = model(tile_tensor)  # assume (1, 1, h, w) or (1, num_classes, h, w)
                if out.shape[1] > 1:
                    # multi-class: take water class index 1 (adjust if different)
                    out = out[:, 1:2, :, :]

                prob = torch.sigmoid(out).cpu().numpy()[0, 0]  # (h, w)

                prob_sum[y0:y1, x0:x1] += prob
                count[y0:y1, x0:x1] += 1.0

    # avoid division by zero
    count[count == 0] = 1.0
    prob_full = prob_sum / count

    # crop back to original size
    prob_full = prob_full[:H, :W]
    mask_full = (prob_full >= 0.5).astype(np.uint8)
    
    # Debug info
    water_pixels = np.sum(mask_full == 1)
    total_pixels = mask_full.size
    print(f"[predict_large_image] Predicted water pixels: {water_pixels}/{total_pixels} ({100*water_pixels/total_pixels:.2f}%)")
    print(f"[predict_large_image] Probability range: [{prob_full.min():.3f}, {prob_full.max():.3f}]")

    return mask_full, prob_full
# -----------------------------
# MASK → POLYGONS → SHAPEFILE
# -----------------------------

def mask_to_vector(mask, transform, crs):
    """
    Convert binary mask (H, W) into a GeoDataFrame of water polygons with statistics.
    """
    from rasterio.crs import CRS
    
    mask = mask.astype(np.uint8)
    
    # Ensure CRS is a CRS object, not a string
    if isinstance(crs, str):
        crs = CRS.from_string(crs)
    
    # Debug info
    water_pixels = np.sum(mask == 1)
    total_pixels = mask.size
    print(f"[mask_to_vector] Water pixels: {water_pixels}/{total_pixels} ({100*water_pixels/total_pixels:.2f}%)")

    if water_pixels == 0:
        print("[mask_to_vector] No water detected in mask")
        return gpd.GeoDataFrame({"id": [], "area_km2": [], "area_m2": [], "perimeter_m": [], "geometry": []}, crs=crs)

    results = shapes(mask, mask=mask == 1, transform=transform)

    geoms = []
    for geom, value in results:
        if value == 1:
            geoms.append(shape_geom(geom))

    print(f"[mask_to_vector] Generated {len(geoms)} polygons")
    
    if len(geoms) == 0:
        # return empty GDF with CRS
        return gpd.GeoDataFrame({"id": [], "area_km2": [], "area_m2": [], "perimeter_m": [], "geometry": []}, crs=crs)

    # Create a GeoDataFrame with one row per geometry
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs)
    
    # Add water body ID
    gdf["id"] = range(1, len(gdf) + 1)
    
    # Calculate area and perimeter
    # Convert to projected CRS for accurate measurements
    if crs and crs.is_geographic:
        gdf_projected = gdf.to_crs(epsg=3857)  # Web Mercator for area calculation
        gdf["area_m2"] = gdf_projected.geometry.area
        gdf["perimeter_m"] = gdf_projected.geometry.length
    else:
        gdf["area_m2"] = gdf.geometry.area
        gdf["perimeter_m"] = gdf.geometry.length
    
    # Convert area to km²
    gdf["area_km2"] = gdf["area_m2"] / 1_000_000
    
    # Round for display
    gdf["area_km2"] = gdf["area_km2"].round(4)
    gdf["area_m2"] = gdf["area_m2"].round(2)
    gdf["perimeter_m"] = gdf["perimeter_m"].round(2)
    
    # Sort by area (largest first)
    gdf = gdf.sort_values("area_km2", ascending=False).reset_index(drop=True)
    gdf["id"] = range(1, len(gdf) + 1)
    
    return gdf



def export_shapefile_zip(gdf, out_dir, base_name="water_mask"):
    """
    Save GDF as ESRI Shapefile and return path to zipped shapefile.
    """
    os.makedirs(out_dir, exist_ok=True)
    shp_path = os.path.join(out_dir, f"{base_name}.shp")
    gdf.to_file(shp_path, driver="ESRI Shapefile")

    zip_path = os.path.join(out_dir, f"{base_name}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for ext in ["shp", "shx", "dbf", "prj", "cpg"]:
            fpath = os.path.join(out_dir, f"{base_name}.{ext}")
            if os.path.exists(fpath):
                zf.write(fpath, arcname=os.path.basename(fpath))

    return shp_path, zip_path


# -----------------------------
# MINIMAL STREAMLIT UI
# -----------------------------

import streamlit as st
import tempfile
import io
import leafmap.foliumap as leafmap


@st.cache_resource
def get_model(model_path: str, device: str = "cpu"):
    return load_model(model_path, device=device)


def main():
    st.title("🌊 Water Segmentation App")
    
    # Welcome banner
    st.success(
        "Model: U-Net ResNet-34 || " 
        "Upload your satellite imagery to detect water bodies"
    )

    st.sidebar.header("Model Settings")
    model_path = st.sidebar.text_input(
        "Model path", 
        value="best_model.pth"
    )
    device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
    
    # Set default values
    st.session_state["max_pixels"] = 2048
    st.session_state["date_months"] = 6
    st.session_state["min_area"] = 0  # No filtering by default
    st.session_state["morph_kernel"] = 0  # No post-processing by default

    # Basemap selection for maps
    basemap_options = list(getattr(leafmap, "basemaps", {}).keys())
    basemap = None
    if basemap_options:
        default_index = basemap_options.index("OpenStreetMap") if "OpenStreetMap" in basemap_options else 0
        basemap = st.sidebar.selectbox("Basemap", basemap_options, index=default_index)

    # AOI Mode Settings
    st.sidebar.subheader("AOI Mode Settings")
    #st.sidebar.info("Using Microsoft Planetary Computer - No authentication required!")
    
    # Satellite source selection
    satellite_type = st.sidebar.selectbox(
        "Satellite Type",
        ["Sentinel-2", "Landsat 8", "Landsat 9"],
        index=0,
    )
    st.session_state["satellite_type"] = satellite_type

    st.header("1. Upload GeoTIFF (Recommended)")
    uploaded_file = st.file_uploader("Upload a GeoTIFF satellite image", type=["tif", "tiff"])

    # If no image is uploaded, allow the user to select an area of interest (AOI) on a map
    if uploaded_file is None:
        if "aoi_in_progress" not in st.session_state:
            st.session_state["aoi_in_progress"] = False

        st.info("**Recommended:** Upload a .tif/.tiff GeoTIFF file above, or try drawing an AOI below.")

        st.subheader("🌍 Alternative: Select Area of Interest (AOI)")
        st.info(
            "**Powered by Microsoft Planetary Computer** - Fast, reliable, no authentication needed!"
        )
        st.write(f"Satellite Type: **{satellite_type}**")
        aoi_geojson = None

        # If a previous run was started but no results are available, warn the user
        if st.session_state["aoi_in_progress"] and "water_mask_aoi" not in st.session_state:
            st.warning(
                "An AOI detection was previously started and may still be running or was interrupted. "
                "If no results appear after a few minutes, you can click 'Detect Water in AOI' again."
            )
            # Consider previous run stale so the user can start a new one
            st.session_state["aoi_in_progress"] = False
        try:
            # Initial map for AOI selection, with drawing tools enabled
            # draw_control=True adds the Leaflet draw toolbar; draw_export=True allows exporting drawn features
            m_aoi = leafmap.Map(center=[0, 0], zoom=2, draw_control=True, draw_export=True)
            if basemap:
                m_aoi.add_basemap(basemap)

            # Render map in Streamlit; the draw toolbar appears on the map (usually top-left)
            st_component = m_aoi.to_streamlit(height=500, bidirectional=True)

            # Get the last drawn feature (GeoJSON-like object) and cache it
            last_draw = m_aoi.st_last_draw(st_component)
            if last_draw:
                try:
                    # last_draw may already be a dict or a JSON string
                    if isinstance(last_draw, str):
                        aoi_geojson = json.loads(last_draw)
                    else:
                        aoi_geojson = last_draw
                    st.session_state["aoi_geojson"] = aoi_geojson
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Could not parse AOI geometry: {e}")
            elif "aoi_geojson" in st.session_state:
                aoi_geojson = st.session_state["aoi_geojson"]

            st.info("Draw a polygon/rectangle on the map, then click 'Detect Water in AOI'.")
        except Exception as e:
            st.warning(f"Could not display AOI selection map: {e}")

        if st.button("Detect Water in AOI", type="primary"):
            if aoi_geojson is None:
                st.error("Please draw an area of interest on the map first.")
            else:
                if not os.path.exists(model_path):
                    st.error(f"Model file not found: {model_path}")
                else:
                    st.session_state["aoi_in_progress"] = True
                    progress_bar = st.progress(0)
                    progress_bar.progress(5)
                    try:
                        # Get user settings
                        date_months = st.session_state.get("date_months", 6)

                        # Fetch imagery from Planetary Computer
                        with st.spinner(f"🌐 Fetching {satellite_type} imagery from Microsoft Planetary Computer..."):
                            try:
                                result = build_planetary_computer_image_for_aoi(
                                    aoi_geojson, satellite_type, months_back=date_months
                                )

                                if result[5] == 0:  # image_count
                                    st.error(f"No cloud-free {satellite_type} images found for this area. Try a different area or date range.")
                                    return

                                image, profile, transform, crs, bounds, image_count = result
                                st.success(f"✓ Found and downloaded {image_count} images from Planetary Computer")
                                progress_bar.progress(45)

                            except Exception as e:
                                st.error(f"Error fetching from Planetary Computer: {e}")
                                st.info("Try using Upload mode instead - upload a 6-band GeoTIFF above.")
                                return
                        
                        # Save image to temp file - keep in original CRS (like upload method)
                        with tempfile.NamedTemporaryFile(suffix="_aoi.tif", delete=False) as tmp_aoi:
                            aoi_tiff_path = tmp_aoi.name
                        
                        with rasterio.open(aoi_tiff_path, 'w', **profile) as dst:
                            dst.write(image)
                        
                        # Run model inference
                        with st.spinner("🤖 Running water detection model..."):
                            model = get_model(model_path, device=device)
                            progress_bar.progress(60)
                            mask, prob = predict_large_image(model, image, device=device)
                            progress_bar.progress(70)
                        
                        # Convert mask to polygons for visualization and export
                        with st.spinner("🗺️ Converting to vector polygons..."):
                            gdf = mask_to_vector(mask, transform, crs)
                            progress_bar.progress(80)

                        # Save mask GeoTIFF to temp file
                        with tempfile.NamedTemporaryFile(suffix="_mask.tif", delete=False) as tmp_mask:
                            mask_tif_path = tmp_mask.name
                        save_mask_geotiff(mask, profile, mask_tif_path)
                        progress_bar.progress(90)

                        # Store AOI results in session state with polygons ready
                        st.session_state["water_mask_aoi"] = {
                                "mask": mask,
                                "profile": profile,
                                "transform": transform,
                                "crs": crs,
                                "gdf": gdf,
                                "mask_tif_path": mask_tif_path,
                                "aoi_tiff_path": aoi_tiff_path,
                                "bounds": bounds,
                                "image": image,
                                "satellite_type": satellite_type,
                                "image_count": image_count,
                            }

                        progress_bar.progress(100)
                        st.success("AOI detection complete.")
                    finally:
                        progress_bar.empty()
                        st.session_state["aoi_in_progress"] = False

        # If we have AOI results, show map + stats + exports
        if "water_mask_aoi" in st.session_state:
            data = st.session_state["water_mask_aoi"]
            mask = data["mask"]
            gdf = data["gdf"]
            transform = data["transform"]
            crs = data["crs"]
            bounds = data["bounds"]
            aoi_tiff_path = data["aoi_tiff_path"]
            image = data.get("image")
            sat_type = data.get("satellite_type", "Unknown")
            img_count = data.get("image_count", "Unknown")
            
            # Display image quality info
            with st.expander("Satellite Image Quality", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Source:** {sat_type}")
                    st.write(f"**Images in composite:** {img_count}")
                    if image is not None:
                        st.write(f"**Dimensions:** {image.shape[2]} x {image.shape[1]} pixels")
                        st.write(f"**Bands:** {image.shape[0]}")
                with col2:
                    if image is not None:
                        # Calculate coverage (non-zero pixels)
                        valid_pixels = np.count_nonzero(image[0] > 0)
                        total_pixels = image.shape[1] * image.shape[2]
                        coverage = (valid_pixels / total_pixels) * 100
                        st.write(f"**Data coverage:** {coverage:.1f}%")
                        
                        # Show band statistics
                        st.write("**Band value ranges:**")
                        for i in range(min(3, image.shape[0])):
                            band_data = image[i][image[i] > 0]
                            if len(band_data) > 0:
                                st.write(f"  Band {i+1}: {band_data.min():.0f} - {band_data.max():.0f}")
            
            # Interactive map with satellite imagery and water outline
            st.subheader("🗺️ Interactive Map with Water Detection")
            st.info("📍 Blue outlines show detected water bodies on the satellite image")
            
            try:
                # leafmap.Map expects center in WGS84 (lat, lon) regardless of raster CRS
                # If bounds are in Web Mercator (values > 180), convert to WGS84
                center_x = (bounds.left + bounds.right) / 2
                center_y = (bounds.top + bounds.bottom) / 2
                
                # Check if coordinates are in Web Mercator (absolute values > 180)
                if abs(center_x) > 180 or abs(center_y) > 180:
                    # Convert from Web Mercator to WGS84
                    from pyproj import Transformer
                    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
                    center_lon, center_lat = transformer.transform(center_x, center_y)
                else:
                    # Already in lat/lon
                    center_lon, center_lat = center_x, center_y

                m = leafmap.Map(center=[center_lat, center_lon], zoom=10)
                
                # Add satellite image as base layer (same as upload method)
                try:
                    m.add_raster(aoi_tiff_path, layer_name="Satellite Image", opacity=1.0)
                except Exception as e:
                    st.warning(f"Could not add raster to map: {e}. Using basemap instead.")
                    if basemap:
                        m.add_basemap(basemap)

                # Add water detection outlines in bright blue with hover tooltips
                if gdf is not None and len(gdf) > 0:
                    # Keep GDF in same CRS as raster for proper alignment (like upload method)
                    gdf_display = gdf
                    
                    style = {
                        "color": "#00BFFF",      # Bright blue
                        "weight": 4,             # Thicker lines for visibility
                        "fillOpacity": 0.0,      # No fill, just outline
                        "opacity": 1.0,          # Fully opaque
                        "fillColor": "#00BFFF"   # Blue fill color (when hovering)
                    }
                    
                    # Create hover fields for tooltip
                    hover_fields = ["id", "area_km2", "area_m2", "perimeter_m"]
                    hover_aliases = ["Water Body ID", "Area (km²)", "Area (m²)", "Perimeter (m)"]
                    
                    m.add_gdf(
                        gdf_display, 
                        layer_name="Water Bodies (detected)", 
                        style=style,
                        hover_fields=hover_fields,
                        hover_aliases=hover_aliases,
                        info_mode="on_hover"
                    )
                    
                    # Show summary statistics
                    total_area = gdf["area_km2"].sum()
                    largest = gdf["area_km2"].max()
                    st.success(
                        f"✓ Showing {len(gdf)} water bodies outlined in blue | "
                        f"Total: {total_area:.2f} km² | Largest: {largest:.4f} km² | "
                        f"Hover over water bodies to see details"
                    )
                else:
                    st.warning("No water bodies detected in this image")

                m.to_streamlit(height=600)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not display AOI map: {e}")

            st.subheader("Water mask preview (AOI)")
            st.image(mask * 255, caption="Binary water mask (AOI)", use_container_width=True)

            if gdf is not None and len(gdf) > 0:
                st.write(f"Detected {len(gdf)} water polygons.")
                
                # Show statistics table
                with st.expander("Water Bodies Statistics Table", expanded=False):
                    st.write("**All detected water bodies (sorted by area):**")
                    # Display table with selected columns
                    display_df = gdf[["id", "area_km2", "area_m2", "perimeter_m"]].copy()
                    display_df.columns = ["ID", "Area (km²)", "Area (m²)", "Perimeter (m)"]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Water Bodies", len(gdf))
                    with col2:
                        st.metric("Total Area", f"{gdf['area_km2'].sum():.2f} km²")
                    with col3:
                        st.metric("Largest", f"{gdf['area_km2'].max():.4f} km²")
                    with col4:
                        st.metric("Smallest", f"{gdf['area_km2'].min():.4f} km²")
            else:
                st.write("Vector polygons not generated yet. Use Export to generate them.")

            # Compute total water area (approximate, in km²)
            total_area_km2 = None
            if gdf is not None and len(gdf) > 0 and gdf.crs is not None:
                try:
                    if gdf.crs.is_projected:
                        area_m2 = gdf.geometry.area.sum()
                    else:
                        gdf_area = gdf.to_crs(epsg=3857)
                        area_m2 = gdf_area.geometry.area.sum()
                    total_area_km2 = area_m2 / 1e6
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Could not compute water area: {e}")

            if total_area_km2 is not None:
                st.write(f"Total water area: {total_area_km2:,.2f} km²")

            # Export buttons
            st.subheader("2. Export Results (AOI)")

            # Shapefile (zipped)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export as Shapefile (ZIP)", use_container_width=True):
                    if gdf is not None and len(gdf) > 0:
                        with st.spinner("Creating shapefile..."):
                            # Convert to WGS84 for proper export
                            gdf_export = gdf.copy()
                            original_crs = gdf_export.crs.to_epsg() if gdf_export.crs else None
                            if gdf_export.crs and gdf_export.crs.to_epsg() != 4326:
                                gdf_export = gdf_export.to_crs(epsg=4326)
                                st.info(f" Converted from EPSG:{original_crs} to WGS84 (EPSG:4326) for export")
                            with tempfile.TemporaryDirectory() as tmpdir:
                                shp_path, zip_path = export_shapefile_zip(gdf_export, tmpdir, base_name="water_mask_aoi")
                                with open(zip_path, "rb") as f:
                                    shp_bytes = f.read()
                        st.success("Shapefile exported in WGS84 (EPSG:4326) coordinate system")
                        st.download_button(
                            label="📥 Download Shapefile ZIP",
                            data=shp_bytes,
                            file_name="water_mask_aoi_shapefile.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                    else:
                        st.warning("No water bodies detected to export.")

            # KML export
            with col2:
                if st.button("Export as KML", use_container_width=True):
                    if gdf is not None and len(gdf) > 0:
                        with st.spinner("Creating KML..."):
                            # Convert to WGS84 for KML (required by KML spec)
                            gdf_export = gdf.copy()
                            original_crs = gdf_export.crs.to_epsg() if gdf_export.crs else None
                            if gdf_export.crs and gdf_export.crs.to_epsg() != 4326:
                                gdf_export = gdf_export.to_crs(epsg=4326)
                                st.info(f"Converted from EPSG:{original_crs} to WGS84 (EPSG:4326) for KML export")
                            with tempfile.TemporaryDirectory() as tmpdir:
                                kml_path = os.path.join(tmpdir, "water_mask_aoi.kml")
                                gdf_export.to_file(kml_path, driver="KML")
                                with open(kml_path, "rb") as f:
                                    kml_bytes = f.read()
                        st.success("KML exported in WGS84 (EPSG:4326) - compatible with Google Earth")
                        st.download_button(
                            label="📥 Download KML",
                            data=kml_bytes,
                            file_name="water_mask_aoi.kml",
                            mime="application/vnd.google-earth.kml+xml",
                            use_container_width=True,
                        )
                    else:
                        st.warning("No water bodies detected to export.")

            # Clear button
            st.markdown("---")
            if st.button("Clear Results", type="secondary"):
                if "water_mask_aoi" in st.session_state:
                    del st.session_state["water_mask_aoi"]
                if "aoi_geojson" in st.session_state:
                    del st.session_state["aoi_geojson"]
                if "aoi_in_progress" in st.session_state:
                    st.session_state["aoi_in_progress"] = False
                st.rerun()

        # Skip the "uploaded GeoTIFF" workflow when in AOI mode
        return

    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tiff_path = tmp.name

    st.success("Image loaded. You can now run detection.")

    # Load raster (for model and bounds)
    image, profile, transform, crs, bounds = read_geotiff(tiff_path)
    
    # Display image information
    with st.expander("Image Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Dimensions:** {image.shape[2]} x {image.shape[1]} pixels")
            st.write(f"**Bands:** {image.shape[0]}")
            st.write(f"**Data type:** {image.dtype}")
        with col2:
            st.write(f"**CRS:** {crs}")
            area_km2 = ((bounds.right - bounds.left) * (bounds.top - bounds.bottom)) / 1e6
            if crs and not crs.is_geographic:
                # For projected CRS, convert to km²
                area_km2 = ((bounds.right - bounds.left) * (bounds.top - bounds.bottom)) / 1e6
            st.write(f"**Approx. area:** {area_km2:.2f} km²")
            st.write(f"**Bounds:** {bounds}")
    
    # Preview uploaded image - create RGB visualization
    st.subheader("Uploaded Image Preview")
    
    if image.shape[0] >= 3:
        # Create RGB preview
        try:
            # For Sentinel-2/6-band: bands 3,2,1 = R,G,B
            rgb_bands = [min(2, image.shape[0]-1), min(1, image.shape[0]-1), 0]
            rgb = np.stack([image[rgb_bands[0]], image[rgb_bands[1]], image[rgb_bands[2]]], axis=-1)
            
            # Normalize to 0-255
            for i in range(3):
                band = rgb[:, :, i]
                valid = band[band > 0]
                if len(valid) > 0:
                    p2, p98 = np.percentile(valid, (2, 98))
                    rgb[:, :, i] = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
            
            st.image(rgb.astype(np.uint8), caption="RGB Preview (bands shown may vary by sensor)", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create RGB preview: {e}")
    else:
        # Single band grayscale
        try:
            band = image[0]
            valid = band[band > 0]
            if len(valid) > 0:
                p2, p98 = np.percentile(valid, (2, 98))
                band_norm = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                st.image(band_norm.astype(np.uint8), caption="Grayscale Preview", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create preview: {e}")

    # Detection button
    if st.button("Detect Water"):
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return

        with st.spinner("Loading model and running inference..."):
            model = get_model(model_path, device=device)
            mask, prob = predict_large_image(model, image, device=device)
            
            # Convert mask to polygons for visualization and export
            with st.spinner("Converting mask to polygons..."):
                gdf = mask_to_vector(mask, transform, crs)

            # Save mask GeoTIFF to temp file
            with tempfile.NamedTemporaryFile(suffix="_mask.tif", delete=False) as tmp_mask:
                mask_tif_path = tmp_mask.name
            save_mask_geotiff(mask, profile, mask_tif_path)

            # Store in session state with polygons ready
            st.session_state["water_mask"] = {
                "mask": mask,
                "profile": profile,
                "transform": transform,
                "crs": crs,
                "gdf": gdf,
                "mask_tif_path": mask_tif_path,
                "tiff_path": tiff_path,
                "bounds": bounds,
            }

        st.success("Detection complete.")

    if "water_mask" in st.session_state:
        data = st.session_state["water_mask"]
        mask = data["mask"]
        gdf = data["gdf"]
        tiff_path = data["tiff_path"]
        bounds = data["bounds"]
        transform = data["transform"]
        crs = data["crs"]
        
        # Interactive map with satellite imagery and water outline
        st.subheader("🗺️ Interactive Map with Water Detection")
        st.info("📍 Blue outlines show detected water bodies on the satellite image")
        try:
            center_y = (bounds.top + bounds.bottom) / 2
            center_x = (bounds.left + bounds.right) / 2

            m = leafmap.Map(center=[center_y, center_x], zoom=12)
            
            # Add uploaded satellite image as base layer
            try:
                m.add_raster(tiff_path, layer_name="Satellite Image", opacity=1.0)
            except Exception as e:
                st.warning(f"Could not add raster to map: {e}. Using basemap instead.")
                if basemap:
                    m.add_basemap(basemap)

            # Overlay water detection outlines in bright blue with hover tooltips
            if gdf is not None and len(gdf) > 0:
                style = {
                    "color": "#00BFFF",      # Bright blue
                    "weight": 4,             # Thicker lines for visibility  
                    "fillOpacity": 0.0,      # No fill, just outline
                    "opacity": 1.0,          # Fully opaque
                    "fillColor": "#00BFFF"   # Blue fill color (when hovering)
                }
                
                # Create hover fields for tooltip
                hover_fields = ["id", "area_km2", "area_m2", "perimeter_m"]
                hover_aliases = ["Water Body ID", "Area (km²)", "Area (m²)", "Perimeter (m)"]
                
                m.add_gdf(
                    gdf, 
                    layer_name="Water Bodies (detected)", 
                    style=style,
                    hover_fields=hover_fields,
                    hover_aliases=hover_aliases,
                    info_mode="on_hover"
                )
                
                # Show summary statistics
                total_area = gdf["area_km2"].sum()
                largest = gdf["area_km2"].max()
                st.success(
                    f"✓ Showing {len(gdf)} water bodies outlined in blue | "
                    f"Total: {total_area:.2f} km² | Largest: {largest:.4f} km² | "
                    f"Hover over water bodies to see details"
                )
            else:
                st.warning("No water bodies detected in this image")

            m.to_streamlit(height=600)
        except Exception as e:
            st.warning(f"Could not display interactive map: {e}")

        st.subheader("Water mask preview")
        st.image(mask * 255, caption="Binary water mask", use_container_width=True)

        if gdf is not None and len(gdf) > 0:
            st.write(f"Detected {len(gdf)} water polygons.")
            
            # Show statistics table
            with st.expander("Water Bodies Statistics Table", expanded=False):
                st.write("**All detected water bodies (sorted by area):**")
                # Display table with selected columns
                display_df = gdf[["id", "area_km2", "area_m2", "perimeter_m"]].copy()
                display_df.columns = ["ID", "Area (km²)", "Area (m²)", "Perimeter (m)"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Water Bodies", len(gdf))
                with col2:
                    st.metric("Total Area", f"{gdf['area_km2'].sum():.2f} km²")
                with col3:
                    st.metric("Largest", f"{gdf['area_km2'].max():.4f} km²")
                with col4:
                    st.metric("Smallest", f"{gdf['area_km2'].min():.4f} km²")
        else:
            st.write("Vector polygons not generated yet. Use Export to generate them.")

        # Compute total water area (approximate, in km²)
        total_area_km2 = None
        if gdf is not None and len(gdf) > 0 and gdf.crs is not None:
            try:
                if gdf.crs.is_projected:
                    area_m2 = gdf.geometry.area.sum()
                else:
                    gdf_area = gdf.to_crs(epsg=3857)
                    area_m2 = gdf_area.geometry.area.sum()
                total_area_km2 = area_m2 / 1e6
            except Exception as e:
                st.warning(f"Could not compute water area: {e}")

        if total_area_km2 is not None:
            st.write(f"Total water area: {total_area_km2:,.2f} km²")

        # Export buttons
        st.subheader("2. Export Results")

        # Shapefile (zipped) using existing utility
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export as Shapefile (ZIP)", use_container_width=True):
                if gdf is not None and len(gdf) > 0:
                    with st.spinner("Creating shapefile..."):
                        # Convert to WGS84 for proper export
                        gdf_export = gdf.copy()
                        if gdf_export.crs and gdf_export.crs.to_epsg() != 4326:
                            gdf_export = gdf_export.to_crs(epsg=4326)
                        with tempfile.TemporaryDirectory() as tmpdir:
                            shp_path, zip_path = export_shapefile_zip(gdf_export, tmpdir, base_name="water_mask")
                            with open(zip_path, "rb") as f:
                                shp_bytes = f.read()
                    st.download_button(
                        label="📥 Download Shapefile ZIP",
                        data=shp_bytes,
                        file_name="water_mask_shapefile.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )
                else:
                    st.warning("No water bodies detected to export.")

        # KML export
        with col2:
            if st.button("Export as KML", use_container_width=True):
                if gdf is not None and len(gdf) > 0:
                    with st.spinner("Creating KML..."):
                        # Convert to WGS84 for KML (required by KML spec)
                        gdf_export = gdf.copy()
                        if gdf_export.crs and gdf_export.crs.to_epsg() != 4326:
                            gdf_export = gdf_export.to_crs(epsg=4326)
                        with tempfile.TemporaryDirectory() as tmpdir:
                            kml_path = os.path.join(tmpdir, "water_mask.kml")
                            # GeoPandas can write KML if the driver is available in GDAL/OGR
                            gdf_export.to_file(kml_path, driver="KML")
                            with open(kml_path, "rb") as f:
                                kml_bytes = f.read()
                    st.download_button(
                        label="📥 Download KML",
                        data=kml_bytes,
                        file_name="water_mask.kml",
                        mime="application/vnd.google-earth.kml+xml",
                        use_container_width=True,
                    )
                else:
                    st.warning("No water bodies detected to export.")

        # Clear button
        st.markdown("---")
        if st.button("Clear Results", type="secondary"):
            if "water_mask" in st.session_state:
                del st.session_state["water_mask"]
            st.rerun()


if __name__ == "__main__":
    main()
