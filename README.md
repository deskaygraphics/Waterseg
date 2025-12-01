# Water Segmentation Application - Deployment Package

This folder contains everything needed to run the water segmentation web application.

## 📁 Contents

```
water_segmentation_deployment/
├── app.py                    # Main Streamlit application
├── best_model.pth           # Trained model weights (97.9 MB, 90% IoU)
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── report_materials/       # Training metrics and documentation
    ├── README.md
    ├── TRAINING_METRICS_SUMMARY.md
    ├── TRAINING_DETAILS.md
    ├── test_result.png
    ├── actual_training_curves.png
    ├── actual_metrics_table.png
    ├── learning_curves.png
    ├── training_metrics_summary.png
    └── water_segmentation_training.ipynb
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Key Dependencies
- streamlit
- torch
- segmentation-models-pytorch
- rasterio
- geopandas
- leafmap
- numpy
- matplotlib
- pyproj
- planetary-computer
- pystac-client
- stackstac

(All dependencies are in `requirements.txt`)

## 🎯 Features

### 1. Upload Mode
- Upload your own GeoTIFF satellite imagery (6-band recommended)
- Supports any coordinate system
- Instant water detection

### 2. AOI Drawing Mode
- Draw an area of interest on an interactive map
- Automatically fetches satellite imagery from Microsoft Planetary Computer
- Supports Sentinel-2, Landsat 8, and Landsat 9
- No authentication required!

### 3. Interactive Results
- View detected water bodies on an interactive map
- Statistics for each water body (area, perimeter)
- Export results as Shapefile or KML

## 🔧 Model Information

- **Architecture**: U-Net with ResNet-34 Encoder
- **Performance**: 90% IoU on test set
- **Input**: 6-band multi-spectral satellite imagery
- **Bands**: Blue, Green, Red, NIR, SWIR1, SWIR2
- **Resolution**: 10m (Sentinel-2)

## 📊 Report Materials

The `report_materials/` folder contains:
- Training metrics and learning curves
- Test result visualizations
- Detailed documentation
- Performance analysis

Use these for academic reports or presentations.

## 🌍 Use Cases

- **Flood Mapping**: Rapid assessment of inundated areas
- **Lake Monitoring**: Track changes in lake extent
- **River Network Mapping**: Detect and map river systems
- **Wetland Identification**: Identify vegetated water areas
- **Reservoir Tracking**: Monitor reservoir water levels
- **Agricultural Water Management**: Map irrigation systems

## 💡 Usage Tips

### For Best Results:
1. Use recent cloud-free imagery
2. Select areas with visible water bodies
3. Use 6-band multi-spectral data when uploading
4. For AOI mode, select smaller regions for faster processing

### Supported Satellites:
- Sentinel-2 (10m resolution, 6 bands)
- Landsat 8 (30m resolution)
- Landsat 9 (30m resolution)

## 🐛 Troubleshooting

### Issue: App won't start
**Solution**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Model file not found
**Solution**: Ensure `best_model.pth` is in the same directory as `app.py`

### Issue: Out of memory
**Solution**: The app automatically handles large images through tiling. If issues persist, try:
- Processing smaller areas
- Closing other applications
- Using a machine with more RAM

### Issue: No satellite data found (AOI mode)
**Solution**: 
- Try a different time period (increase months back)
- Select a different area
- Check your internet connection
- Try the Upload mode instead

## 📦 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Production Deployment

**Streamlit Cloud:**
1. Push to GitHub repository
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Deploy directly from repo

**Docker:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Heroku/AWS/GCP:**
- Include `setup.sh` and `Procfile` for Heroku
- Use container services for AWS/GCP

## 🔒 Security Notes

- The app uses Microsoft Planetary Computer (no API key needed)
- No user data is stored or transmitted
- All processing happens locally or via public APIs
- Model weights are fixed (no user-trainable components)

## 📝 License

This application is for educational and research purposes.

## 🤝 Citation

If you use this application in your research, please cite:

```
Water Segmentation Application using U-Net Deep Learning
Model: U-Net with ResNet-34 Encoder
Performance: 90% IoU on test dataset
Dataset: Sentinel-2 multi-spectral imagery
```

## 📧 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the documentation in `report_materials/`
3. Ensure all dependencies are correctly installed

## 🔄 Version History

- **v1.0** (November 2024)
  - Initial release
  - 90% IoU performance
  - Dual mode: Upload + AOI drawing
  - Export to Shapefile and KML
  - Interactive mapping with Leafmap
  - Fixed coordinate system handling for global coverage

## 🎓 Technical Details

### Model Training
- Trained on 1,977 tiles from Sentinel-2 imagery
- 80/20 train/validation split
- 2 epochs with transfer learning
- Best validation IoU: 85.9%
- Test set IoU: 90.0%

### Processing Pipeline
1. Image acquisition (upload or Planetary Computer)
2. Preprocessing (normalization, tiling)
3. Model inference (tiled prediction with overlap)
4. Post-processing (morphological operations)
5. Vectorization (raster to polygon conversion)
6. Export (coordinate transformation to WGS84)

### Coordinate System Handling
- Internal processing: Web Mercator (EPSG:3857) or original CRS
- Exports: Always WGS84 (EPSG:4326) for compatibility
- Special handling for Western Hemisphere (negative longitudes)

---

**Built with ❤️ using PyTorch, Streamlit, and Open Source Geospatial Tools**

**Ready to Deploy! 🚀**
