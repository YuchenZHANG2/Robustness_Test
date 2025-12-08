#!/bin/bash
# Script to prepare the app for Hugging Face Spaces deployment

echo "🚀 Preparing Detector Robustness Test for Hugging Face Spaces..."

# Check if Hugging Face space directory is provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide the path to your cloned Hugging Face Space"
    echo "Usage: ./prepare_deployment.sh /path/to/your-space"
    echo ""
    echo "First, create and clone your space:"
    echo "  1. Go to https://huggingface.co/spaces"
    echo "  2. Create a new Space with Docker SDK"
    echo "  3. Clone it: git clone https://huggingface.co/spaces/YOUR_USERNAME/your-space"
    echo "  4. Run this script: ./prepare_deployment.sh /path/to/your-space"
    exit 1
fi

SPACE_DIR="$1"

# Check if directory exists
if [ ! -d "$SPACE_DIR" ]; then
    echo "❌ Error: Directory $SPACE_DIR does not exist"
    exit 1
fi

echo "📂 Target directory: $SPACE_DIR"

# Copy essential Python files
echo "📝 Copying Python files..."
cp app.py "$SPACE_DIR/"
cp model_loader.py "$SPACE_DIR/"
cp evaluator.py "$SPACE_DIR/"
cp visualization.py "$SPACE_DIR/"
cp batch_optimized_pipeline.py "$SPACE_DIR/"
cp torch_corruptions.py "$SPACE_DIR/"

# Copy configuration files
echo "⚙️  Copying configuration files..."
cp requirements.txt "$SPACE_DIR/"
cp Dockerfile "$SPACE_DIR/"
cp .dockerignore "$SPACE_DIR/"
cp README.md "$SPACE_DIR/"

# Copy templates and static directories
echo "🎨 Copying templates and static files..."
cp -r templates "$SPACE_DIR/"
cp -r static "$SPACE_DIR/"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p "$SPACE_DIR/uploads/temp/folder1"
mkdir -p "$SPACE_DIR/uploads/temp/folder2"
mkdir -p "$SPACE_DIR/static/validation"
mkdir -p "$SPACE_DIR/static/previews"
mkdir -p "$SPACE_DIR/matched_crops"

# Handle datasets
echo ""
echo "📊 Dataset handling:"
echo "⚠️  Large datasets (COCO) should NOT be included in the Docker image"
echo ""
echo "Options:"
echo "  1. Download datasets on first run (recommended for COCO)"
echo "  2. Use Git LFS for medium-sized datasets (Construction dataset)"
echo "  3. Host datasets externally and download at startup"
echo ""
read -p "Do you want to copy the Construction dataset? (y/n): " copy_dataset

if [ "$copy_dataset" = "y" ]; then
    echo "📦 Copying Construction dataset..."
    cp -r DustyConstruction.v2i.coco "$SPACE_DIR/"
    
    echo "Setting up Git LFS..."
    cd "$SPACE_DIR"
    git lfs install
    git lfs track "DustyConstruction.v2i.coco/**"
    cd -
fi

# Copy model files if they exist (optional)
echo ""
read -p "Do you have YOLO model files to copy? (y/n): " copy_models

if [ "$copy_models" = "y" ]; then
    echo "📦 Copying model files..."
    [ -f yolo11m.pt ] && cp yolo11m.pt "$SPACE_DIR/"
    [ -f yolo11n.pt ] && cp yolo11n.pt "$SPACE_DIR/"
    [ -f yolov8m.pt ] && cp yolov8m.pt "$SPACE_DIR/"
    
    cd "$SPACE_DIR"
    git lfs track "*.pt"
    cd -
fi

echo ""
echo "✅ Preparation complete!"
echo ""
echo "📋 Next steps:"
echo "  1. cd $SPACE_DIR"
echo "  2. Review and edit dataset paths in app.py if needed"
echo "  3. git add ."
echo "  4. git commit -m 'Initial deployment'"
echo "  5. git push"
echo ""
echo "  6. Go to your Space on Hugging Face"
echo "  7. Enable GPU in Settings (recommended: T4 or A10G)"
echo "  8. Wait for build to complete"
echo ""
echo "🌐 Your app will be available at:"
echo "   https://huggingface.co/spaces/YOUR_USERNAME/your-space"
echo ""
