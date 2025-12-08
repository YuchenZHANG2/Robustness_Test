from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import threading
import torch

# Import our custom modules
from model_loader import ModelLoader, MODEL_CONFIGS
from evaluator import COCOEvaluator, format_coco_label_mapping
from visualization import visualize_predictions, fig_to_base64
from batch_optimized_pipeline import BatchOptimizedRobustnessTest

# Global variable to track test progress
test_progress = {
    'status': 'idle',  # idle, running, completed, error
    'progress': 0,
    'message': '',
    'results_ready': False
}

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/validation', exist_ok=True)

# Initialize model loader
model_loader = ModelLoader()

# Dataset configurations
DATASET_CONFIG = {
    'COCO': {
        'annotation_file': '/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json',
        'image_dir': '/home/yuchen/YuchenZ/Datasets/coco/val2017',
        'filter_classes': None,  # Use all classes
        'class_mapping': None  # No mapping needed (COCO labels are standard)
    },
    'Construction': {
        'annotation_file': '/home/yuchen/YuchenZ/lab/Detector_test/DustyConstruction.v2i.coco/_annotations.coco.json',
        'image_dir': '/home/yuchen/YuchenZ/lab/Detector_test/DustyConstruction.v2i.coco/train',
        'filter_classes': [3],  # Only evaluate "person" class (id=3 in Construction dataset)
        'class_mapping': {3: 1}  # Map Construction's person (id=3) to COCO's person (id=1)
    }
}

# Global evaluator (will be set by user selection)
evaluator = None

# ============================================================================
# DEBUG: Set to True to test with ALL images, False for random 50 images
# ============================================================================
USE_ALL_IMAGES = False  # Change to True for full dataset testing
# ============================================================================

# Predefined detector models
# Maps display name -> (model_config_key, full_name)
PREDEFINED_DETECTORS = {
    'frcnn_v2': 'Faster R-CNN V2',
    'retinanet_v2': 'RetinaNet V2',
    'fcos_v1': 'FCOS',
    'ssd300': 'SSD 300',
    'yolov11': 'YOLO11',
    'detr': 'DETR',
    'rt_detr': 'RT-DETR',
}

# Additional models (shown under "More Models" button)
ADDITIONAL_DETECTORS = {
    'frcnn_v1': 'Faster R-CNN V1',
    'frcnn_mobilenet_large': 'Faster R-CNN MobileNet Large',
    'frcnn_mobilenet_320': 'Faster R-CNN MobileNet 320',
    'retinanet_v1': 'RetinaNet V1',
    'ssdlite_320': 'SSDLite320 MobileNet',
    'deformable_detr': 'Deformable DETR',
    'conditional_detr': 'Conditional DETR',
    'dab_detr': 'DAB-DETR',
}

# Corruption categories
CORRUPTIONS = {
    'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'Blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'Weather': ['snow', 'frost', 'fog', 'brightness', 'dust'],
    'Digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
}

# Dataset-specific corruptions
DATASET_SPECIFIC_CORRUPTIONS = {
    'dust': ['Construction']  # dust is only available for Construction dataset
}


def get_selected_corruptions(form_data):
    """Extract selected corruptions from form data."""
    selected = []
    for category, corruption_list in CORRUPTIONS.items():
        for corruption in corruption_list:
            if form_data.get(corruption):
                selected.append(corruption)
    return selected


@app.route('/')
def index():
    return redirect(url_for('step1'))


@app.route('/step1', methods=['GET', 'POST'])
def step1():
    """Step 1/3: Detector Selection"""
    if request.method == 'POST':
        # Get selected predefined detectors
        selected_detectors = request.form.getlist('detectors')
        
        # Handle custom HuggingFace detector
        custom_detector_hf = request.form.get('custom_detector_hf', '').strip()
        
        # Store in session
        session['selected_detectors'] = selected_detectors
        session['custom_detector_hf'] = custom_detector_hf if custom_detector_hf else None
        
        return redirect(url_for('step2'))
    
    return render_template('step1.html', 
                         detectors=PREDEFINED_DETECTORS,
                         additional_detectors=ADDITIONAL_DETECTORS)


@app.route('/step2', methods=['GET', 'POST'])
def step2():
    """Step 2/3: Dataset and Corruption Selection"""
    global evaluator
    
    if request.method == 'POST':
        action = request.form.get('action')
        selected_dataset = request.form.get('dataset')
        
        if action == 'back':
            return redirect(url_for('step1'))
        
        # Collect selected corruptions
        selected_corruptions = get_selected_corruptions(request.form)
        session['selected_corruptions'] = selected_corruptions
        
        if action == 'preview':
            # Store selected dataset and initialize evaluator for preview
            session['selected_dataset'] = selected_dataset
            
            dataset_config = DATASET_CONFIG[selected_dataset]
            evaluator = COCOEvaluator(
                annotation_file=dataset_config['annotation_file'],
                image_dir=dataset_config['image_dir'],
                filter_classes=dataset_config['filter_classes'],
                class_mapping=dataset_config['class_mapping']
            )
            
            return redirect(url_for('interactive_preview'))
        
        elif action == 'next':
            # Store selected dataset and update evaluator
            session['selected_dataset'] = selected_dataset
            
            dataset_config = DATASET_CONFIG[selected_dataset]
            evaluator = COCOEvaluator(
                annotation_file=dataset_config['annotation_file'],
                image_dir=dataset_config['image_dir'],
                filter_classes=dataset_config['filter_classes'],
                class_mapping=dataset_config['class_mapping']
            )
            return redirect(url_for('step3'))
    
    # Get previously selected values from session (for GET requests or back navigation)
    selected_dataset = session.get('selected_dataset', None)
    selected_corruptions = session.get('selected_corruptions', [])
    
    return render_template('step2.html', 
                         corruptions=CORRUPTIONS,
                         dataset_specific_corruptions=DATASET_SPECIFIC_CORRUPTIONS,
                         datasets=DATASET_CONFIG, 
                         selected_dataset=selected_dataset,
                         selected_corruptions=selected_corruptions,
                         show_preview=False)


@app.route('/step3', methods=['GET', 'POST'])
def step3():
    """Step 3/3: Report Options"""
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'back':
            return redirect(url_for('step2'))
        elif action == 'start':
            # Get report preference
            generate_pdf = request.form.get('report_type') == 'pdf'
            session['generate_pdf'] = generate_pdf
            return redirect(url_for('results'))
    
    # Get summary of selections
    detector_keys = session.get('selected_detectors', [])
    # Map keys to human-readable names
    detector_names = [PREDEFINED_DETECTORS.get(key, key) for key in detector_keys]
    custom_detector_hf = session.get('custom_detector_hf')
    corruptions = session.get('selected_corruptions', [])
    
    return render_template('step3.html', 
                         detectors=detector_names,
                         custom_detector_hf=custom_detector_hf,
                         corruptions=corruptions)


@app.route('/results')
def results():
    """Start the testing process"""
    model_keys = session.get('selected_detectors', [])
    custom_detector_hf = session.get('custom_detector_hf')
    corruptions = session.get('selected_corruptions', [])
    generate_pdf = session.get('generate_pdf', False)
    
    if not model_keys and not custom_detector_hf:
        return redirect(url_for('step1'))
    
    # Get images based on USE_ALL_IMAGES debug flag
    if USE_ALL_IMAGES:
        image_ids = evaluator.get_all_images()
        print(f"DEBUG: Using ALL {len(image_ids)} images from dataset")
    else:
        image_ids = evaluator.get_random_images(n=50)
        print(f"DEBUG: Using random sample of {len(image_ids)} images")
    
    session['test_image_ids'] = image_ids
    
    # If custom detector exists, show optional validation, otherwise go straight to testing
    if custom_detector_hf:
        return render_template('validate_model.html',
                             has_custom_detector=True,
                             custom_detector_hf=custom_detector_hf)
    else:
        return redirect(url_for('run_testing'))


@app.route('/load_and_predict_custom')
def load_and_predict_custom():
    """Load custom HuggingFace model and make prediction on sample image"""
    try:
        custom_detector_hf = session.get('custom_detector_hf')
        if not custom_detector_hf:
            return jsonify({'success': False, 'error': 'No custom detector configured'}), 400
        
        # Add custom model to MODEL_CONFIGS temporarily
        custom_key = 'custom_hf_preview'
        # Auto-detect if it's an Ultralytics/YOLO model
        # Check for .pt extension (local YOLO files) or no "/" (e.g., "yolo11n.pt")
        # HuggingFace models always have "/" in the path (e.g., "hustvl/yolos-small")
        is_ultralytics = (
            custom_detector_hf.endswith('.pt') or 
            ('/' not in custom_detector_hf and 'yolo' in custom_detector_hf.lower())
        )
        
        if is_ultralytics:
            MODEL_CONFIGS[custom_key] = {
                "name": f"Custom Ultralytics: {custom_detector_hf}",
                "type": "ultralytics",
                "model_name": custom_detector_hf
            }
        else:
            MODEL_CONFIGS[custom_key] = {
                "name": f"Custom HuggingFace: {custom_detector_hf}",
                "type": "huggingface",
                "hf_model_id": custom_detector_hf
            }
        
        model_loader.load_model(custom_key)
        
        # Get a random test image
        image_ids = session.get('test_image_ids', [])
        if not image_ids:
            image_ids = evaluator.get_random_images(n=50)
            session['test_image_ids'] = image_ids
        
        # Use first image for validation
        sample_image_id = image_ids[0]
        img_path = evaluator.get_image_path(sample_image_id)
        image = Image.open(img_path).convert('RGB')
        
        # Get predictions
        predictions = model_loader.predict(custom_key, image, score_threshold=0.3)
        
        # Visualize
        fig = visualize_predictions(
            image, predictions, format_coco_label_mapping(),
            score_threshold=0.3,
            title=f"{custom_detector_hf} - Sample Prediction"
        )
        
        # Convert to base64 for web display
        img_base64 = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}',
            'num_detections': len(predictions['boxes']),
            'model_name': custom_detector_hf
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/run_testing')
def run_testing():
    """Run the actual robustness testing"""
    return render_template('run_testing.html')


@app.route('/execute_test')
def execute_test():
    """Execute the robustness test in background thread"""
    model_keys = session.get('selected_detectors', [])
    custom_detector_hf = session.get('custom_detector_hf')
    corruptions = session.get('selected_corruptions', [])
    image_ids = session.get('test_image_ids', [])
    
    # Add custom HuggingFace model to MODEL_CONFIGS if provided
    if custom_detector_hf:
        custom_key = 'custom_hf'
        # Auto-detect if it's an Ultralytics/YOLO model
        # Check for .pt extension (local YOLO files) or no "/" (e.g., "yolo11n.pt")
        # HuggingFace models always have "/" in the path (e.g., "hustvl/yolos-small")
        is_ultralytics = (
            custom_detector_hf.endswith('.pt') or 
            ('/' not in custom_detector_hf and 'yolo' in custom_detector_hf.lower())
        )
        
        if is_ultralytics:
            MODEL_CONFIGS[custom_key] = {
                "name": f"Custom Ultralytics: {custom_detector_hf}",
                "type": "ultralytics",
                "model_name": custom_detector_hf
            }
        else:
            MODEL_CONFIGS[custom_key] = {
                "name": f"Custom HuggingFace: {custom_detector_hf}",
                "type": "huggingface",
                "hf_model_id": custom_detector_hf
            }
        model_keys.append(custom_key)
    
    def run_test_background():
        """Background function to run the test"""
        global test_progress
        try:
            test_progress.update({
                'status': 'running',
                'progress': 0,
                'message': 'Initializing test...'
            })
            
            test = BatchOptimizedRobustnessTest(
                model_loader, evaluator,
                batch_size=4, num_workers=2
            )
            
            def progress_callback(current, total, message):
                global test_progress
                test_progress['progress'] = int((current / total) * 100)
                test_progress['message'] = message
            
            results = test.run_full_test(
                model_keys=model_keys,
                corruption_names=corruptions,
                image_ids=image_ids,
                severities=[1, 2, 3, 4, 5],
                progress_callback=progress_callback
            )
            
            test.save_results('static/test_results.json')
            
            test_progress.update({
                'status': 'completed',
                'progress': 100,
                'message': 'Test completed successfully!',
                'results_ready': True
            })
            
        except Exception as e:
            test_progress.update({
                'status': 'error',
                'message': str(e)
            })
            import traceback
            traceback.print_exc()
    
    thread = threading.Thread(target=run_test_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Test started in background'})


@app.route('/test_progress')
def get_test_progress():
    """Get current test progress"""
    global test_progress
    return jsonify(test_progress)


@app.route('/show_results')
def show_results():
    """Display final results"""
    import json
    
    with open('static/test_results.json', 'r') as f:
        results = json.load(f)
    
    # Sort results by clean mAP in descending order
    sorted_results = dict(sorted(
        results.items(),
        key=lambda x: x[1].get('clean', {}).get('mAP', 0),
        reverse=True
    ))
    
    # Generate plot data for each corruption
    plot_data = generate_corruption_plots(sorted_results)
    
    # Get category names from evaluator
    category_names = evaluator.get_category_names() if evaluator else {}
    
    return render_template('show_results.html', 
                         results=sorted_results, 
                         plot_data=plot_data,
                         category_names=json.dumps(category_names))


def generate_corruption_plots(results):
    """Generate data for corruption line plots"""
    import json
    
    # Collect all corruptions
    all_corruptions = set()
    for model_data in results.values():
        all_corruptions.update(model_data.get('corrupted', {}).keys())
    
    plot_data = {}
    
    for corruption in all_corruptions:
        plot_data[corruption] = {
            'labels': [0, 1, 2, 3, 4, 5],  # Severity levels (0 = clean)
            'datasets': []
        }
        
        for model_key, model_data in results.items():
            model_name = model_data['name']
            
            if corruption in model_data.get('corrupted', {}):
                severities_data = model_data['corrupted'][corruption]
                mAP_values = []
                
                # Add clean mAP as severity 0
                clean_mAP = model_data.get('clean', {}).get('mAP', None)
                mAP_values.append(clean_mAP)
                
                # Add corrupted severities 1-5
                for severity in [1, 2, 3, 4, 5]:
                    severity_key = str(severity)
                    if severity_key in severities_data:
                        mAP_values.append(severities_data[severity_key]['mAP'])
                    else:
                        mAP_values.append(None)
                
                plot_data[corruption]['datasets'].append({
                    'label': model_name,
                    'data': mAP_values
                })
    
    return json.dumps(plot_data)


@app.route('/preview_corruption')
def preview_corruption():
    """Generate corruption preview image"""
    from corruption_preview import generate_preview
    
    corruptions = session.get('selected_corruptions', [])
    if not corruptions:
        return jsonify({'error': 'No corruptions selected'}), 400
    
    # Generate preview image
    preview_path = generate_preview(corruptions)
    
    return jsonify({'preview_url': url_for('static', filename=preview_path)})


@app.route('/interactive_preview')
def interactive_preview():
    """Interactive corruption preview page"""
    print(f"DEBUG: interactive_preview called")
    print(f"DEBUG: evaluator = {evaluator}")
    print(f"DEBUG: session = {dict(session)}")
    
    corruptions = session.get('selected_corruptions', [])
    if not corruptions:
        print("ERROR: No corruptions in session, redirecting to step2")
        return redirect(url_for('step2'))
    
    if evaluator is None:
        print("ERROR: evaluator is None, redirecting to step2")
        return redirect(url_for('step2'))
    
    print(f"DEBUG: Rendering interactive_preview with {len(corruptions)} corruptions")
    return render_template('interactive_preview.html', corruptions=corruptions)



@app.route('/api/random_image')
def api_random_image():
    """Get a random image from the selected dataset"""
    print(f"DEBUG: api_random_image called, evaluator is: {evaluator}")
    
    if evaluator is None:
        print("ERROR: evaluator is None")
        return jsonify({'error': 'No dataset selected. Please go back to Step 2 and select a dataset.'}), 400
    
    try:
        # Get a random image
        image_ids = evaluator.get_random_images(n=1)
        if not image_ids:
            print("ERROR: No images available")
            return jsonify({'error': 'No images available'}), 400
        
        image_id = image_ids[0]
        image_info = evaluator.coco_gt.loadImgs(image_id)[0]
        
        print(f"DEBUG: Returning image {image_id}: {image_info['file_name']}")
        
        return jsonify({
            'image_id': image_id,
            'image_name': image_info['file_name']
        })
    except Exception as e:
        print(f"ERROR in api_random_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/apply_corruption')
def api_apply_corruption():
    """Apply corruption to an image and return the result"""
    from torch_corruptions import corrupt
    from io import BytesIO
    import cv2
    import numpy as np
    
    if evaluator is None:
        return jsonify({'error': 'No dataset selected'}), 400
    
    image_id = request.args.get('image_id', type=int)
    corruption = request.args.get('corruption')
    severity = request.args.get('severity', type=int)
    
    if image_id is None or corruption is None or severity is None:
        return jsonify({'error': 'Missing parameters'}), 400
    
    # Load image
    image_info = evaluator.coco_gt.loadImgs(image_id)[0]
    image_path = os.path.join(evaluator.image_dir, image_info['file_name'])
    
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply corruption if severity > 0
        if severity > 0:
            # Special handling for dust corruption
            if corruption == 'dust':
                # For dust, we need to load the corresponding dusty image from test folder
                from pathlib import Path
                dust_dir = Path(evaluator.image_dir).parent / 'test'
                
                if dust_dir.exists():
                    # Get clean image filename and extract first 7 digits as prefix
                    clean_filename = image_info['file_name']
                    prefix = clean_filename[:7]
                    
                    # Find matching dusty image
                    matching_files = list(dust_dir.glob(f"{prefix}*"))
                    
                    if matching_files:
                        # Load dusty image
                        dust_path = matching_files[0]
                        dust_img = cv2.imread(str(dust_path))
                        dust_img = cv2.cvtColor(dust_img, cv2.COLOR_BGR2RGB)
                        
                        # Resize dust image to match clean image size
                        clean_array = np.array(image)
                        dust_img_resized = cv2.resize(dust_img, (clean_array.shape[1], clean_array.shape[0]))
                        
                        # Blend based on severity
                        # severity 1: 20% dust, severity 5: 100% dust
                        alpha_values = [0.2, 0.4, 0.6, 0.8, 1.0]
                        alpha = alpha_values[severity - 1]
                        beta = 1.0 - alpha
                        
                        # Blend images
                        blended = cv2.addWeighted(dust_img_resized, alpha, clean_array, beta, 0)
                        corrupted_image = Image.fromarray(blended.astype('uint8'))
                    else:
                        # No matching dust image found, return clean image
                        print(f"Warning: No dust image found for {prefix}")
                        corrupted_image = image
                else:
                    # Dust directory doesn't exist, return clean image
                    print(f"Warning: Dust directory not found at {dust_dir}")
                    corrupted_image = image
            else:
                # Apply standard corruption
                import numpy as np
                img_array = np.array(image)
                
                # Apply corruption
                corrupted_array = corrupt(img_array, corruption, severity=severity)
                
                # Convert back to PIL
                corrupted_image = Image.fromarray(corrupted_array.astype('uint8'))
        else:
            # Severity 0 means clean image
            corrupted_image = image
        
        # Convert to bytes
        img_io = BytesIO()
        corrupted_image.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        
        from flask import send_file
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        print(f"Error applying corruption: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

