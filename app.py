from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import random
from PIL import Image
import threading
import time

# Import our custom modules
from model_loader import ModelLoader, MODEL_CONFIGS
from evaluator import COCOEvaluator, format_coco_label_mapping
from visualization import visualize_predictions, fig_to_base64
from batch_optimized_pipeline import BatchOptimizedRobustnessTest
import torch

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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/validation', exist_ok=True)

# Initialize model loader and evaluator
model_loader = ModelLoader()
evaluator = COCOEvaluator(
    annotation_file='/home/yuchen/YuchenZ/Datasets/coco/annotations/instances_val2017.json',
    image_dir='/home/yuchen/YuchenZ/Datasets/coco/val2017'
)

# Predefined detector models
# Maps display name -> (model_config_key, full_name)
PREDEFINED_DETECTORS = {
    'frcnn_v2': 'Faster R-CNN',
    'retinanet_v2': 'RetinaNet',
    'fcos_v1': 'FCOS',
    'detr': 'DETR',
    'rt_detr': 'RT-DETR',
}

# Corruption categories
CORRUPTIONS = {
    'Noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'Blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'Weather': ['snow', 'frost', 'fog', 'brightness'],
    'Digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
}


@app.route('/')
def index():
    return redirect(url_for('step1'))


@app.route('/step1', methods=['GET', 'POST'])
def step1():
    """Step 1/3: Detector Selection"""
    if request.method == 'POST':
        # Get selected predefined detectors
        selected_detectors = request.form.getlist('detectors')
        
        # Handle custom detector upload
        custom_detector = None
        if 'custom_detector' in request.files:
            file = request.files['custom_detector']
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                custom_detector = filepath
        
        # Store in session
        session['selected_detectors'] = selected_detectors
        session['custom_detector'] = custom_detector
        
        return redirect(url_for('step2'))
    
    return render_template('step1.html', detectors=PREDEFINED_DETECTORS)


@app.route('/step2', methods=['GET', 'POST'])
def step2():
    """Step 2/3: Corruption Selection with Preview"""
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'back':
            return redirect(url_for('step1'))
        elif action == 'preview':
            # Get selected corruptions for preview
            selected_corruptions = []
            for category, corruption_list in CORRUPTIONS.items():
                for corruption in corruption_list:
                    if request.form.get(corruption):
                        selected_corruptions.append(corruption)
            
            session['selected_corruptions'] = selected_corruptions
            # Return to same page to show preview
            return render_template('step2.html', 
                                 corruptions=CORRUPTIONS,
                                 selected_corruptions=selected_corruptions,
                                 show_preview=True)
        elif action == 'next':
            # Store corruptions and move to next step
            selected_corruptions = []
            for category, corruption_list in CORRUPTIONS.items():
                for corruption in corruption_list:
                    if request.form.get(corruption):
                        selected_corruptions.append(corruption)
            
            session['selected_corruptions'] = selected_corruptions
            return redirect(url_for('step3'))
    
    return render_template('step2.html', corruptions=CORRUPTIONS, show_preview=False)


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
            
            # Here you would start the actual testing process
            # For now, redirect to a results page
            return redirect(url_for('results'))
    
    # Get summary of selections
    detector_keys = session.get('selected_detectors', [])
    # Map keys to human-readable names
    detector_names = [PREDEFINED_DETECTORS.get(key, key) for key in detector_keys]
    custom_detector = session.get('custom_detector')
    corruptions = session.get('selected_corruptions', [])
    
    return render_template('step3.html', 
                         detectors=detector_names,
                         custom_detector=custom_detector,
                         corruptions=corruptions)


@app.route('/results')
def results():
    """Start the testing process and show validation"""
    model_keys = session.get('selected_detectors', [])
    corruptions = session.get('selected_corruptions', [])
    generate_pdf = session.get('generate_pdf', False)
    
    if not model_keys:
        return redirect(url_for('step1'))
    
    # Get 50 random images for testing
    image_ids = evaluator.get_random_images(n=50)
    session['test_image_ids'] = image_ids
    session['current_validation_model_idx'] = 0
    session['validation_approved'] = []
    
    return redirect(url_for('validate_model'))


@app.route('/validate_model')
def validate_model():
    """Show prediction validation for current model"""
    model_keys = session.get('selected_detectors', [])
    current_idx = session.get('current_validation_model_idx', 0)
    
    if current_idx >= len(model_keys):
        # All models validated, start testing
        return redirect(url_for('run_testing'))
    
    model_key = model_keys[current_idx]
    model_name = MODEL_CONFIGS[model_key]['name']
    
    return render_template('validate_model.html',
                         model_key=model_key,
                         model_name=model_name,
                         current=current_idx + 1,
                         total=len(model_keys))


@app.route('/load_and_predict/<model_key>')
def load_and_predict(model_key):
    """Load model and make prediction on sample image"""
    try:
        # Load model with progress tracking
        def progress_callback(step, total, message):
            # In production, use websockets for real-time updates
            pass
        
        model_loader.load_model(model_key, progress_callback=progress_callback)
        
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
        predictions = model_loader.predict(model_key, image, score_threshold=0.3)
        
        # Get category names
        category_names = format_coco_label_mapping()
        
        # Visualize
        fig = visualize_predictions(
            image, predictions, category_names,
            score_threshold=0.3,
            title=f"{MODEL_CONFIGS[model_key]['name']} - Sample Prediction"
        )
        
        # Convert to base64 for web display
        img_base64 = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}',
            'num_detections': len(predictions['boxes']),
            'model_name': MODEL_CONFIGS[model_key]['name']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/approve_model/<model_key>', methods=['POST'])
def approve_model(model_key):
    """User approves the model predictions"""
    approved = session.get('validation_approved', [])
    approved.append(model_key)
    session['validation_approved'] = approved
    
    # Move to next model
    current_idx = session.get('current_validation_model_idx', 0)
    session['current_validation_model_idx'] = current_idx + 1
    
    return jsonify({'success': True})


@app.route('/run_testing')
def run_testing():
    """Run the actual robustness testing"""
    return render_template('run_testing.html')


@app.route('/execute_test')
def execute_test():
    """Execute the robustness test in background thread"""
    model_keys = session.get('selected_detectors', [])
    corruptions = session.get('selected_corruptions', [])
    image_ids = session.get('test_image_ids', [])
    
    def run_test_background():
        """Background function to run the test"""
        global test_progress
        try:
            test_progress['status'] = 'running'
            test_progress['progress'] = 0
            test_progress['message'] = 'Initializing test...'
            
            # Use optimized batch processing
            print("Using optimized batch processing with DataLoader")
            test = BatchOptimizedRobustnessTest(
                model_loader, 
                evaluator, 
                batch_size=4,  # Conservative for single GPU
                num_workers=2   # Parallel data loading
            )
            
            # Progress callback
            def progress_callback(current, total, message):
                global test_progress
                test_progress['progress'] = int((current / total) * 100)
                test_progress['message'] = message
                print(f"[PROGRESS] {current}/{total} ({test_progress['progress']}%) - {message}")
                import sys
                sys.stdout.flush()
            
            # Run tests
            results = test.run_full_test(
                model_keys=model_keys,
                corruption_names=corruptions,
                image_ids=image_ids,
                severities=[1, 2, 3, 4, 5],
                progress_callback=progress_callback
            )
            
            # Save results
            test.save_results('static/test_results.json')
            
            test_progress['status'] = 'completed'
            test_progress['progress'] = 100
            test_progress['message'] = 'Test completed successfully!'
            test_progress['results_ready'] = True
            
        except Exception as e:
            test_progress['status'] = 'error'
            test_progress['message'] = str(e)
            print(f"Error in test: {e}")
            import traceback
            traceback.print_exc()
    
    # Start background thread
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
    
    # Generate plot data for each corruption
    plot_data = generate_corruption_plots(results)
    
    return render_template('show_results.html', results=results, plot_data=plot_data)


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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
