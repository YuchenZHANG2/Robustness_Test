from flask import Flask, render_template, request, jsonify, send_file, url_for
import cv2
import numpy as np
from pathlib import Path
import base64
import io
from PIL import Image
import json
import os
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('uploads/temp')
app.config['OUTPUT_FOLDER'] = Path('matched_crops')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure directories exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(parents=True, exist_ok=True)

# Global storage for session data (in production, use proper session management)
session_data = {
    'folder1_path': None,
    'folder2_path': None,
    'image_pairs': [],
    'current_pair_index': 0,
    'match_result': None,
    'alignment_result': None
}


def match_image_pair(img1_path, img2_path, method='sift'):
    """Match two images using feature detection."""
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        raise ValueError("Could not read one or both images")
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if method.lower() == 'sift':
        detector = cv2.SIFT_create()
    elif method.lower() == 'orb':
        detector = cv2.ORB_create(nfeatures=2000)
    elif method.lower() == 'akaze':
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    
    if method.lower() == 'orb':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    result = {
        'num_matches': len(good_matches),
        'total_keypoints_img1': len(kp1),
        'total_keypoints_img2': len(kp2),
    }
    
    if len(kp1) > 0 and len(kp2) > 0:
        result['match_score'] = len(good_matches) / min(len(kp1), len(kp2))
    else:
        result['match_score'] = 0.0
    
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        result['homography'] = H.tolist() if H is not None else None
        result['inliers'] = int(mask.sum()) if mask is not None else 0
    else:
        result['homography'] = None
        result['inliers'] = 0
    
    return result


def align_and_crop_images(img1_path, img2_path, homography):
    """Align two images and find their largest overlapping region."""
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    H = np.array(homography)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    aligned_img1 = cv2.warpPerspective(img1, H, (w2, h2))
    
    mask1 = cv2.warpPerspective(np.ones((h1, w1), dtype=np.uint8) * 255, H, (w2, h2))
    mask2 = np.ones((h2, w2), dtype=np.uint8) * 255
    
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    
    coords = cv2.findNonZero(overlap_mask)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    
    cropped_img1 = aligned_img1[y:y+h, x:x+w]
    cropped_img2 = img2[y:y+h, x:x+w]
    
    return {
        'aligned_img1': aligned_img1,
        'img2': img2,
        'crop_coords': (x, y, w, h),
        'cropped_img1': cropped_img1,
        'cropped_img2': cropped_img2,
    }


def numpy_to_base64(img):
    """Convert numpy array image to base64 string."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


@app.route('/')
def index():
    return render_template('image_matcher.html')


def find_matching_pairs(folder1, folder2):
    """Find image pairs with matching names across two folders."""
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    
    if not folder1_path.exists() or not folder2_path.exists():
        raise ValueError("One or both folders do not exist")
    
    # Get all image files from folder1
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    folder1_images = {f.name: f for f in folder1_path.iterdir() 
                     if f.is_file() and f.suffix.lower() in image_extensions}
    
    # Find matching images in folder2
    pairs = []
    for img_name, img1_path in sorted(folder1_images.items()):
        img2_path = folder2_path / img_name
        if img2_path.exists():
            pairs.append({
                'name': img_name,
                'img1_path': str(img1_path),
                'img2_path': str(img2_path)
            })
    
    return pairs


@app.route('/set_folders', methods=['POST'])
def set_folders():
    """Set the two folders to match images from."""
    data = request.json
    folder1 = data.get('folder1')
    folder2 = data.get('folder2')
    
    if not folder1 or not folder2:
        return jsonify({'error': 'Both folder paths must be provided'}), 400
    
    try:
        pairs = find_matching_pairs(folder1, folder2)
        
        if not pairs:
            return jsonify({'error': 'No matching image pairs found in the folders'}), 400
        
        session_data['folder1_path'] = folder1
        session_data['folder2_path'] = folder2
        session_data['image_pairs'] = pairs
        session_data['current_pair_index'] = 0
        
        return jsonify({
            'success': True,
            'num_pairs': len(pairs),
            'pairs': [p['name'] for p in pairs],
            'current_pair': pairs[0]['name']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload_folders', methods=['POST'])
def upload_folders():
    """Handle folder uploads from browser."""
    print("\n" + "="*60)
    print("UPLOAD FOLDERS ENDPOINT CALLED")
    print("="*60)
    
    print(f"Request files keys: {list(request.files.keys())}")
    
    if 'folder1_files' not in request.files or 'folder2_files' not in request.files:
        print("ERROR: Missing folder files in request")
        return jsonify({'error': 'Both folders must be uploaded'}), 400
    
    folder1_files = request.files.getlist('folder1_files')
    folder2_files = request.files.getlist('folder2_files')
    
    print(f"Folder 1 files count: {len(folder1_files)}")
    print(f"Folder 2 files count: {len(folder2_files)}")
    
    if not folder1_files or not folder2_files:
        print("ERROR: One or both folders are empty")
        return jsonify({'error': 'Both folders must contain files'}), 400
    
    try:
        # Create temporary directories for uploaded files
        upload_dir1 = app.config['UPLOAD_FOLDER'] / 'folder1'
        upload_dir2 = app.config['UPLOAD_FOLDER'] / 'folder2'
        
        print(f"Upload dir 1: {upload_dir1}")
        print(f"Upload dir 2: {upload_dir2}")
        
        # Clear existing files
        if upload_dir1.exists():
            print(f"Removing existing folder1 directory")
            shutil.rmtree(upload_dir1)
        if upload_dir2.exists():
            print(f"Removing existing folder2 directory")
            shutil.rmtree(upload_dir2)
        
        upload_dir1.mkdir(parents=True, exist_ok=True)
        upload_dir2.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        saved_count1 = 0
        for file in folder1_files:
            if file.filename and Path(file.filename).suffix.lower() in image_extensions:
                filename = Path(file.filename).name
                filepath = upload_dir1 / filename
                file.save(str(filepath))
                saved_count1 += 1
                print(f"  Saved to folder1: {filename}")
        
        print(f"Total files saved to folder1: {saved_count1}")
        
        saved_count2 = 0
        for file in folder2_files:
            if file.filename and Path(file.filename).suffix.lower() in image_extensions:
                filename = Path(file.filename).name
                filepath = upload_dir2 / filename
                file.save(str(filepath))
                saved_count2 += 1
                print(f"  Saved to folder2: {filename}")
        
        print(f"Total files saved to folder2: {saved_count2}")
        
        # Find matching pairs
        print("Finding matching pairs...")
        pairs = find_matching_pairs(upload_dir1, upload_dir2)
        
        print(f"Found {len(pairs)} matching pairs")
        
        if not pairs:
            print("ERROR: No matching pairs found")
            return jsonify({'error': 'No matching image pairs found in the uploaded folders'}), 400
        
        # Filter out pairs that already exist in matched_crops
        output_dir1 = app.config['OUTPUT_FOLDER'] / 'cropped_img1'
        output_dir2 = app.config['OUTPUT_FOLDER'] / 'cropped_img2'
        
        filtered_pairs = []
        skipped_pairs = []
        
        for pair in pairs:
            base_name = Path(pair['name']).stem
            ext = Path(pair['name']).suffix
            
            out1 = output_dir1 / f"{base_name}_cropped{ext}"
            out2 = output_dir2 / f"{base_name}_cropped{ext}"
            
            # Check if both cropped files already exist
            if out1.exists() and out2.exists():
                skipped_pairs.append(pair['name'])
                print(f"  Skipping {pair['name']} - already processed")
            else:
                filtered_pairs.append(pair)
        
        print(f"Filtered pairs: {len(filtered_pairs)} new, {len(skipped_pairs)} skipped")
        
        if not filtered_pairs:
            return jsonify({
                'error': f'All {len(pairs)} matching pairs have already been processed',
                'skipped_count': len(skipped_pairs)
            }), 400
        
        session_data['folder1_path'] = str(upload_dir1)
        session_data['folder2_path'] = str(upload_dir2)
        session_data['image_pairs'] = filtered_pairs
        session_data['current_pair_index'] = 0
        
        print("SUCCESS: Returning response")
        print("="*60 + "\n")
        
        return jsonify({
            'success': True,
            'num_pairs': len(filtered_pairs),
            'pairs': [p['name'] for p in filtered_pairs],
            'current_pair': filtered_pairs[0]['name'],
            'skipped_count': len(skipped_pairs)
        })
    
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/get_current_pair', methods=['GET'])
def get_current_pair():
    """Get information about the current pair."""
    if not session_data['image_pairs']:
        return jsonify({'error': 'No image pairs loaded'}), 400
    
    idx = session_data['current_pair_index']
    pair = session_data['image_pairs'][idx]
    
    return jsonify({
        'success': True,
        'index': idx,
        'total': len(session_data['image_pairs']),
        'name': pair['name'],
        'img1_path': pair['img1_path'],
        'img2_path': pair['img2_path']
    })


@app.route('/navigate_pair', methods=['POST'])
def navigate_pair():
    """Navigate to next or previous pair."""
    if not session_data['image_pairs']:
        return jsonify({'error': 'No image pairs loaded'}), 400
    
    direction = request.json.get('direction')  # 'next' or 'prev'
    current_idx = session_data['current_pair_index']
    total = len(session_data['image_pairs'])
    
    if direction == 'next':
        new_idx = (current_idx + 1) % total
    elif direction == 'prev':
        new_idx = (current_idx - 1) % total
    else:
        return jsonify({'error': 'Invalid direction'}), 400
    
    session_data['current_pair_index'] = new_idx
    session_data['match_result'] = None
    session_data['alignment_result'] = None
    
    pair = session_data['image_pairs'][new_idx]
    
    return jsonify({
        'success': True,
        'index': new_idx,
        'total': total,
        'name': pair['name']
    })


@app.route('/match', methods=['POST'])
def match():
    """Match the current pair of images."""
    if not session_data['image_pairs']:
        return jsonify({'error': 'No image pairs loaded'}), 400
    
    try:
        idx = session_data['current_pair_index']
        pair = session_data['image_pairs'][idx]
        
        # Match images
        match_result = match_image_pair(
            pair['img1_path'],
            pair['img2_path'],
            method='sift'
        )
        
        if match_result['num_matches'] < 4:
            return jsonify({
                'error': f"Only {match_result['num_matches']} matches found. Need at least 4 for alignment."
            }), 400
        
        # Align and crop
        alignment_result = align_and_crop_images(
            pair['img1_path'],
            pair['img2_path'],
            match_result['homography']
        )
        
        if alignment_result is None:
            return jsonify({'error': 'Could not find overlapping region'}), 400
        
        # Store results (without large image arrays)
        session_data['match_result'] = {
            'num_matches': match_result['num_matches'],
            'match_score': match_result['match_score'],
            'inliers': match_result['inliers'],
            'homography': match_result['homography']
        }
        session_data['alignment_result'] = {
            'crop_coords': alignment_result['crop_coords']
        }
        
        # Create side-by-side preview
        side_by_side = np.hstack([
            alignment_result['cropped_img1'],
            alignment_result['cropped_img2']
        ])
        
        return jsonify({
            'success': True,
            'num_matches': match_result['num_matches'],
            'match_score': round(match_result['match_score'], 3),
            'inliers': match_result['inliers'],
            'overlap_width': alignment_result['crop_coords'][2],
            'overlap_height': alignment_result['crop_coords'][3],
            'preview': numpy_to_base64(side_by_side)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_view', methods=['POST'])
def get_view():
    """Get current view (overlay or side-by-side)."""
    if not session_data.get('alignment_result'):
        return jsonify({'error': 'No matched images available'}), 400
    
    try:
        show_overlay = request.json.get('show_overlay', False)
        alpha = float(request.json.get('alpha', 0.5))
        
        idx = session_data['current_pair_index']
        pair = session_data['image_pairs'][idx]
        
        # Reload images and crop
        alignment_result = align_and_crop_images(
            pair['img1_path'],
            pair['img2_path'],
            session_data['match_result']['homography']
        )
        
        cropped1 = alignment_result['cropped_img1']
        cropped2 = alignment_result['cropped_img2']
        
        if show_overlay:
            # Fixed overlay logic: alpha controls img1 opacity, (1-alpha) controls img2 opacity
            # When alpha = 0: img1 is transparent (0), img2 is opaque (1)
            # When alpha = 1: img1 is opaque (1), img2 is transparent (0)
            display_img = cv2.addWeighted(cropped1, alpha, cropped2, 1 - alpha, 0)
        else:
            # Side by side
            display_img = np.hstack([cropped1, cropped2])
        
        return jsonify({
            'success': True,
            'image': numpy_to_base64(display_img)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_cropped', methods=['POST'])
def save_cropped():
    """Save the cropped overlapping regions."""
    if not session_data.get('alignment_result'):
        return jsonify({'error': 'No matched images available'}), 400
    
    try:
        idx = session_data['current_pair_index']
        pair = session_data['image_pairs'][idx]
        
        # Reload and crop images
        alignment_result = align_and_crop_images(
            pair['img1_path'],
            pair['img2_path'],
            session_data['match_result']['homography']
        )
        
        # Create output directories
        dir1 = app.config['OUTPUT_FOLDER'] / 'cropped_img1'
        dir2 = app.config['OUTPUT_FOLDER'] / 'cropped_img2'
        dir1.mkdir(parents=True, exist_ok=True)
        dir2.mkdir(parents=True, exist_ok=True)
        
        # Use original filename
        base_name = Path(pair['name']).stem
        ext = Path(pair['name']).suffix
        
        out1 = dir1 / f"{base_name}_cropped{ext}"
        out2 = dir2 / f"{base_name}_cropped{ext}"
        
        # Save images
        cv2.imwrite(str(out1), alignment_result['cropped_img1'])
        cv2.imwrite(str(out2), alignment_result['cropped_img2'])
        
        return jsonify({
            'success': True,
            'path1': str(out1),
            'path2': str(out2),
            'base_dir': str(app.config['OUTPUT_FOLDER'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save_all_cropped', methods=['POST'])
def save_all_cropped():
    """Save cropped regions for all matched pairs."""
    if not session_data['image_pairs']:
        return jsonify({'error': 'No image pairs loaded'}), 400
    
    try:
        # Create output directories
        dir1 = app.config['OUTPUT_FOLDER'] / 'cropped_img1'
        dir2 = app.config['OUTPUT_FOLDER'] / 'cropped_img2'
        dir1.mkdir(parents=True, exist_ok=True)
        dir2.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        errors = []
        
        for idx, pair in enumerate(session_data['image_pairs']):
            try:
                # Match images
                match_result = match_image_pair(
                    pair['img1_path'],
                    pair['img2_path'],
                    method='sift'
                )
                
                if match_result['num_matches'] < 4:
                    errors.append(f"{pair['name']}: insufficient matches ({match_result['num_matches']})")
                    continue
                
                # Align and crop
                alignment_result = align_and_crop_images(
                    pair['img1_path'],
                    pair['img2_path'],
                    match_result['homography']
                )
                
                if alignment_result is None:
                    errors.append(f"{pair['name']}: no overlap found")
                    continue
                
                # Save
                base_name = Path(pair['name']).stem
                ext = Path(pair['name']).suffix
                
                out1 = dir1 / f"{base_name}_cropped{ext}"
                out2 = dir2 / f"{base_name}_cropped{ext}"
                
                cv2.imwrite(str(out1), alignment_result['cropped_img1'])
                cv2.imwrite(str(out2), alignment_result['cropped_img2'])
                
                saved_count += 1
                
            except Exception as e:
                errors.append(f"{pair['name']}: {str(e)}")
        
        return jsonify({
            'success': True,
            'saved_count': saved_count,
            'total_pairs': len(session_data['image_pairs']),
            'errors': errors,
            'base_dir': str(app.config['OUTPUT_FOLDER'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Image Matcher Web Application")
    print("=" * 60)
    print("\nStarting server...")
    print("Access the application at: http://localhost:5001")
    print("Or from remote: http://<your-server-ip>:5001")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
