import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


def match_image_pair(img1_path, img2_path, method='sift', save_visualization=None):
    """
    Match two images using feature detection and matching.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        method: Feature detection method ('sift', 'orb', or 'akaze')
        save_visualization: Path to save the match visualization (None to skip)
    
    Returns:
        Dictionary containing:
        - num_matches: Number of good matches found
        - match_score: Quality score of the match
        - homography: Transformation matrix (if enough matches found)
        - visualization_path: Path where visualization was saved (if save_visualization provided)
    """
    # Read images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        raise ValueError("Could not read one or both images")
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Select feature detector
    if method.lower() == 'sift':
        detector = cv2.SIFT_create()
    elif method.lower() == 'orb':
        detector = cv2.ORB_create(nfeatures=2000)
    elif method.lower() == 'akaze':
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Detect keypoints and compute descriptors
    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)
    
    # Match descriptors
    if method.lower() == 'orb':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test (Lowe's ratio test)
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
    
    # Calculate match score
    if len(kp1) > 0 and len(kp2) > 0:
        result['match_score'] = len(good_matches) / min(len(kp1), len(kp2))
    else:
        result['match_score'] = 0.0
    
    # Compute homography if enough matches
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        result['homography'] = H
        result['inliers'] = int(mask.sum()) if mask is not None else 0
    else:
        result['homography'] = None
        result['inliers'] = 0
    
    # Save visualization if path provided
    if save_visualization:
        matched_img = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        # Create directory if it doesn't exist
        save_path = Path(save_visualization)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_visualization), matched_img)
        result['visualization_path'] = str(save_visualization)
    
    return result


def batch_match_images(image_pairs, method='sift', output_dir=None):
    """
    Match multiple pairs of images.
    
    Args:
        image_pairs: List of tuples (img1_path, img2_path)
        method: Feature detection method
        output_dir: Directory to save match visualizations (optional)
    
    Returns:
        List of results for each pair
    """
    results = []
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    for idx, (img1_path, img2_path) in enumerate(image_pairs):
        print(f"Processing pair {idx + 1}/{len(image_pairs)}: {img1_path} <-> {img2_path}")
        
        try:
            # Prepare visualization path if output_dir is specified
            viz_path = None
            if output_dir:
                viz_path = output_dir / f"match_{idx:03d}.jpg"
            
            result = match_image_pair(img1_path, img2_path, method=method, save_visualization=viz_path)
            result['img1_path'] = str(img1_path)
            result['img2_path'] = str(img2_path)
            
            print(f"  Found {result['num_matches']} good matches (score: {result['match_score']:.3f})")
            
            results.append(result)
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'img1_path': str(img1_path),
                'img2_path': str(img2_path),
                'error': str(e)
            })
    
    return results


# Example usage
if __name__ == "__main__":
    # Example 1: Match a single pair of images
    print("Example 1: Matching a single pair")
    print("-" * 50)
    
    # Replace these with your actual image paths
    img1_path = "1.jpg"
    img2_path = "2.jpg"
    
    # Uncomment to test with actual images:
    result = match_image_pair(img1_path, img2_path, method='sift', save_visualization='match_result.jpg')
    print(f"Matches found: {result['num_matches']}")
    print(f"Match score: {result['match_score']:.3f}")
    if result['homography'] is not None:
        print(f"Inliers: {result['inliers']}")
    if 'visualization_path' in result:
        print(f"Visualization saved to: {result['visualization_path']}")
    
    # # Example 2: Batch process multiple pairs
    # print("\nExample 2: Batch processing")
    # print("-" * 50)
    
    # # Define your image pairs
    # image_pairs = [
    #     # ("path/to/img1_a.jpg", "path/to/img1_b.jpg"),
    #     # ("path/to/img2_a.jpg", "path/to/img2_b.jpg"),
    #     # ("path/to/img3_a.jpg", "path/to/img3_b.jpg"),
    # ]
    
    # # Uncomment to test with actual images:
    # # results = batch_match_images(image_pairs, method='sift', output_dir='match_results')
    # # 
    # # # Print summary
    # # print("\nSummary:")
    # # for r in results:
    # #     if 'error' not in r:
    # #         print(f"{r['img1_path']} <-> {r['img2_path']}: {r['num_matches']} matches")
    
    # print("\nTo use this code:")
    # print("1. Replace the example image paths with your actual image paths")
    # print("2. Uncomment the example code you want to run")
    # print("3. Choose a method: 'sift' (best quality), 'orb' (fastest), or 'akaze' (good balance)")
    # print("4. Run: python test.py")




def align_and_crop_images(img1, img2, homography):
    """
    Align two images using homography and find their largest overlapping region.
    
    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
        homography: Homography matrix from img1 to img2
    
    Returns:
        Dictionary containing:
        - aligned_img1: Image 1 warped to align with image 2
        - img2: Original image 2
        - crop_coords: (x, y, w, h) of the overlapping region
        - cropped_img1: Cropped region from aligned image 1
        - cropped_img2: Cropped region from image 2
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Warp img1 to align with img2
    aligned_img1 = cv2.warpPerspective(img1, homography, (w2, h2))
    
    # Create masks to find overlap
    mask1 = cv2.warpPerspective(np.ones((h1, w1), dtype=np.uint8) * 255, homography, (w2, h2))
    mask2 = np.ones((h2, w2), dtype=np.uint8) * 255
    
    # Find overlap
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    
    # Find bounding box of overlap
    coords = cv2.findNonZero(overlap_mask)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop both images to overlap region
    cropped_img1 = aligned_img1[y:y+h, x:x+w]
    cropped_img2 = img2[y:y+h, x:x+w]
    
    return {
        'aligned_img1': aligned_img1,
        'img2': img2,
        'crop_coords': (x, y, w, h),
        'cropped_img1': cropped_img1,
        'cropped_img2': cropped_img2,
        'overlap_mask': overlap_mask
    }


class ImageMatcherGUI:
    """
    GUI application for viewing and comparing image pairs with overlay visualization.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Image Matcher & Overlay Viewer")
        self.root.geometry("1200x800")
        
        self.img1_path = None
        self.img2_path = None
        self.img1 = None
        self.img2 = None
        self.match_result = None
        self.alignment_result = None
        self.show_overlay = False
        self.overlay_alpha = 0.5
        
        self.setup_ui()
    
    def setup_ui(self):
        # Control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # File selection
        ttk.Button(control_frame, text="Load Image 1", command=self.load_image1).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Image 2", command=self.load_image2).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Match Images", command=self.match_images).pack(side=tk.LEFT, padx=5)
        
        # Overlay controls
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.overlay_btn = ttk.Button(control_frame, text="Show Overlay", command=self.toggle_overlay, state=tk.DISABLED)
        self.overlay_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Opacity:").pack(side=tk.LEFT, padx=5)
        self.alpha_scale = ttk.Scale(control_frame, from_=0, to=1, orient=tk.HORIZONTAL, 
                                     command=self.update_alpha, length=150)
        self.alpha_scale.set(0.5)
        self.alpha_scale.pack(side=tk.LEFT, padx=5)
        
        # Save controls
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.save_btn = ttk.Button(control_frame, text="Save Cropped Images", command=self.save_cropped, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Info label
        info_frame = ttk.Frame(self.root, padding="5")
        info_frame.pack(side=tk.TOP, fill=tk.X)
        self.info_label = ttk.Label(info_frame, text="Load two images to begin", foreground="blue")
        self.info_label.pack(side=tk.LEFT)
        
        # Canvas for image display
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_label = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image1(self):
        path = filedialog.askopenfilename(
            title="Select Image 1",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if path:
            self.img1_path = path
            self.img1 = cv2.imread(path)
            self.status_label.config(text=f"Image 1 loaded: {Path(path).name}")
            self.check_ready()
    
    def load_image2(self):
        path = filedialog.askopenfilename(
            title="Select Image 2",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if path:
            self.img2_path = path
            self.img2 = cv2.imread(path)
            self.status_label.config(text=f"Image 2 loaded: {Path(path).name}")
            self.check_ready()
    
    def check_ready(self):
        if self.img1 is not None and self.img2 is not None:
            self.info_label.config(text="Both images loaded. Click 'Match Images' to proceed.", foreground="green")
    
    def match_images(self):
        if self.img1 is None or self.img2 is None:
            messagebox.showerror("Error", "Please load both images first!")
            return
        
        self.status_label.config(text="Matching images...")
        self.root.update()
        
        try:
            # Match images
            self.match_result = match_image_pair(self.img1_path, self.img2_path, method='sift')
            
            if self.match_result['num_matches'] < 4:
                messagebox.showwarning("Warning", 
                    f"Only {self.match_result['num_matches']} matches found. Need at least 4 for alignment.")
                self.status_label.config(text="Insufficient matches for alignment")
                return
            
            # Align and crop
            self.alignment_result = align_and_crop_images(
                self.img1, self.img2, self.match_result['homography']
            )
            
            if self.alignment_result is None:
                messagebox.showerror("Error", "Could not find overlapping region")
                return
            
            # Update UI
            self.overlay_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.info_label.config(
                text=f"Found {self.match_result['num_matches']} matches | "
                     f"Overlap region: {self.alignment_result['crop_coords'][2]}x{self.alignment_result['crop_coords'][3]}px",
                foreground="green"
            )
            self.status_label.config(text="Images matched successfully")
            
            # Display the cropped overlap
            self.show_overlay = False
            self.overlay_btn.config(text="Show Overlay")
            self.display_current_view()
            
        except Exception as e:
            messagebox.showerror("Error", f"Matching failed: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")
    
    def toggle_overlay(self):
        self.show_overlay = not self.show_overlay
        if self.show_overlay:
            self.overlay_btn.config(text="Hide Overlay")
        else:
            self.overlay_btn.config(text="Show Overlay")
        self.display_current_view()
    
    def update_alpha(self, value):
        self.overlay_alpha = float(value)
        if self.show_overlay:
            self.display_current_view()
    
    def display_current_view(self):
        if self.alignment_result is None:
            return
        
        cropped1 = self.alignment_result['cropped_img1']
        cropped2 = self.alignment_result['cropped_img2']
        
        if self.show_overlay:
            # Create overlay
            display_img = cv2.addWeighted(cropped1, self.overlay_alpha, cropped2, 1 - self.overlay_alpha, 0)
        else:
            # Show side by side
            display_img = np.hstack([cropped1, cropped2])
        
        # Convert to RGB for display
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        h, w = display_img.shape[:2]
        scale = min(canvas_width / w, canvas_height / h, 1.0)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL and display
        img_pil = Image.fromarray(display_img)
        self.photo = ImageTk.PhotoImage(img_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)
    
    def save_cropped(self):
        if self.alignment_result is None:
            messagebox.showerror("Error", "No aligned images to save!")
            return
        
        # Ask user to select output directories
        result = messagebox.askyesno(
            "Save Cropped Images",
            "Save the cropped overlapping regions of both images?\n\n"
            "They will be saved in separate folders:\n"
            "- cropped_img1/\n"
            "- cropped_img2/"
        )
        
        if not result:
            return
        
        try:
            # Create output directories
            base_dir = Path.cwd() / "matched_crops"
            dir1 = base_dir / "cropped_img1"
            dir2 = base_dir / "cropped_img2"
            dir1.mkdir(parents=True, exist_ok=True)
            dir2.mkdir(parents=True, exist_ok=True)
            
            # Generate filenames
            timestamp = Path(self.img1_path).stem + "_" + Path(self.img2_path).stem
            out1 = dir1 / f"{timestamp}_img1.png"
            out2 = dir2 / f"{timestamp}_img2.png"
            
            # Save images
            cv2.imwrite(str(out1), self.alignment_result['cropped_img1'])
            cv2.imwrite(str(out2), self.alignment_result['cropped_img2'])
            
            messagebox.showinfo(
                "Success",
                f"Cropped images saved:\n\n"
                f"Image 1: {out1}\n"
                f"Image 2: {out2}"
            )
            self.status_label.config(text=f"Saved to {base_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save images: {str(e)}")


def launch_gui():
    """Launch the Image Matcher GUI application."""
    root = tk.Tk()
    app = ImageMatcherGUI(root)
    root.mainloop()


# Example usage
if __name__ == "__main__":
    # Launch GUI
    print("Launching Image Matcher GUI...")
    print("Use the GUI to:")
    print("1. Load two images")
    print("2. Click 'Match Images' to find and align overlapping regions")
    print("3. Toggle 'Show Overlay' to see both images overlaid")
    print("4. Adjust opacity slider to change overlay blend")
    print("5. Click 'Save Cropped Images' to save the aligned overlap regions")
    print()
    
    launch_gui()

