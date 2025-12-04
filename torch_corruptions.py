"""
PyTorch-based image corruptions for GPU-accelerated parallel processing.
Reimplements imagecorruptions library using PyTorch for batch processing on GPU.
"""
import torch
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from pathlib import Path


# Corruption categories
CORRUPTION_CATEGORIES = {
    'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
    'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur'],
    'weather': ['snow', 'frost', 'fog', 'spatter'],
    'digital': ['contrast', 'brightness', 'saturate', 'jpeg_compression', 'pixelate', 'elastic_transform']
}


def get_corruption_names(subset='all'):
    """Get list of corruption names for a given subset."""
    if subset == 'all':
        return [c for cats in CORRUPTION_CATEGORIES.values() for c in cats]
    elif subset in CORRUPTION_CATEGORIES:
        return CORRUPTION_CATEGORIES[subset]
    elif subset == 'common':
        return ['gaussian_noise', 'shot_noise', 'impulse_noise', 
                'defocus_blur', 'glass_blur', 'motion_blur',
                'snow', 'frost', 'fog',
                'brightness', 'contrast']
    elif subset == 'validation':
        return ['gaussian_noise', 'shot_noise', 'motion_blur', 
                'snow', 'brightness']
    else:
        raise ValueError(f"Unknown subset: {subset}")


class TorchCorruptions:
    """PyTorch-based corruption generator for GPU acceleration."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize corruption generator.
        
        Args:
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        print(f"TorchCorruptions initialized on device: {self.device}")
        
        # Preload frost images (if available)
        self.frost_images = self._load_frost_images()
    
    def _load_frost_images(self):
        """Preload frost texture images and convert to tensors."""
        frost_dir = Path(__file__).parent / 'frost'
        if not frost_dir.exists():
            return None
        
        frost_images = []
        for i in range(1, 7):
            frost_path = frost_dir / f'frost{i}.{"png" if i <= 3 else "jpg"}'
            if frost_path.exists():
                img = Image.open(frost_path).convert('RGB')
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                frost_images.append(img_tensor.to(self.device))
        
        return frost_images if frost_images else None
    
    def to_tensor(self, images):
        """
        Convert images to torch tensors.
        
        Args:
            images: Single image (H,W,C) or batch (B,H,W,C) as numpy array or PIL Image
        
        Returns:
            Tensor of shape (B,C,H,W) normalized to [0,1]
        """
        if isinstance(images, Image.Image):
            # Direct PIL to tensor conversion
            from torchvision import transforms
            images = transforms.ToTensor()(images)
            # ToTensor returns (C,H,W) in [0,1], so add batch dimension
            images = images.unsqueeze(0)
            return images.to(self.device)
        
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        
        # Normalize to [0, 1]
        if images.max() > 1:
            images = images / 255.0
        
        # Add batch dimension if single image
        if images.ndim == 3:
            images = images.unsqueeze(0)
        
        # Convert (B,H,W,C) to (B,C,H,W)
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        
        return images.to(self.device)
    
    def to_numpy(self, tensor, original_shape):
        """
        Convert tensor back to numpy array.
        
        Args:
            tensor: Tensor of shape (B,C,H,W) or (C,H,W)
            original_shape: Original input shape to determine if we should squeeze
        
        Returns:
            Numpy array (H,W,C) or (B,H,W,C) in range [0, 255]
        """
        # Move to CPU and convert to numpy
        tensor = tensor.cpu()
        
        # Convert (B,C,H,W) to (B,H,W,C)
        tensor = tensor.permute(0, 2, 3, 1)
        
        # Clip and scale to [0, 255]
        array = (tensor.clamp(0, 1) * 255).byte().numpy()
        
        # Remove batch dimension if original input was single image (3D)
        if len(original_shape) == 3:
            array = array[0]
        
        return array
    
    def to_pil(self, tensor):
        """
        Convert tensor to PIL Image.
        
        Args:
            tensor: Tensor of shape (C,H,W) in range [0,1]
        
        Returns:
            PIL Image
        """
        from torchvision import transforms
        # Convert single image tensor (C,H,W) to PIL
        return transforms.ToPILImage()(tensor.cpu())
    
    # ============ Noise Corruptions ============
    
    def gaussian_noise(self, x, severity=1):
        """Add Gaussian noise."""
        c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        noise = torch.randn_like(x) * c
        return torch.clamp(x + noise, 0, 1)
    
    def shot_noise(self, x, severity=1):
        """Add shot (Poisson) noise."""
        c = [60, 25, 12, 5, 3][severity - 1]
        # Poisson noise approximation
        x_scaled = x * c
        noise = torch.poisson(x_scaled) / float(c)
        return torch.clamp(noise, 0, 1)
    
    def impulse_noise(self, x, severity=1):
        """Add salt and pepper noise."""
        c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
        
        # Generate random mask for salt and pepper
        mask = torch.rand_like(x[:, :1, :, :])
        
        # Salt (white pixels)
        salt_mask = mask < c / 2
        # Pepper (black pixels)
        pepper_mask = (mask >= c / 2) & (mask < c)
        
        result = x.clone()
        result = torch.where(salt_mask.expand_as(x), torch.ones_like(x), result)
        result = torch.where(pepper_mask.expand_as(x), torch.zeros_like(x), result)
        
        return result
    
    def speckle_noise(self, x, severity=1):
        """Add speckle noise."""
        c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
        noise = torch.randn_like(x) * c
        return torch.clamp(x + x * noise, 0, 1)
    
    # ============ Blur Corruptions ============
    
    def gaussian_blur(self, x, severity=1):
        """Apply Gaussian blur."""
        c = [1, 2, 3, 4, 6][severity - 1]
        
        # Create Gaussian kernel
        kernel_size = int(2 * math.ceil(2 * c) + 1)
        sigma = c
        
        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # Create 2D kernel by outer product
        kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        
        # Apply to each channel
        result = []
        for i in range(x.shape[1]):
            blurred = F.conv2d(x[:, i:i+1, :, :], kernel_2d, padding=kernel_size//2)
            result.append(blurred)
        
        return torch.cat(result, dim=1)
    
    def defocus_blur(self, x, severity=1):
        """Apply defocus blur using disk kernel."""
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
        radius = c[0]
        
        # Create disk kernel
        kernel_size = 2 * radius + 1
        y, x_grid = torch.meshgrid(
            torch.arange(kernel_size, dtype=torch.float32, device=self.device) - radius,
            torch.arange(kernel_size, dtype=torch.float32, device=self.device) - radius,
            indexing='ij'
        )
        
        disk = (x_grid ** 2 + y ** 2) <= radius ** 2
        kernel = disk.float()
        kernel /= kernel.sum()
        
        # Apply small Gaussian blur for anti-aliasing
        if c[1] > 0:
            coords = torch.arange(3, dtype=torch.float32, device=self.device) - 1
            g = torch.exp(-(coords ** 2) / (2 * c[1] ** 2))
            g /= g.sum()
            blur_kernel = g.unsqueeze(0) * g.unsqueeze(1)
            blur_kernel = blur_kernel.unsqueeze(0).unsqueeze(0)
            kernel = F.conv2d(kernel.unsqueeze(0).unsqueeze(0), blur_kernel, padding=1)[0, 0]
            kernel /= kernel.sum()
        
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        # Apply to each channel
        result = []
        for i in range(x.shape[1]):
            blurred = F.conv2d(x[:, i:i+1, :, :], kernel, padding=kernel_size//2)
            result.append(blurred)
        
        return torch.cat(result, dim=1)
    
    def motion_blur(self, x, severity=1):
        """Apply motion blur."""
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
        radius, sigma = c
        
        # Random angle
        angle = torch.rand(1).item() * 90 - 45  # -45 to 45 degrees
        
        # Create motion blur kernel
        kernel_size = 2 * radius + 1
        kernel = torch.zeros(kernel_size, dtype=torch.float32, device=self.device)
        
        # Gaussian weights along the line
        coords = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        coords -= kernel_size // 2
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        
        # Create 2D directional kernel
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # For simplicity, use horizontal kernel and rotate
        kernel_2d = kernel.unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        
        # Apply to each channel (simplified version - full rotation would require grid_sample)
        result = []
        for i in range(x.shape[1]):
            blurred = F.conv2d(x[:, i:i+1, :, :], kernel_2d, padding=(0, kernel_size//2))
            result.append(blurred)
        
        return torch.cat(result, dim=1)
    
    def zoom_blur(self, x, severity=1):
        """Apply zoom blur."""
        zoom_factors = [
            torch.linspace(1, 1.11, 11, device=self.device),
            torch.linspace(1, 1.16, 16, device=self.device),
            torch.linspace(1, 1.21, 11, device=self.device),
            torch.linspace(1, 1.26, 13, device=self.device),
            torch.linspace(1, 1.31, 11, device=self.device)
        ][severity - 1]
        
        out = torch.zeros_like(x)
        
        for zoom_factor in zoom_factors:
            # Scale up
            h, w = x.shape[2], x.shape[3]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            
            zoomed = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # Center crop back to original size
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            
            if start_h >= 0 and start_w >= 0:
                zoomed_crop = zoomed[:, :, start_h:start_h+h, start_w:start_w+w]
            else:
                zoomed_crop = zoomed
            
            out += zoomed_crop
        
        out = (x + out) / (len(zoom_factors) + 1)
        return torch.clamp(out, 0, 1)
    
    def glass_blur(self, x, severity=1):
        """Apply glass blur effect."""
        # Glass blur is complex and requires local pixel shuffling
        # For now, return gaussian blur with pixel shuffling (simplified)
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        sigma, max_delta, iterations = c
        
        # First apply Gaussian blur
        result = self.gaussian_blur(x, severity=min(3, severity))
        
        # Note: Full implementation would require pixel shuffling which is 
        # hard to parallelize on GPU. For batch processing, we keep it simple.
        
        return result
    
    # ============ Weather Corruptions ============
    
    def fog(self, x, severity=1):
        """Apply fog effect using atmospheric scattering."""
        c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
        
        B, C, H, W = x.shape
        
        # Create fog layer (uniform gray)
        fog_color = 0.7  # Gray fog
        
        # Generate depth-like pattern for fog density variation
        fog_density = torch.zeros((B, 1, H, W), device=self.device)
        
        for scale in [1, 2, 4, 8]:
            h_scale = max(1, H // scale)
            w_scale = max(1, W // scale)
            noise = torch.rand((B, 1, h_scale, w_scale), device=self.device)
            noise = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)
            fog_density += noise / scale
        
        # Normalize fog density
        fog_density = fog_density / fog_density.max()
        
        # Apply fog using atmospheric scattering formula: result = x * (1 - alpha) + fog_color * alpha
        # where alpha depends on severity and fog_density
        alpha = fog_density * (severity / c[1])  # Adjust alpha based on severity
        alpha = torch.clamp(alpha, 0, 0.85)  # Limit max fog
        
        result = x * (1 - alpha) + fog_color * alpha
        
        return torch.clamp(result, 0, 1)
    
    def snow(self, x, severity=1):
        """Apply snow effect."""
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
             (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
        
        B, C, H, W = x.shape
        
        # Generate snow layer
        snow_layer = torch.randn((B, 1, H, W), device=self.device) * c[1] + c[0]
        
        # Zoom effect (simplified)
        zoom_factor = c[2]
        if zoom_factor > 1:
            new_h, new_w = int(H * zoom_factor), int(W * zoom_factor)
            snow_layer = F.interpolate(snow_layer, size=(new_h, new_w), mode='bilinear', align_corners=False)
            start_h, start_w = (new_h - H) // 2, (new_w - W) // 2
            snow_layer = snow_layer[:, :, start_h:start_h+H, start_w:start_w+W]
        
        # Threshold
        snow_layer = torch.where(snow_layer < c[3], torch.zeros_like(snow_layer), snow_layer)
        snow_layer = torch.clamp(snow_layer, 0, 1)
        
        # Blend with image
        # Convert to grayscale for brightness calculation
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = c[6] * x + (1 - c[6]) * torch.maximum(x, gray * 1.5 + 0.5)
        
        # Add snow layer
        result = x + snow_layer
        
        return torch.clamp(result, 0, 1)
    
    def frost(self, x, severity=1):
        """Apply frost effect using preloaded textures or procedural generation."""
        c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
        
        B, C, H, W = x.shape
        
        if self.frost_images is not None and len(self.frost_images) > 0:
            # Use frost texture
            idx = torch.randint(len(self.frost_images), (1,)).item()
            frost = self.frost_images[idx]
            
            # Resize frost to match image size (with some scaling)
            frost_h, frost_w = frost.shape[0], frost.shape[1]
            scale = max(H / frost_h, W / frost_w) * 1.1
            
            new_h, new_w = int(frost_h * scale), int(frost_w * scale)
            frost = frost.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            frost = F.interpolate(frost, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # Random crop
            start_h = torch.randint(0, max(1, new_h - H), (1,)).item()
            start_w = torch.randint(0, max(1, new_w - W), (1,)).item()
            frost = frost[:, :, start_h:start_h+H, start_w:start_w+W]
            
            # Expand to batch size
            frost = frost.repeat(B, 1, 1, 1)
        else:
            # Generate procedural frost pattern
            # Create crystalline pattern using multiple noise scales
            frost_pattern = torch.zeros((B, 1, H, W), device=self.device)
            
            for scale in [1, 2, 4, 8, 16]:
                h_scale = max(1, H // scale)
                w_scale = max(1, W // scale)
                noise = torch.rand((B, 1, h_scale, w_scale), device=self.device)
                noise = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)
                frost_pattern += noise / scale * 0.3
            
            # Normalize and enhance contrast for crystalline effect
            frost_pattern = frost_pattern / frost_pattern.max()
            frost_pattern = torch.pow(frost_pattern, 0.7)  # Enhance bright areas
            frost_pattern = frost_pattern * 0.9 + 0.1  # Shift to lighter values
            
            # Make it slightly blue-ish (create 3-channel frost)
            frost = torch.zeros((B, 3, H, W), device=self.device)
            frost[:, 0:1] = frost_pattern * 0.95  # R
            frost[:, 1:2] = frost_pattern * 0.98  # G  
            frost[:, 2:3] = frost_pattern * 1.0   # B (slightly more blue)
        
        # Blend using screen mode for frost overlay
        # result = image * (1 - alpha) + frost * alpha
        result = c[0] * x + c[1] * frost
        
        return torch.clamp(result, 0, 1)
    
    def spatter(self, x, severity=1):
        """Apply spatter effect."""
        # Simplified spatter using random blobs
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
             (0.65, 0.3, 3, 0.68, 0.6, 0),
             (0.65, 0.3, 2, 0.68, 0.5, 0),
             (0.65, 0.3, 1, 0.65, 1.5, 1),
             (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
        
        B, C, H, W = x.shape
        
        # Generate liquid/mud pattern
        liquid = torch.randn((B, 1, H, W), device=self.device) * c[1] + c[0]
        
        # Blur
        liquid = self.gaussian_blur(liquid, severity=min(2, c[2]))
        
        # Threshold
        liquid = torch.where(liquid < c[3], torch.zeros_like(liquid), torch.ones_like(liquid))
        
        if c[5] == 0:
            # Water (turquoise)
            color = torch.tensor([175/255., 238/255., 238/255.], device=self.device).view(1, 3, 1, 1)
        else:
            # Mud (brown)
            color = torch.tensor([63/255., 42/255., 20/255.], device=self.device).view(1, 3, 1, 1)
        
        color = color * liquid * c[4]
        result = x * (1 - liquid) + color
        
        return torch.clamp(result, 0, 1)
    
    # ============ Digital Corruptions ============
    
    def contrast(self, x, severity=1):
        """Adjust contrast."""
        c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
        
        means = x.mean(dim=[2, 3], keepdim=True)
        result = (x - means) * c + means
        
        return torch.clamp(result, 0, 1)
    
    def brightness(self, x, severity=1):
        """Adjust brightness."""
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        
        result = x + c
        
        return torch.clamp(result, 0, 1)
    
    def saturate(self, x, severity=1):
        """Adjust saturation."""
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
        
        # Convert to HSV (simplified - using approximation)
        # For batch processing, we use a simplified saturation adjustment
        # True HSV conversion is complex for tensors
        
        # Get grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Adjust saturation by blending with grayscale
        result = gray + c[0] * (x - gray) + c[1]
        
        return torch.clamp(result, 0, 1)
    
    def pixelate(self, x, severity=1):
        """Apply pixelation."""
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        
        B, C, H, W = x.shape
        
        # Downsample
        new_h, new_w = int(H * c), int(W * c)
        downsampled = F.interpolate(x, size=(new_h, new_w), mode='nearest')
        
        # Upsample back
        result = F.interpolate(downsampled, size=(H, W), mode='nearest')
        
        return result
    
    def jpeg_compression(self, x, severity=1):
        """Simulate JPEG compression artifacts."""
        # JPEG compression is difficult to implement in pure PyTorch
        # This is a simplified approximation using quantization
        c = [25, 18, 15, 10, 7][severity - 1]
        
        # Quantization level (lower quality = more quantization)
        quant_level = 255 // c
        
        # Quantize
        result = torch.round(x * quant_level) / quant_level
        
        return torch.clamp(result, 0, 1)
    
    def elastic_transform(self, x, severity=1):
        """Apply elastic deformation."""
        # Elastic transform requires grid sampling which is available in PyTorch
        # but complex to implement efficiently for batches
        # For now, return slightly blurred image (placeholder)
        
        return self.gaussian_blur(x, severity=min(2, severity))
    
    # ============ Main API ============
    
    def corrupt(self, images, corruption_name, severity=1, return_tensor=False):
        """
        Apply corruption to a batch of images.
        
        Args:
            images: Batch of images as numpy array (B,H,W,C) or (H,W,C), PIL Image, or torch.Tensor
            corruption_name: Name of the corruption to apply
            severity: Severity level (1-5)
            return_tensor: If True, return tensor (C,H,W) instead of numpy array
        
        Returns:
            Corrupted images as numpy array (default) or tensor if return_tensor=True
        """
        # Check if input is already a tensor
        if isinstance(images, torch.Tensor):
            x = images.to(self.device)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            input_was_tensor = True
            original_shape = None
        else:
            # Store original shape to determine output format
            if isinstance(images, Image.Image):
                original_shape = np.array(images).shape
            elif isinstance(images, np.ndarray):
                original_shape = images.shape
            else:
                original_shape = images.shape
            
            # Convert to tensor
            x = self.to_tensor(images)
            input_was_tensor = False
        
        # Apply corruption
        corruption_fn = getattr(self, corruption_name, None)
        if corruption_fn is None:
            raise ValueError(f"Unknown corruption: {corruption_name}")
        
        corrupted = corruption_fn(x, severity)
        
        # Return in requested format
        if return_tensor or input_was_tensor:
            # Return as tensor (B,C,H,W) or (C,H,W) if single image
            if corrupted.shape[0] == 1:
                return corrupted.squeeze(0)
            return corrupted
        else:
            # Convert back to numpy
            result = self.to_numpy(corrupted, original_shape)
            return result
    
    def corrupt_batch(self, images, corruption_names, severities):
        """
        Apply different corruptions to different images in a batch.
        
        Args:
            images: Batch of images (B,H,W,C)
            corruption_names: List of corruption names (length B)
            severities: List of severities (length B)
        
        Returns:
            Batch of corrupted images (B,H,W,C)
        """
        batch_size = len(images)
        assert len(corruption_names) == batch_size
        assert len(severities) == batch_size
        
        original_shape = images.shape
        
        # Convert to tensor
        x = self.to_tensor(images)
        
        # Group by corruption type for efficiency
        corruption_groups = {}
        for i, (corr, sev) in enumerate(zip(corruption_names, severities)):
            key = (corr, sev)
            if key not in corruption_groups:
                corruption_groups[key] = []
            corruption_groups[key].append(i)
        
        # Apply corruptions
        results = torch.zeros_like(x)
        for (corr, sev), indices in corruption_groups.items():
            corruption_fn = getattr(self, corr, None)
            if corruption_fn is None:
                continue
            
            batch = x[indices]
            corrupted = corruption_fn(batch, sev)
            results[indices] = corrupted
        
        # Convert back to numpy
        return self.to_numpy(results, original_shape)


# Convenience function for backward compatibility
_torch_corruptor = None

def corrupt(image, corruption_name, severity=1, device='cuda'):
    """
    Apply corruption to an image (backward compatible with imagecorruptions).
    
    Args:
        image: Image as numpy array (H,W,C) or PIL Image
        corruption_name: Name of the corruption
        severity: Severity level (1-5)
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Corrupted image as numpy array (H,W,C)
    """
    global _torch_corruptor
    
    if _torch_corruptor is None:
        _torch_corruptor = TorchCorruptions(device=device)
    
    return _torch_corruptor.corrupt(image, corruption_name, severity)


if __name__ == '__main__':
    # Test the corruptions
    print("Testing TorchCorruptions...")
    
    # Create test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize
    corruptor = TorchCorruptions(device=device)
    
    # Test each corruption
    test_corruptions = ['gaussian_noise', 'gaussian_blur', 'snow', 'contrast']
    
    for corr in test_corruptions:
        print(f"Testing {corr}...")
        result = corruptor.corrupt(test_img, corr, severity=3)
        print(f"  Input shape: {test_img.shape}, Output shape: {result.shape}")
        assert result.shape == test_img.shape
        assert result.dtype == np.uint8
    
    # Test batch processing
    print("\nTesting batch processing...")
    batch = np.stack([test_img, test_img, test_img])
    batch_result = corruptor.corrupt(batch, 'gaussian_noise', severity=2)
    print(f"  Batch input shape: {batch.shape}, Output shape: {batch_result.shape}")
    
    print("\nAll tests passed!")
