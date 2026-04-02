import numpy as np
import torch
import pywt
from einops import rearrange


class WaveletMaskingGenerator:
    def __init__(self, input_size, mask_ratio, wavelet='haar', level=1):
        """
        2D Wavelet-based Masking Generator for video sequences
        
        Args:
            input_size: tuple (frames, height, width)
            mask_ratio: float, ratio of tokens to mask (0.0 to 1.0)
            wavelet: str, wavelet type for DWT (default: 'haar')
            level: int, decomposition level for DWT
        """
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.mask_ratio = mask_ratio
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
        self.wavelet = wavelet
        self.level = level
        
        # Validate wavelet parameters
        try:
            wavelet_obj = pywt.Wavelet(wavelet)
            if wavelet_obj.orthogonal is False:
                print(f"Warning: Wavelet {wavelet} is not orthogonal, using {wavelet} anyway")
        except ValueError as e:
            print(f"Warning: {e}, using default 'haar' wavelet")
            self.wavelet = 'haar'

    def __repr__(self):
        repr_str = "Wavelet Masking: total patches {}, mask patches {}, wavelet {}, level {}".format(
            self.total_patches, self.total_masks, self.wavelet, self.level
        )
        return repr_str

    def compute_frame_energy(self, frame):
        """
        Compute energy map for a single frame using 2D DWT
        
        Args:
            frame: numpy array of shape (H, W, C) or (H, W)
            
        Returns:
            energy_map: numpy array of shape (H, W) containing energy values
        """
        # Convert to grayscale — ensure result is strictly 2D (H, W)
        if frame.ndim == 3:
            if frame.shape[2] == 3:
                frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])  # (H, W)
            else:
                frame = frame[..., 0]  # take first channel -> (H, W)
        
        # frame is now guaranteed 2D
        H, W = frame.shape
        
        # Apply 2D Discrete Wavelet Transform
        coeffs = pywt.wavedec2(frame, self.wavelet, level=self.level)
        
        if len(coeffs) > 1:
            detail_coeffs = coeffs[-1]  # Tuple of (LH, HL, HH)
            lh, hl, hh = detail_coeffs
            energy_map = np.abs(lh) + np.abs(hl) + np.abs(hh)  # (H', W')
            
            # Upsample back to original (H, W) if needed
            eH, eW = energy_map.shape
            if (eH, eW) != (H, W):
                scale_h = max(1, H // eH)
                scale_w = max(1, W // eW)
                energy_map = np.kron(energy_map, np.ones((scale_h, scale_w)))
                # Crop or pad to exact (H, W)
                energy_map = energy_map[:H, :W]
                if energy_map.shape != (H, W):
                    padded = np.zeros((H, W))
                    h_, w_ = energy_map.shape
                    padded[:h_, :w_] = energy_map
                    energy_map = padded
        else:
            energy_map = np.abs(frame)  # already 2D
        
        return energy_map  # guaranteed (H, W)

    def compute_average_energy(self, video_frames):
        """
        Compute average energy map across all frames in a video sequence
        
        Args:
            video_frames: torch tensor of shape (B, T, C, H, W) or (T, C, H, W)
            
        Returns:
            avg_energy: numpy array of shape (H_patch, W_patch) containing average energy
        """
        # Handle batch dimension — always reduce to (T, C, H, W)
        if video_frames.dim() == 5:  # (B, T, C, H, W)
            video_frames = video_frames[0]  # take first sample -> (T, C, H, W)
        
        # Convert to numpy
        frames_np = video_frames.cpu().float().numpy()  # (T, C, H, W)
        
        T, C, H, W = frames_np.shape
        
        # Compute energy for each frame
        frame_energies = []
        for t in range(T):
            frame = frames_np[t].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            energy = self.compute_frame_energy(frame)  # (H, W)
            frame_energies.append(energy)
        
        # Compute average energy across time -> (H, W)
        avg_energy = np.mean(frame_energies, axis=0)
        
        # Resize energy map to patch dimensions (H/16, W/16)
        # This matches the patch embedding size expected by the transformer
        patch_h = H // 16
        patch_w = W // 16
        
        # Simple downsampling by averaging
        if avg_energy.shape != (patch_h, patch_w):
            # Reshape to patches and average within each patch
            energy_reshaped = avg_energy.reshape(patch_h, 16, patch_w, 16)
            avg_energy = energy_reshaped.mean(axis=(1, 3))
        
        return avg_energy  # guaranteed (H_patch, W_patch)

    def generate_2d_mask_template(self, avg_energy, mask_ratio=None):
        """
        Generate 2D base mask template from average energy map
        
        Args:
            avg_energy: numpy array of shape (H, W) containing average energy
            mask_ratio: float, override mask ratio (optional)
            
        Returns:
            mask_2d: numpy array of shape (H, W) with 1 for masked, 0 for visible
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Ensure 2D
        if avg_energy.ndim != 2:
            raise ValueError(f"avg_energy must be 2D, got shape {avg_energy.shape}")
            
        H, W = avg_energy.shape
        total_tokens = H * W
        num_masks = int(mask_ratio * total_tokens)
        
        # Flatten energy map and get indices sorted by energy (highest first)
        flat_energy = avg_energy.flatten()
        sorted_indices = np.argsort(flat_energy)[::-1]  # descending
        
        # Create mask: set highest energy positions to 1 (masked)
        mask_flat = np.zeros(total_tokens)
        mask_flat[sorted_indices[:num_masks]] = 1
        
        mask_2d = mask_flat.reshape(H, W)
        
        return mask_2d

    def __call__(self, video_frames=None):
        """
        Generate wavelet-based tubular mask
        
        Args:
            video_frames: torch tensor of shape (B, T, C, H, W) for energy computation
                         If None, returns random mask (backward compatibility)
        
        Returns:
            mask: numpy array of shape (T, H, W) with 1 for masked, 0 for visible
        """
        if video_frames is not None:
            avg_energy = self.compute_average_energy(video_frames)  # (H, W)
            mask_2d = self.generate_2d_mask_template(avg_energy)    # (H, W)
            mask = np.tile(mask_2d, (self.frames, 1, 1))             # (T, H, W)
        else:
            # Fallback to random masking
            mask_per_frame = np.hstack([
                np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
                np.ones(self.num_masks_per_frame),
            ])
            np.random.shuffle(mask_per_frame)
            mask = np.tile(mask_per_frame, (self.frames, 1)).reshape(self.frames, self.height, self.width)
        
        return mask

    def get_mask_tensor(self, video_frames=None, device='cpu'):
        """
        Generate mask as PyTorch tensor
        
        Args:
            video_frames: torch tensor for energy computation
            device: device for output tensor
            
        Returns:
            mask_tensor: torch tensor of shape (T, H, W) on specified device
        """
        mask = self(video_frames)
        mask_tensor = torch.from_numpy(mask).float().to(device)
        return mask_tensor


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1))
        return mask
