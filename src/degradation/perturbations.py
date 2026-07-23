import numpy as np
import SimpleITK as sitk

def apply_gaussian_blur(image, sigma_mm):
    """
    Apply isotropic Gaussian spatial blurring using SimpleITK.
    sigma_mm: Blur standard deviation in mm.
    """
    if sigma_mm <= 0:
        return image
    
    blur_filter = sitk.DiscreteGaussianImageFilter()
    blur_filter.SetVariance(sigma_mm ** 2)
    blur_filter.SetUseImageSpacing(True)
    return blur_filter.Execute(image)

def apply_rician_noise(image, noise_level, seed=42):
    """
    Apply physically realistic Rician noise to magnitude MRI voxels (Gudbjartsson & Patz, 1995).
    noise_level: noise std dev relative to foreground 99th percentile intensity.
    M_noisy = sqrt((M + N_real)^2 + N_imag^2)
    """
    if noise_level <= 0:
        return image
    
    if seed is not None:
        np.random.seed(seed)
    
    img_np = sitk.GetArrayFromImage(image).astype(np.float32)
    foreground = img_np[img_np > 0]
    if len(foreground) == 0:
        max_val = 1.0
    else:
        max_val = np.percentile(foreground, 99)

    std_dev = noise_level * max_val
    n_real = np.random.normal(0, std_dev, size=img_np.shape).astype(np.float32)
    n_imag = np.random.normal(0, std_dev, size=img_np.shape).astype(np.float32)
    
    rician_np = np.sqrt((img_np + n_real) ** 2 + n_imag ** 2).astype(np.float32)
    
    out_img = sitk.GetImageFromArray(rician_np)
    out_img.SetSpacing(image.GetSpacing())
    out_img.SetOrigin(image.GetOrigin())
    out_img.SetDirection(image.GetDirection())
    return out_img

# Backward compatibility alias
apply_gaussian_noise = apply_rician_noise

def apply_resolution_reduction(image, target_spacing):
    """
    Simulate slice thickness & anisotropic downsampling.
    target_spacing: tuple of (x_spacing, y_spacing, z_spacing) in mm.
    """
    orig_spacing = image.GetSpacing()
    if np.allclose(orig_spacing, target_spacing):
        return image
    
    orig_size = image.GetSize()
    target_size = [
        int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]
    
    # 1. Downsample using B-Spline or Linear interpolation
    resample_down = sitk.ResampleImageFilter()
    resample_down.SetInterpolator(sitk.sitkLinear)
    resample_down.SetOutputSpacing(target_spacing)
    resample_down.SetSize(target_size)
    resample_down.SetOutputDirection(image.GetDirection())
    resample_down.SetOutputOrigin(image.GetOrigin())
    resample_down.SetTransform(sitk.Transform())
    downsampled = resample_down.Execute(image)

    # 2. Re-upsample back to original space to simulate slice interpolation artifacts
    resample_up = sitk.ResampleImageFilter()
    resample_up.SetInterpolator(sitk.sitkLinear)
    resample_up.SetOutputSpacing(orig_spacing)
    resample_up.SetSize(orig_size)
    resample_up.SetOutputDirection(image.GetDirection())
    resample_up.SetOutputOrigin(image.GetOrigin())
    resample_up.SetTransform(sitk.Transform())
    
    return resample_up.Execute(downsampled)

def apply_motion_artifacts(image, frequency=4.0, amplitude=0.10):
    """
    Simulate patient motion phase ringing/ghosting artifacts via k-space sinusoidal phase shifts.
    """
    if amplitude <= 0:
        return image
    
    img_np = sitk.GetArrayFromImage(image).astype(np.complex64)
    # 3D FFT along spatial axes
    k_space = np.fft.fftn(img_np)
    
    depth, height, width = img_np.shape
    phase_shift = np.zeros_like(k_space, dtype=np.complex64)
    
    # Generate phase perturbation grid
    y_coords = np.linspace(-np.pi, np.pi, height)
    z_coords = np.linspace(-np.pi, np.pi, depth)
    Z, Y = np.meshgrid(z_coords, y_coords, indexing='ij')
    
    modulation = np.exp(1j * amplitude * np.sin(frequency * Y))
    for x in range(width):
        k_space[:, :, x] *= modulation
    
    corrupted_np = np.abs(np.fft.ifftn(k_space)).astype(np.float32)
    
    out_img = sitk.GetImageFromArray(corrupted_np)
    out_img.SetSpacing(image.GetSpacing())
    out_img.SetOrigin(image.GetOrigin())
    out_img.SetDirection(image.GetDirection())
    return out_img
