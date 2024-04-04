import numpy as np

def SAM(I1, I2):
  """
  Calculates the Spectral Angle Mapper (SAM) index and image.

  Args:
      I1: First multispectral image (numpy array).
      I2: Second multispectral image (numpy array).

  Returns:
      SAM_index: Spectral Angle Mapper index (float).
      SAM_map: Image of SAM values (numpy array).
  """
  M, N, B = I2.shape

  # Calculate dot product of corresponding bands
  prod_scal = np.sum(I1 * I2, axis=2)

  # Calculate L2 norm of each spectrum
  norm_orig = np.linalg.norm(I1, axis=2)
  norm_fusa = np.linalg.norm(I2, axis=2)

  # Prevent division by zero
  prod_norm = np.sqrt(norm_orig * norm_fusa)
  prod_map = prod_norm.copy()
  prod_map[prod_map == 0] = np.finfo(float).eps  # Replace zeros with epsilon

  # Calculate SAM map
  SAM_map = np.arccos(np.clip(prod_scal / prod_map, -1, 1))

  # Reshape for vectorized operations
  prod_scal = prod_scal.flatten()
  prod_norm = prod_norm.flatten()
  prod_map = prod_map.flatten()

  # Remove elements with zero norm (avoid NaNs)
  valid_pixels = prod_norm != 0
  prod_scal = prod_scal[valid_pixels]
  prod_norm = prod_norm[valid_pixels]
  prod_map = prod_map[valid_pixels]

  # Calculate average spectral angle
  angolo = np.sum(np.arccos(np.clip(prod_scal / prod_map, -1, 1))) / len(prod_norm)

  # Convert to degrees
  SAM_index = np.real(angolo) * 180 / np.pi

  return SAM_index, SAM_map

def SAM_group(I1_group, I2_group):
    N, H, W, C = I1_group.shape
    SAM_map = np.zeros((N, H, W))
    SAM_index = 0
    for i in range(N):
        I1 = I1_group[i]
        I2 = I2_group[i]
        sam_index, _ = SAM(I1, I2)
        SAM_index += sam_index
        SAM_map[i] = _

    return SAM_index / N, SAM_map


