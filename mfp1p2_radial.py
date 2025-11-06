import numpy as np
import math


def mfp1p2radial(n, prob1, prob2, iterations):
    """
    Generates a 1-D variant of the multifractal mfp1p2 cascade model on a circular support.
    It also calculates the theoretical multifractal spectrum (f(alpha)).

    Args:
        n (int): The side length of the square matrix (size n x n).
        prob1 (float): The first probability parameter (p1).
        prob2 (float): The second probability parameter (p2).
        iterations (int): The number of cascade iterations (k).

    Returns:
        tuple: (mf_image, alpha_theory, f_theory)
            mf_image (np.ndarray): The final n x n binary image (0s and 1s).
            alpha_theory (np.ndarray): The theoretical singularity exponents (alpha).
            f_theory (np.ndarray): The theoretical multifractal spectrum (f(alpha)).
    """
    p1 = prob1
    p2 = prob2
    k = iterations
    sz = n

    # Multiplicative Cascade (resulting in pvec of size 2^k)
    pvec = np.array([1.0], dtype=float)
    M = 2 ** k

    for i in range(k):
        # Double the vector: [pvec, pvec]
        temp = np.concatenate((pvec, pvec))

        # Calculate split index
        len_i = 2 ** (i + 1)
        split_idx = len_i // 2

        # Random coin flip
        if np.random.rand() < 0.5:
            # Left *= p1, Right *= p2
            temp[:split_idx] *= p1
            temp[split_idx:] *= p2
        else:
            # Left *= p2, Right *= p1
            temp[:split_idx] *= p2
            temp[split_idx:] *= p1

        pvec = temp

    # Image Setup and Ellipse Mask

    # Calculate mask parameters
    imrows, imcols = np.meshgrid(np.arange(1, sz + 1), np.arange(1, sz + 1))

    centery = sz // 2
    centerx = sz // 2
    radiusy = centery
    radiusx = centerx

    # Calculate the elliptical mask (boolean array)
    myellipse = ((imrows - centery) ** 2 / radiusy ** 2 +
                 (imcols - centerx) ** 2 / radiusx ** 2) <= 1

    # Angular Coordinate Setup and Mapping

    # Generate coordinates centered at (0, 0)
    # This is the coordinate system that maps the center of the grid to (0,0)
    x_coords = np.arange(sz) - (sz / 2 - 0.5)
    Xim, Yim = np.meshgrid(x_coords, x_coords)

    # Calculate the angle for every pixel
    angmat = np.arctan2(Yim, Xim)

    # Generate angular bins (2^k bins)
    theta = np.linspace(-np.pi, np.pi, M + 1)

    pmat = np.zeros((sz, sz), dtype=float)

    # Map the pvec values onto the angular sectors
    for j in range(M):
        t_min, t_max = theta[j], theta[j + 1]
        p_val = pvec[j]

        # Angular Mask logic: (angmat > t_min) & (angmat <= t_max)
        # Handle the last segment which must include +pi
        if abs(t_max - np.pi) < 1e-9:
            mask_theta = (angmat > t_min) & (angmat <= t_max)
        else:
            mask_theta = (angmat > t_min) & (angmat <= t_max)

        # Add probability value to the matrix for pixels within this sector
        pmat[mask_theta] += p_val

    # Apply the elliptical mask: areas outside the ellipse are zeroed out.
    pmat = pmat * myellipse.astype(float)

    # Final Binarization

    # Initialize image to all 1s (inactive, following MATLAB initmat = ones(sz))
    mf_image = np.ones((sz, sz), dtype=np.uint8)
    rand_matrix = np.random.rand(sz, sz)

    # If random number <= pmat, the pixel is set to 0 (active)
    mf_image[rand_matrix <= pmat] = 0

    # Theoretical Multifractal Analysis

    h = 0.1
    # q vector from -10 to 10 with step h
    q = np.arange(-10, 10 + h / 2, h)
    Dqtheory = np.zeros(len(q))

    # The binomial cascade parameters for the theoretical spectrum
    a = (p1 + p2) / 2
    b = p1 / p2

    for currq in range(len(q)):
        q_val = q[currq]
        # Check for q = 1 (L'Hopital's rule limit) - using 1D formula
        if abs(q_val - 1) < 1e-9:
            Dqtheory[currq] = np.log2(b + 1) - (b * np.log2(b)) / (b + 1)
        else:
            # Formula for generalized dimensions Dq (1D)
            Dqtheory[currq] = (np.log2(b ** q_val + 1) - q_val * np.log2(b + 1)) / (1 - q_val)

    # Calculate tau(q) (mass exponent)
    tauq = (q - 1) * Dqtheory

    # Calculate alpha (singularity exponent) using numerical differentiation
    alpha_theory = np.zeros(len(Dqtheory))

    # Numerical differentiation (Central difference)
    alpha_theory[0] = (tauq[1] - tauq[0]) / h
    alpha_theory[-1] = (tauq[-1] - tauq[-2]) / h

    for step in range(1, len(alpha_theory) - 1):
        alpha_theory[step] = (tauq[step + 1] - tauq[step - 1]) / (2 * h)

    # Calculate f(alpha) (multifractal spectrum) using Legendre transform
    f_theory = q * alpha_theory - tauq

    return mf_image, alpha_theory, f_theory


if __name__ == '__main__':
    # Example usage (Test Case)
    N = 1024  # Size of the image
    P1 = 1  # Probability 1
    P2 = 0.6  # Probability 2
    K = 8  # Number of angular bins is 2^K = 256

    print(f"Running mfp1p2radial with N={N}, p1={P1}, p2={P2}, k={K}")

    mf_image, alpha_theory, f_theory = mfp1p2radial(N, P1, P2, K)

    print("\nTheoretical Spectrum (alpha_theory vs f_theory, 1D):")
    max_f_idx = np.argmax(f_theory)

    print(f"  alpha_min: {alpha_theory.min():.4f}")
    print(f"  alpha_max: {alpha_theory.max():.4f}")
    print(f"  Max f(alpha) = {f_theory[max_f_idx]:.4f} at alpha = {alpha_theory[max_f_idx]:.4f}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mf_image, cmap='gray')
    plt.title('MFP1P2 Cascade Image')
    plt.subplot(1, 2, 2)
    plt.plot(alpha_theory, f_theory)
    plt.title('Theoretical f(alpha) Spectrum')
    plt.xlabel('alpha')
    plt.ylabel('f(alpha)')
    plt.grid(True)
    plt.show()