import numpy as np
import math


def mfp1p2_polar(n, prob1, prob2, iterations):
    """
    Generates a variant of the multifractal mfp1p2 cascade model on a circular support.
    It also calculates the theoretical multifractal spectrum (f(alpha)).

    NOTE: The number of iterations MUST be even.

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

    if k % 2 != 0:
        print("Warning: The number of iterations MUST be even.")

    # Initialize Matrices and Cascade Sequence Setup

    # Generate the sequence of powers of 2 for partitioning (x), truncated to k elements.
    powers_of_2 = 2 ** np.arange(1, k + 1)
    sqsz = np.repeat(powers_of_2, 2)[:k]

    # Initialize the matrix that holds the final probability (initially all 1s)
    probmat = np.ones((sz, sz), dtype=float)

    # row_sizes and col_sizes track the current block sizes along each dimension.
    row_sizes = [sz]
    col_sizes = [sz]

    # Multiplicative Cascade Process

    for counter, x in enumerate(sqsz, 1):
        pxsz = sz // x

        if counter % 2 == 0:  # Even counter: Vertical Cut (Columns)
            num_boxes_C = sz // pxsz
            col_sizes = [pxsz] * num_boxes_C
            col_indices = np.cumsum([0] + col_sizes)

            r_start = 0
            for r_size in row_sizes:
                r_end = r_start + r_size

                for j in range(0, num_boxes_C, 2):
                    c1_start, c1_end = col_indices[j], col_indices[j + 1]
                    c2_start, c2_end = col_indices[j + 1], col_indices[j + 2]

                    if np.random.rand() < 0.5:
                        probmat[r_start:r_end, c1_start:c1_end] *= p1
                        probmat[r_start:r_end, c2_start:c2_end] *= p2
                    else:
                        probmat[r_start:r_end, c1_start:c1_end] *= p2
                        probmat[r_start:r_end, c2_start:c2_end] *= p1

                r_start = r_end

        else:  # Odd counter: Horizontal Cut (Rows)
            num_boxes_R = sz // pxsz
            row_sizes = [pxsz] * num_boxes_R
            row_indices = np.cumsum([0] + row_sizes)

            c_start = 0
            for c_size in col_sizes:
                c_end = c_start + c_size

                for i in range(0, num_boxes_R, 2):
                    r1_start, r1_end = row_indices[i], row_indices[i + 1]
                    r2_start, r2_end = row_indices[i + 1], row_indices[i + 2]

                    if np.random.rand() < 0.5:
                        probmat[r1_start:r1_end, c_start:c_end] *= p1
                        probmat[r2_start:r2_end, c_start:c_end] *= p2
                    else:
                        probmat[r1_start:r1_end, c_start:c_end] *= p2
                        probmat[r2_start:r2_end, c_start:c_end] *= p1

                c_start = c_end

    # Polar Mapping and Image Generation

    # Determine the final block dimensions (assuming k is even, blocks are M x M)
    M = 2 ** (k // 2)

    # Calculate the mean probability for each final Cartesian block
    bandaid_matrix = np.zeros((M, M))
    r_idx = np.cumsum([0] + row_sizes[:-1])
    c_idx = np.cumsum([0] + col_sizes[:-1])
    len1 = row_sizes[0]
    len2 = col_sizes[0]

    for i in range(M):
        r_start, r_end = r_idx[i], r_idx[i] + len1
        for j in range(M):
            c_start, c_end = c_idx[j], c_idx[j] + len2
            bandaid_matrix[i, j] = np.mean(probmat[r_start:r_end, c_start:c_end])

    # Generate Cartesian coordinates centered at (0, 0)
    x = np.arange(sz) - (sz / 2 - 0.5)
    Xim, Yim = np.meshgrid(x, x)

    # Convert to Polar coordinates
    theta, rho = np.arctan2(Yim, Xim), np.sqrt(Xim ** 2 + Yim ** 2)

    # Calculate Radial Partition Boundaries
    d1 = np.pi * np.ones(M)
    d2 = -np.pi * np.ones(M - 1)
    A = np.diag(d1) + np.diag(d2, k=-1)

    areas = (np.pi / M) * np.ones(M)
    r_sq = np.linalg.solve(A, areas)

    # rhorange is the square root of the cumulative sum of r_sq, starting with 0
    rhorange_sq = np.concatenate(([0], np.cumsum(r_sq)))
    rhorange = np.sqrt(rhorange_sq)

    # Rescale rhorange: map [0, max_value] to [0, sz/2]
    # The max value rhorange[-1] maps to sz/2
    if rhorange[-1] > 0:
        rhorange = rhorange / rhorange[-1] * (sz / 2)

    # Angular Partition Boundaries
    thetarange = np.linspace(-np.pi, np.pi, M + 1)

    # Map the bandaid probabilities onto the polar grid
    initmat = np.zeros((sz, sz), dtype=float)

    for i in range(M):  # Radial index
        r_min, r_max = rhorange[i], rhorange[i + 1]

        for j in range(M):  # Angular index
            t_min, t_max = thetarange[j], thetarange[j + 1]

            # Radial Mask: r_min <= rho < r_max
            mask_rho = (rho < r_max) & (rho >= r_min)

            # Angular Mask: t_min <= theta < t_max
            # Handle the last segment which includes pi
            if abs(t_max - np.pi) < 1e-9:
                mask_theta = (theta <= t_max) & (theta >= t_min)
            else:
                mask_theta = (theta < t_max) & (theta >= t_min)

            mask_combined = mask_rho & mask_theta

            # Assign the mean probability of the corresponding Cartesian block
            initmat[mask_combined] = bandaid_matrix[i, j]

    # Final Binarization
    mf_image = (np.random.rand(sz, sz) > initmat).astype(np.uint8)

    # Theoretical Multifractal Analysis

    h = 0.1
    q = np.arange(-10, 10 + h / 2, h)
    Dqtheory = np.zeros(len(q))

    a = (p1 + p2) / 2
    b = p1 / p2

    for currq in range(len(q)):
        q_val = q[currq]
        if abs(q_val - 1) < 1e-9:  # Check for q = 1 (L'Hopital's rule limit)
            Dqtheory[currq] = 2 * np.log2(b + 1) - (2 * b * np.log2(b)) / (b + 1)
        else:
            Dqtheory[currq] = (2 * np.log2(b ** q_val + 1) - 2 * q_val * np.log2(b + 1)) / (1 - q_val)

    # Calculate tau(q)
    tauq = (q - 1) * Dqtheory

    # Calculate alpha (numerical differentiation)
    alpha_theory = np.zeros(len(Dqtheory))
    alpha_theory[0] = (tauq[1] - tauq[0]) / h
    alpha_theory[-1] = (tauq[-1] - tauq[-2]) / h

    for step in range(1, len(alpha_theory) - 1):
        alpha_theory[step] = (tauq[step + 1] - tauq[step - 1]) / (2 * h)

    # Calculate f(alpha) (Legendre transform)
    f_theory = q * alpha_theory - tauq

    return mf_image, alpha_theory, f_theory


if __name__ == '__main__':
    # Example usage (Test Case)
    N = 1024  # Size of the image
    P1 = 1  # Probability 1
    P2 = 0.6  # Probability 2
    K = 8  # Number of iterations

    print(f"Running mfp1p2polar with N={N}, p1={P1}, p2={P2}, k={K}")

    mf_image, alpha_theory, f_theory = mfp1p2_polar(N, P1, P2, K)

    print("\nTheoretical Spectrum (alpha_theory vs f_theory):")
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