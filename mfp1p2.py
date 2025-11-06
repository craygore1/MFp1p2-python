import numpy as np
import math


def mfp1p2(n, prob1, prob2, iterations):
    """
    Generates a multifractal on a square support using the mfp1p2 cascade model.
    It also calculates the theoretical multifractal spectrum (f(alpha)).

    The model performs an iterative, alternating horizontal and vertical block
    partitioning, multiplying sub-blocks by p1 and p2 based on a random coin flip.

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

    # Initialize Matrices and Cascade Sequence

    powers_of_2 = 2 ** np.arange(1, k + 1)
    sqsz = np.repeat(powers_of_2, 2)[:k]

    # Initialize the matrix that holds the final probability
    probmat = np.ones((sz, sz), dtype=float)

    # row_sizes and col_sizes track the sizes of the current blocks along each dimension.
    # Initially, the whole matrix is one block of size n x n.
    row_sizes = [sz]
    col_sizes = [sz]

    # Multiplicative Cascade Process

    for counter, x in enumerate(sqsz, 1):
        # pxsz is the block size for the new partition in the cut direction
        pxsz = sz // x

        if counter % 2 == 0:  # Even counter: Vertical Cut
            # Update col_sizes to reflect the new, finer partition
            num_boxes_C = sz // pxsz
            col_sizes = [pxsz] * num_boxes_C

            # Get start indices for column slices (e.g., [0, n/2, n])
            col_indices = np.cumsum([0] + col_sizes)

            # Iterate over the existing row blocks (defined by row_sizes)
            r_start = 0
            for r_size in row_sizes:
                r_end = r_start + r_size

                # Iterate over the new column blocks in pairs (j, j+1)
                for j in range(0, len(col_indices) - 1, 2):
                    c1_start, c1_end = col_indices[j], col_indices[j + 1]
                    c2_start, c2_end = col_indices[j + 1], col_indices[j + 2]

                    # Random coin flip (half < 0.5 or half >= 0.5)
                    if np.random.rand() < 0.5:
                        # Left *= p1, Right *= p2
                        probmat[r_start:r_end, c1_start:c1_end] *= p1
                        probmat[r_start:r_end, c2_start:c2_end] *= p2
                    else:
                        # Left *= p2, Right *= p1
                        probmat[r_start:r_end, c1_start:c1_end] *= p2
                        probmat[r_start:r_end, c2_start:c2_end] *= p1

                r_start = r_end

        else:  # Odd counter: Horizontal Cut
            # Update row_sizes to reflect the new, finer partition
            num_boxes_R = sz // pxsz
            row_sizes = [pxsz] * num_boxes_R

            # Get start indices for row slices
            row_indices = np.cumsum([0] + row_sizes)

            # Iterate over the existing column blocks (defined by col_sizes)
            c_start = 0
            for c_size in col_sizes:
                c_end = c_start + c_size

                # Iterate over the new row blocks in pairs (i, i+1)
                for i in range(0, len(row_indices) - 1, 2):
                    r1_start, r1_end = row_indices[i], row_indices[i + 1]
                    r2_start, r2_end = row_indices[i + 1], row_indices[i + 2]

                    # Random coin flip
                    if np.random.rand() < 0.5:
                        # Top *= p1, Bottom *= p2
                        probmat[r1_start:r1_end, c_start:c_end] *= p1
                        probmat[r2_start:r2_end, c_start:c_end] *= p2
                    else:
                        # Top *= p2, Bottom *= p1
                        probmat[r1_start:r1_end, c_start:c_end] *= p2
                        probmat[r2_start:r2_end, c_start:c_end] *= p1

                c_start = c_end

    # 3. Final Image Generation (Creating the Binary Image)

    # Initialize the final image as all zeros
    mf_image = np.zeros((sz, sz), dtype=np.uint8)

    # Get the block dimensions from the final cascade step
    r_idx = np.cumsum([0] + row_sizes[:-1])
    c_idx = np.cumsum([0] + col_sizes[:-1])

    # The final block size (len1 x len2) is used inside the loop
    len1 = row_sizes[0] if row_sizes else sz
    len2 = col_sizes[0] if col_sizes else sz

    # Total pixels in one final block
    block_size = len1 * len2

    # Iterate over the final partitioned blocks
    for i in range(len(r_idx)):
        r_start = r_idx[i]
        r_end = r_start + len1

        for j in range(len(c_idx)):
            c_start = c_idx[j]
            c_end = c_start + len2

            # The probability is constant within this block.
            # We take the value from the top-left corner of the block.
            tempprob = probmat[r_start, c_start]

            # Calculate the number of 'active' pixels (value 1 in the final image)
            # MATLAB sets 'num' pixels to 0, and the rest to 1, but then shuffles.
            # Here, we calculate how many 1s should be present in the block.
            # The original MATLAB code calculates 'num' (number of 0s) and then
            # sets the first 'num' pixels to 0, and the rest to 1.
            # Total 1s = (len1*len2) - num
            num_zeros = math.ceil(block_size * tempprob)
            num_ones = block_size - num_zeros

            # Create the block vector with the correct number of 1s and 0s
            block_vector = np.concatenate((np.zeros(num_zeros), np.ones(num_ones)))

            # Randomly shuffle the 1s and 0s within the block
            np.random.shuffle(block_vector)

            # Reshape and place it into the final image
            block_matrix = block_vector.reshape(len1, len2).astype(np.uint8)
            mf_image[r_start:r_end, c_start:c_end] = block_matrix

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
        if abs(q_val - 1) < 1e-9:  # Check for q = 1
            Dqtheory[currq] = 2 * np.log2(b + 1) - (2 * b * np.log2(b)) / (b + 1)
        else:
            # Formula for generalized dimensions Dq
            Dqtheory[currq] = (2 * np.log2(b ** q_val + 1) - 2 * q_val * np.log2(b + 1)) / (1 - q_val)

    # Calculate tau(q)
    tauq = (q - 1) * Dqtheory

    # Calculate alpha (singularity exponent) using numerical differentiation
    # alpha(q) = d(tau)/d(q)
    alpha_theory = np.zeros(len(Dqtheory))

    # Forward difference for the first point
    alpha_theory[0] = (tauq[1] - tauq[0]) / h
    # Backward difference for the last point
    alpha_theory[-1] = (tauq[-1] - tauq[-2]) / h

    # Central difference for interior points
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
    K = 6  # Number of iterations (up to log2(N) for pixel level)

    print(f"Running mfp1p2 with N={N}, p1={P1}, p2={P2}, k={K}")

    mf_image, alpha_theory, f_theory = mfp1p2(N, P1, P2, K)

    print("\n--- Results ---")

    print("\nTheoretical Spectrum (alpha_theory vs f_theory):")
    # Find the range of alpha and the maximum f(alpha)
    max_f_idx = np.argmax(f_theory)

    print(f"  alpha_min: {alpha_theory.min():.4f}")
    print(f"  alpha_max: {alpha_theory.max():.4f}")
    print(
        f"  Max f(alpha) = {f_theory[max_f_idx]:.4f} at alpha = {alpha_theory[max_f_idx]:.4f} (This should be close to 1)")

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
