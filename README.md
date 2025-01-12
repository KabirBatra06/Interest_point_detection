# Interest_point_detection

Here I implement an automated approach for
interest point detection and correspondence search for a given pair of images of the same scene.


For the detection part, I do the following:
1. Implement my own Harris Corner Detection algorithm.
2. Test the SIFT or SURF implementation that are available in OpenCV.
3. Test the CNN-based SuperPoint interest point detector.

   
And to establish the point-to-point correspondences between the two views, I do the following:
1. Implement my own functions for computing the SSD (Sum of Squared Differences) and the NCC
(Normalized Cross Correlation) as the feature similarity measures.
2. Test the GNN-based SuperGlue feature matching network.

# Harris Corner Detector

The Harris corner detector defines a corner as a pixel around which there is a significant variation in gray scale values in at least two directions. The Harris corner detector possesses invariance to in-plane rotations of the image. This makes it a great operator for detecting interest points and then matching those interest points in different images of the same scene.

## Gradient Calculation

Harris corner detection begins by calculating the gradient of grayscale values in both the x and y directions of the image. The calculation of this gradient is done using the x-oriented and y-oriented Haar filters.

The size of the Haar filters depends on the value of sigma chosen. Generally, the size of the Haar filter is \( M \times M \), where \( M \) is the smallest even integer greater than four times the sigma value.

For a sigma value of 0.8, the x-oriented Haar filter looks like:

$$
4 \times \sigma = 3.2 \implies M = 4
$$

$$
\text{Haar}_x =
\begin{bmatrix}
-1 & -1 & 1 & 1 \\
-1 & -1 & 1 & 1 \\
-1 & -1 & 1 & 1 \\
-1 & -1 & 1 & 1
\end{bmatrix}
$$

$$
\text{Haar}_y =
\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 \\
-1 & -1 & -1 & -1 \\
-1 & -1 & -1 & -1
\end{bmatrix}
$$

Applying these filters to the original grayscale images gives us the gradients of the pixel values in the x and y directions.

## Constructing the \( C \) Matrix

In order to detect the presence of a corner at a pixel, we construct the \( C \) matrix in a \( 5\sigma \times 5\sigma \) neighborhood around the pixel. The \( C \) matrix is defined as:

$$
C =
\begin{bmatrix}
\Sigma d_x^2 & \Sigma d_x d_y \\
\Sigma d_x d_y & \Sigma d_y^2
\end{bmatrix}
$$

The eigenvalues of this \( C \) matrix are used to classify a pixel as a corner. When the \( C \) matrix is full rank, the pixel is a true corner. For pixels that lie on edges, this matrix has a rank of 1.

## Characterizing a Pixel as a Corner Point

Let’s say that the two eigenvalues of the \( C \) matrix are \( \lambda_1 \) and \( \lambda_2 \), where \( \lambda_1 \geq \lambda_2 \).

We know:

$$
\text{Tr}(C) = \Sigma d_x^2 + \Sigma d_y^2 = \lambda_1 + \lambda_2
$$

$$
\text{det}(C) = \Sigma d_x^2 \cdot \Sigma d_y^2 - (\Sigma d_x d_y)^2 = \lambda_1 \cdot \lambda_2
$$

Using these values, we calculate the Harris Corner Score as:

$$
R = \text{det}(C) - k(\text{Tr}(C))^2
$$

For my implementation, I set the value of \( k \) to 0.08.

To identify the best corners, I only consider the highest \( R \) value in a window that is the same size as the summation neighborhood of \( 5\sigma \times 5\sigma \). This highest \( R \) value is checked to see whether it is greater than the mean of all the \( R \) values found in the image. For corner scores that meet both criteria, the pixel is considered a corner by my implementation.

## Establishing Correspondence

We are able to establish correspondence between the corners detected in two images by comparing the grayscale pixel values around the corner points. We choose a window size for which we want to compare the surrounding grayscale values. For my implementation, I use a neighboring pixel grid of \( 10 \times 10 \).

### Using SSD

For the Sum of Squared Differences (SSD) method, the neighboring pixels are compared using the following formula:

$$
\text{SSD} = \sum_i \sum_j \left| f_1(i,j) - f_2(i,j) \right|^2
$$

Here:  
- \( f_1(x, y) \) is the grayscale value of the pixel located at \( (x, y) \) in the first image.  
- \( f_2(x, y) \) is the grayscale value of the pixel located at \( (x, y) \) in the second image.

The summation goes over all the pixels in the \( 10 \times 10 \) grid.

My implementation of SSD correspondence calculates the SSD for all pairs of interest points in the two images. The point in the second image that has the lowest SSD value for a point in the first image is deemed the corresponding interest point of the first image in the second image.

### Using NCC

For the Normalized Cross-Correlation (NCC) method, the neighboring pixels are compared using the following formula:

$$
\text{NCC} = \frac{\sum_i \sum_j (f_1(i,j) - m_1)(f_2(i,j) - m_2)}{\sqrt{\left[\sum_i \sum_j (f_1(i,j) - m_1)^2\right] + \left[\sum_i \sum_j (f_2(i,j) - m_2)^2\right]}}
$$

Here:  
- \( f_1(x, y) \) is the grayscale value of the pixel located at \( (x, y) \) in the first image.  
- \( f_2(x, y) \) is the grayscale value of the pixel located at \( (x, y) \) in the second image.  
- \( m_1 \) is the mean of the grayscale values in the \( 10 \times 10 \) grid of the first image.  
- \( m_2 \) is the mean of the grayscale values in the \( 10 \times 10 \) grid of the second image.

The summation goes over all the pixels in the \( 10 \times 10 \) grid.

My implementation of NCC correspondence calculates the NCC for all pairs of interest points in the two images. The point in the second image that has the highest value of NCC for a point in the first image is deemed the corresponding interest point of the first image in the second image. I also ensure that the NCC score is greater than 0.3 so that the matches are more accurate, and false positives are more likely to be discarded.

## Results

<p align="center">
  <img src="https://github.com/KabirBatra06/Interest_point_detection/blob/main/hovde/hovde_sift.jpg" width="350" title="Interest point using SIFT">
  <img src="https://github.com/KabirBatra06/Interest_point_detection/blob/main/hovde/hovde_ssd2.jpg" width="350" title="Interest point using SSD">
 <br>
  <img src="https://github.com/KabirBatra06/Interest_point_detection/blob/main/rawls/rawls_sift.jpg" width="350" title="Interest point using SIFT">
  <img src="https://github.com/KabirBatra06/Interest_point_detection/blob/main/rawls/rawls_ssd1.4.jpg" width="350" title="Interest point using SSD">
 <br>
  <img src="https://github.com/KabirBatra06/Interest_point_detection/blob/main/temple/temple_sift.jpg" width="350" title="Interest point using SIFT">
  <img src="https://github.com/KabirBatra06/Interest_point_detection/blob/main/temple/temple_ssd1.4.jpg" width="350" title="Interest point using SSD">
  <br>
  <img src="https://github.com/KabirBatra06/Interest_point_detection/blob/main/socket/socket_sift.jpg" width="350" title="Interest point using SIFT">
  <img src="https://github.com/KabirBatra06/Interest_point_detection/blob/main/socket/socket_ncc2.jpg" width="350" title="Interest point using NCC">
</p>
