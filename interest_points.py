import math
import numpy as np
import cv2
import random

#################################### Function to make Haar filter for X-direction ##########################
def x_haar_generator(sigma):
    sig4 = math.ceil(4*sigma)
    if sig4 % 2:
        sig4+=1
    
    left = -1 * np.ones((int(sig4), int(sig4/2)))
    right = np.ones((int(sig4), int(sig4/2)))
    haar = np.hstack((left, right))

    return haar

#################################### Function to make Haar filter for Y-direction ##########################
def y_haar_generator(sigma):
    sig4 = math.ceil(4*sigma)
    if sig4 % 2:
        sig4+=1
    
    down = -1 * np.ones((int(sig4/2), int(sig4)))
    up = np.ones((int(sig4/2), int(sig4)))
    haar = np.vstack((up, down))

    return haar

#################################### Function to Calculate Deravatives of Image Using Haar operator ##########################
def deravative_calc(img, filter_x, filter_y):
    dx = cv2.filter2D(src=img, ddepth=-1 , kernel=filter_x)
    dy = cv2.filter2D(src=img, ddepth=-1 , kernel=filter_y)
    dx2 = np.multiply(dx,dx)
    dy2 = np.multiply(dy,dy)
    dxy = np.multiply(dx,dy)

    return dx2, dy2, dxy 

#################################### Function to Sum up a 5*sigma X 5*sigma subset of a matrix  ##########################
def summation_calc(mat, sigma):
    sig5 = math.ceil(5*sigma)
    window = np.ones((sig5, sig5))
    ans = cv2.filter2D(src=mat, ddepth=-1 , kernel=window)

    return ans

#################################### Function to find Harris Corner points ##########################
def apply_harris(image, sigma, out_image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255 # making image grayscale and normalizing values
    y_h = y_haar_generator(sigma)
    x_h = x_haar_generator(sigma)

    dx2, dy2, dxy = deravative_calc(img, x_h, y_h) # calculating the x, y and dxdy deravatives

    # Calculating the C matrix elements 
    c1 = summation_calc(dx2, sigma)
    c23 = summation_calc(dxy, sigma)
    c4 = summation_calc(dy2, sigma)

    # Finding determinant and trace of C
    tr2 = np.multiply((c1 + c4), (c1 + c4))
    dt = (np.multiply(c1,c4)) - np.multiply(c23,c23)

    # Setting up threshold values
    r = dt - (0.08 * tr2)
    threshold = np.mean(np.abs(r))

    N = math.ceil(5*sigma) # Window in which max R is found
    points = np.empty((0, 3))
    img_copy = np.copy(image)

    # Finding all corner points
    for left in range(img.shape[1] - 2*N):
        for up in range(img.shape[0] - 2*N):
            max_r = np.max(r[up:up+(2*N)+1 , left:left+(2*N)+1])
            if r[up + N , left + N] == max_r and r[up + N , left + N] > threshold:
                points = np.vstack((points, np.array([left + N ,up + N ,r[up + N , left + N]])))
    
    # Selecting 50 corner points with highest R values
    best_points = np.int_(np.array(sorted(points, key = lambda point:point[2]))[-50:,:2])

    # Plotting points on image                       
    for point in best_points :
        cv2.circle(img_copy,(point[0], point[1]), radius=5, color=(0 ,0, 255), thickness =-1)

    # Saving image and returning best points
    cv2.imwrite(out_image, img_copy)
    return best_points

#################################### Function to Calculate NCC correspondence ##########################
def NCC_correspondence(image_1, image_2, ip_1, ip_2, output):
    img1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY) / 255
    img2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY) / 255

    window = 4 # Window for checking similarity in surrounding pixel values
    corres_ip2 = np.zeros_like(ip_1)

    # Finding point in second image with highest NCC for interest point in first image 
    for i, p1 in enumerate(ip_1):
        area_1 = img1[p1[1]-window:p1[1]+window, p1[0]-window:p1[0]+window]
        m1 = np.mean(area_1)
        ncc_max = 0
        for p2 in ip_2:
            area_2 = img2[p2[1]-window:p2[1]+window, p2[0]-window:p2[0]+window]
            m2 = np.mean(area_2)
            ncc = np.sum((area_1 - m1) * (area_2 - m2)) / np.sqrt((np.sum((area_1 - m1) ** 2)) * (np.sum((area_2 - m2) ** 2)))
            if ncc > ncc_max and ncc > 0.3:
                ncc_max = ncc
                corres_ip2[i] = p2
    
    corres_ip2[:, 0] += img1.shape[1] # offsetting interest points in second image for plotting purpose
    combo_image = np.concatenate((image_1, image_2), axis=1)

    # Plotting correspondences
    for i in range(len(ip_1)):
        cv2.circle(combo_image, ip_1[i], radius=5, color=(0, 0, 255), thickness=-1)
        cv2.circle(combo_image, corres_ip2[i], radius=5, color=(0 ,0, 255), thickness = -1)
        cv2.line(combo_image, ip_1[i], corres_ip2[i], (random.randrange(255) ,random.randrange(255) ,random.randrange(255)), 1)

    cv2.imwrite(output, combo_image)

def SSD_correspondence(image_1, image_2, ip_1, ip_2, output):
    img1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY) / 255
    img2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY) / 255

    window = 4 # Window for checking similarity in surrounding pixel values
    corres_ip2 = np.zeros_like(ip_1)

    # Finding point in second image with lowest SSD for interest point in first image 
    for i, p1 in enumerate(ip_1):
        area_1 = img1[p1[1]-window:p1[1]+window, p1[0]-window:p1[0]+window]
        ssd_min = 999999
        for p2 in ip_2:
            area_2 = img2[p2[1]-window:p2[1]+window, p2[0]-window:p2[0]+window]
            ssd = np.sum((area_1 - area_2) ** 2)
            if ssd < ssd_min:
                ssd_min = ssd
                corres_ip2[i] = p2
    
    corres_ip2[:, 0] += img1.shape[1] # offsetting interest points in second image for plotting purpose
    combo_image = np.concatenate((image_1, image_2), axis=1)

    # Plotting correspondences
    for i in range(len(ip_1)):
        cv2.circle(combo_image, ip_1[i], radius=5, color=(0, 0, 255), thickness=-1)
        cv2.circle(combo_image, corres_ip2[i], radius=5, color=(0, 0, 255), thickness = -1)
        cv2.line(combo_image, ip_1[i], corres_ip2[i], (random.randrange(255) ,random.randrange(255) ,random.randrange(255)), 1)

    cv2.imwrite(output, combo_image)

#################################### Function to Impliment SIFT using CV2 Library ##########################
def sift(image1, image2, output):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift_obj = cv2.SIFT_create()
    matching_criteria = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    interest_points1, desc_1 = sift_obj.detectAndCompute(img1, None)
    interest_points2, desc_2 = sift_obj.detectAndCompute(img2, None)

    corres_features = matching_criteria.match(desc_1, desc_2)
    corres_features = sorted(corres_features, key = lambda x:x.distance)

    img1 = cv2.drawMatches(image1, interest_points1, image2, interest_points2, corres_features[:100], None)

    cv2.imwrite(output, img1)


#################################### Function Calls ##########################

# Openning all images
temple_1 = cv2.imread("temple_1.jpg")
temple_2 = cv2.imread("temple_2.jpg")

hovde_1 = cv2.imread("hovde_2.jpg")
hovde_2 = cv2.imread("hovde_3.jpg")

socket_1 = cv2.imread("socket1.jpg")
socket_2 = cv2.imread("socket2.jpg")

rawls_1 = cv2.imread("rawls1.jpg")
rawls_2 = cv2.imread("rawls2.jpg")

for sigma in [0.5, 0.8, 1.4, 2]: # going over 4 sigma values
    temple1_interest_points = apply_harris(temple_1, sigma, "temple/temple1_points" + str(sigma) + ".jpg")
    temple2_interest_points = apply_harris(temple_2, sigma, "temple/temple2_points" + str(sigma) + ".jpg")

    hovde1_interest_points = apply_harris(hovde_1, sigma, "hovde/hovde1_points" + str(sigma) + ".jpg")
    hovde2_interest_points = apply_harris(hovde_2, sigma, "hovde/hovde2_points" + str(sigma) + ".jpg")

    rawls1_interest_points = apply_harris(rawls_1, sigma, "rawls/rawls1_points" + str(sigma) + ".jpg")
    rawls2_interest_points = apply_harris(rawls_2, sigma, "rawls/rawls2_points" + str(sigma) + ".jpg")

    socket1_interest_points = apply_harris(socket_1, sigma, "socket/socket1_points" + str(sigma) + ".jpg")
    socket2_interest_points = apply_harris(socket_2, sigma, "socket/socket2_points" + str(sigma) + ".jpg")

    NCC_correspondence(temple_1, temple_2, temple1_interest_points, temple2_interest_points, "temple/temple_ncc" + str(sigma) + ".jpg")
    NCC_correspondence(hovde_1, hovde_2, hovde1_interest_points, hovde2_interest_points, "hovde/hovde_ncc" + str(sigma) + ".jpg")
    NCC_correspondence(socket_1, socket_2, socket1_interest_points, socket2_interest_points, "socket/socket_ncc" + str(sigma) + ".jpg")
    NCC_correspondence(rawls_1, rawls_2, rawls1_interest_points, rawls2_interest_points, "rawls/rawls_ncc" + str(sigma) + ".jpg")

    SSD_correspondence(temple_1, temple_2, temple1_interest_points, temple2_interest_points, "temple/temple_ssd" + str(sigma) + ".jpg")
    SSD_correspondence(hovde_1, hovde_2, hovde1_interest_points, hovde2_interest_points, "hovde/hovde_ssd" + str(sigma) + ".jpg")
    SSD_correspondence(socket_1, socket_2, socket1_interest_points, socket2_interest_points, "socket/socket_ssd" + str(sigma) + ".jpg")
    SSD_correspondence(rawls_1, rawls_2, rawls1_interest_points, rawls2_interest_points, "rawls/rawls_ssd" + str(sigma) + ".jpg")

sift(temple_1, temple_2, "temple/temple_sift.jpg")
sift(hovde_1, hovde_2, "hovde/hovde_sift.jpg")
sift(rawls_1, rawls_2, "rawls/rawls_sift.jpg")
sift(socket_1, socket_2, "socket/socket_sift.jpg")