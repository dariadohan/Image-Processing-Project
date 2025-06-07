#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace cv;
using namespace std;
#define PI 3.14159265358979323846


 //Manual grayscale conversion
Mat customGrayscale(const Mat& src) {
    Mat gray(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3b pixel = src.at<Vec3b>(y, x);
            uchar value = static_cast<uchar>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            gray.at<uchar>(y, x) = value;
        }
    }
    return gray;
}

// Simple mean threshold (instead of adaptive)
Mat customThreshold(const Mat& gray, int blockSize = 15, int C = 10) {
    Mat result = Mat::zeros(gray.size(), CV_8UC1);
    int offset = blockSize / 2;

    for (int y = offset; y < gray.rows - offset; y++) {
        for (int x = offset; x < gray.cols - offset; x++) {
            int sum = 0;
            for (int dy = -offset; dy <= offset; dy++) {
                for (int dx = -offset; dx <= offset; dx++) {
                    sum += gray.at<uchar>(y + dy, x + dx);
                }
            }
            int area = blockSize * blockSize;
            int mean = sum / area;
            result.at<uchar>(y, x) = (gray.at<uchar>(y, x) < mean - C) ? 255 : 0;
        }
    }

    return result;
}

// Dilation
Mat customDilate(const Mat& src, int kernelSize = 5) {
    int offset = kernelSize / 2;
    Mat result = src.clone();
    for (int y = offset; y < src.rows - offset; y++) {
        for (int x = offset; x < src.cols - offset; x++) {
            uchar maxVal = 0;
            for (int dy = -offset; dy <= offset; dy++) {
                for (int dx = -offset; dx <= offset; dx++) {
                    maxVal = max(maxVal, src.at<uchar>(y + dy, x + dx));
                }
            }
            result.at<uchar>(y, x) = maxVal;
        }
    }
    return result;
}

// Erosion
Mat customErode(const Mat& src, int kernelSize = 5) {
    int offset = kernelSize / 2;
    Mat result = src.clone();
    for (int y = offset; y < src.rows - offset; y++) {
        for (int x = offset; x < src.cols - offset; x++) {
            uchar minVal = 255;
            for (int dy = -offset; dy <= offset; dy++) {
                for (int dx = -offset; dx <= offset; dx++) {
                    minVal = min(minVal, src.at<uchar>(y + dy, x + dx));
                }
            }
            result.at<uchar>(y, x) = minVal;
        }
    }
    return result;
}

// Morphological close: dilate then erode
Mat customMorphClose(const Mat& src, int kernelSize = 5) {
    return customErode(customDilate(src, kernelSize), kernelSize);
}

// Function to detect the 4 corners of the paper
vector<Point> findPaperCorners(const Mat& src) {
    Mat gray = customGrayscale(src);
    Mat thresh = customThreshold(gray);
    Mat morph = customMorphClose(thresh);

    vector<vector<Point>> contours;
    findContours(morph, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Point> bestApprox;
    double maxArea = 0;

    for (const auto& contour : contours) {
        double peri = arcLength(contour, true);
        vector<Point> approx;
        approxPolyDP(contour, approx, 0.02 * peri, true);

        double area = contourArea(approx);
        if (approx.size() >= 4 && area > maxArea && isContourConvex(approx)) {
            maxArea = area;
            bestApprox = approx;
        }
    }

    return bestApprox;
}

// Sort corners in TL, TR, BR, BL order
vector<Point2f> sortCorners(const vector<Point>& corners) {
    vector<Point2f> sorted(4);
    vector<Point> temp = corners;

    sort(temp.begin(), temp.end(), [](const Point& a, const Point& b) {
        return a.y < b.y;
    });

    Point2f topLeft, topRight, bottomLeft, bottomRight;

    if (temp[0].x < temp[1].x) {
        topLeft = temp[0];
        topRight = temp[1];
    } else {
        topLeft = temp[1];
        topRight = temp[0];
    }

    if (temp[2].x < temp[3].x) {
        bottomLeft = temp[2];
        bottomRight = temp[3];
    } else {
        bottomLeft = temp[3];
        bottomRight = temp[2];
    }

    sorted[0] = topLeft;
    sorted[1] = topRight;
    sorted[2] = bottomRight;
    sorted[3] = bottomLeft;

    return sorted;
}

// Warp and display aligned paper
void alignPaper(const Mat& src, const vector<Point>& corners) {
    if (corners.size() != 4) {
        cout << "Nu s-au detectat exact 4 colțuri pentru aliniere!" << endl;
        return;
    }

    vector<Point2f> sortedCorners = sortCorners(corners);

    float widthA = norm(sortedCorners[2] - sortedCorners[3]);
    float widthB = norm(sortedCorners[1] - sortedCorners[0]);
    float maxWidth = max(widthA, widthB);

    float heightA = norm(sortedCorners[1] - sortedCorners[2]);
    float heightB = norm(sortedCorners[0] - sortedCorners[3]);
    float maxHeight = max(heightA, heightB);

    vector<Point2f> dstCorners = {
            Point2f(0, 0),
            Point2f(maxWidth - 1, 0),
            Point2f(maxWidth - 1, maxHeight - 1),
            Point2f(0, maxHeight - 1)
    };

    Mat transformMatrix = getPerspectiveTransform(sortedCorners, dstCorners);
    Mat warped;
    warpPerspective(src, warped, transformMatrix, Size((int)maxWidth, (int)maxHeight));

    imshow("Foaie Aliniata", warped);
}

int main() {
    Mat image = imread("C:/Users/daria/Downloads/paper5.jpg");
    if (image.empty()) {
        cout << "Eroare la incarcarea imaginii." << endl;
        return -1;
    }

    vector<Point> corners = findPaperCorners(image);

    if (corners.size() >= 4) {
        if (corners.size() > 4) {
            RotatedRect box = minAreaRect(corners);
            Point2f boxPoints[4];
            box.points(boxPoints);
            corners = vector<Point>(boxPoints, boxPoints + 4);
        }

        for (int i = 0; i < 4; i++) {
            line(image, corners[i], corners[(i + 1) % 4], Scalar(0, 255, 0), 2);
            circle(image, corners[i], 5, Scalar(0, 0, 255), -1);
        }

        imshow("Colturi detectate", image);
        alignPaper(image, corners);
    } else {
        cout << "Nu s-au detectat 4 colțuri." << endl;
        imshow("Imagine originală", image);
    }

    waitKey(0);
    return 0;
}

/*White paper alignment based on line and corner detection [Daria Dohan]
Given an image of a white paper, with possibly written text on it, find
the corners of the paper and the transformation which transforms the paper
into a rectangle. */