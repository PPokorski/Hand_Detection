#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

//Binary thresh for binary threshold
const int Binary_thresh = 230;
//Size of kernel for morhpology open
const int Morphology_kernel_size = 12;
const Size Size_v(Morphology_kernel_size, Morphology_kernel_size);
//Minimum contour size, which can be hand
const int Min_contour_size = 2000;
//Minimum standard deviation for counting rough palm cener
const double Min_rough_center_std_dev = 400.0;
//Minimum denominator while counting a circle from points, provided to avoid too big numbers
const double Min_circle_denominator = 0.1;
//The number of elements, for which the average (and final) palm center is calculated
const int Max_palm_centers_number = 10;
//The two following consts are threshes for a band-stop filter
//We don't consider palm centers, which are closer to final palm center than this value
const int Upper_center_change_thresh = 100;
//We don't consider palm centers, which are farther to final palm center than this value
const int Lower_center_change_thresh = 40;
//We don't consider palm centers, whose radius are greater than this value
const int Maximum_center_radius = 400;

//Function to find the biggest contour of all found
void findBiggestContour(vector<vector<Point> > contours, int &max_area, int &max_index)
{
    for(int i = 0; i< contours.size(); i++)
    {
        int current_area = contourArea(contours.at(i));

        if(current_area > max_area)
        {
            max_area = current_area;
            max_index = i;
        }
    }
}

//Function to find convex hull and its defects for provided contour
void findHullsAndDefects(vector<Point> contour, vector<Point> &hull, vector<Vec4i> &defects)
{
    vector<int> hull_ints;

    convexHull(contour, hull);
    convexHull(contour, hull_ints);
    if(hull_ints.size() > 3)
    {
        convexityDefects(contour, hull_ints, defects);
    }
}

//Function to draw contour, convex hull and its defect points
void drawContourAndHull(Mat frame, vector<Point> contour, vector<Point> hull, vector<Point> palm_points, Point center, double radius)
{
    vector<vector<Point> > contours(1, contour);
    vector<vector<Point> > hulls(1, hull);

    drawContours(frame, hulls, 0, Scalar(0, 0, 255), 1);
    drawContours(frame, contours, 0, Scalar(0, 0, 255), 1);

    for(int i = 0; i < palm_points.size(); i++)
    {
        circle(frame, palm_points.at(i), 5, Scalar(0, 0, 255), -1);
    }

    circle(frame, center, 10, Scalar(255, 255, 255), -1);
    circle(frame, center, radius, Scalar(255, 255, 255));
}

double distancePoints(Point x, Point y)
{
    return pow(x.x - y.x, 2) + pow(x.y - y.y, 2);
}

//Function to find the center and radius of a circle from 3 points
pair<Point, double> circleFromPoints(Point p1, Point p2, Point p3)
{
    double aa = pow(p1.x, 2) + pow(p1.y, 2);
    double bb = pow(p2.x, 2) + pow(p2.y, 2);
    double cc = pow(p3.x, 2) + pow(p3.y, 2);

    double nominator_x = p3.y*(bb - aa) + p2.y*(aa - cc) + p1.y*(cc - bb);
    double nominator_y = p3.x*(aa - bb) + p2.x*(cc - aa) + p1.x*(bb - cc);

    double denominator = 2.0*(p1.y*(p3.x - p2.x) + p2.y*(p1.x - p3.x) + p3.y*(p2.x - p1.x));

    if(abs(denominator) < Min_circle_denominator)
    {
        return make_pair(Point(0, 0), 0);
    }

    double x_center = nominator_x/denominator;
    double y_center = nominator_y/denominator;

    Point center(x_center, y_center);

    double radius = sqrt(distancePoints(p1, center));

    return make_pair(center, radius);
}

//Find the average point and radius from the vector of given
pair<Point, double> meanStdDeviation(vector<pair<Point, double> >vector_of)
{
    Point average_point;
    double average_radius = 0;

    for(int i = 0; i < vector_of.size(); i++)
    {
        average_point += vector_of.at(i).first;
        average_radius += vector_of.at(i).second;
    }
    if(vector_of.size() != 0)
    {
        average_point.x /= vector_of.size();
        average_point.y /= vector_of.size();
        average_radius /= vector_of.size();
    }
    else
    {
        average_point.x = 0;
        average_point.y = 0;
        average_radius = 0;
    }
    return make_pair(average_point, average_radius);
}

//Overloaded function of the previous one. The difference is that it calculates standard deviation
Point meanStdDeviation(vector<Point> vector_of, double &std_dev)
{
    std_dev = 0;

    Point average_point;

    for(int i = 0; i < vector_of.size(); i++)
    {
        average_point += vector_of.at(i);
    }
    if(vector_of.size() != 0)
    {
        average_point.x /= vector_of.size();
        average_point.y /= vector_of.size();

        for(int i = 0; i < vector_of.size(); i++)
        {
            std_dev += distancePoints(vector_of.at(i), average_point);
        }

        std_dev /= vector_of.size();
        std_dev = sqrt(std_dev);
    }
    else
    {
        average_point.x = 0;
        average_point.y = 0;
    }


    return average_point;
}

int main (int argc, char** argv)
{
    VideoCapture cap(0);
    if(!cap.isOpened())
        return -1;

    Mat frame;
    //The foreground mask
    Mat fg_mask_mog2;

    Ptr<BackgroundSubtractor> p_mog2;
    p_mog2 = new BackgroundSubtractorMOG2();

    vector<vector<Point> > contours;
    vector<pair<Point, double> > palm_centers;

    //Number of frames to create a foreground model
    int background_frame = 500;


    while(true)
    {
        cap >> frame; // get a new frame from camera

        if(background_frame > 0)
        {
            p_mog2->operator ()(frame, fg_mask_mog2);
            background_frame--;
            putText(frame, "Recording background", Point(30, 30), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 0, 255));
        }
        else
        {
            p_mog2->operator ()(frame, fg_mask_mog2, 0);
            putText(frame, "Background recorded", Point(30, 30), FONT_HERSHEY_COMPLEX, 0.8, Scalar(0, 255, 0));
        }


        threshold(fg_mask_mog2, fg_mask_mog2, Binary_thresh, 255, THRESH_BINARY);

        morphologyEx(fg_mask_mog2, fg_mask_mog2, MORPH_OPEN, getStructuringElement(CV_SHAPE_ELLIPSE, Size_v));


        Mat processing;
        fg_mask_mog2.copyTo(processing);

        findContours(processing, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        //The area of the biggest contour
        int max_area = 0;
        //The index of the biggest contour
        int max_index = 0;

        if(!contours.empty())
        {
            findBiggestContour(contours, max_area, max_index);

            if(max_area > Min_contour_size)
            {
                //The palm center calculated as an average of all defect points
                Point rough_palm_center;

                vector<Point> contour;
                contour = contours.at(max_index);
                vector<Point> hull_points;
                vector<Vec4i> defect;

                findHullsAndDefects(contour, hull_points, defect);

                //We need at least three defect points to find a hand
                if(defect.size() >= 3)
                {
                    vector<Point> palm_points;

                    for(int i = 0; i < defect.size(); i++)
                    {
                        int start_id = defect.at(i)[0]; Point point_start(contour.at(start_id));
                        int end_id = defect.at(i)[1]; Point point_end(contour.at(end_id));
                        int far_id = defect.at(i)[2]; Point point_far(contour.at(far_id));

                        palm_points.push_back(point_start);
                        palm_points.push_back(point_end);
                        palm_points.push_back(point_far);
                    }


                    //Remove all defect points that are too far away from a rough palm center, because it may be e.g. a forearm
                    double std_dev;
                    double std_dev_prev;

                    do
                    {
                        rough_palm_center = meanStdDeviation(palm_points, std_dev);
                        std_dev_prev = std_dev;

                        for(int i = 0; i < palm_points.size(); i++)
                        {
                            if(distancePoints(rough_palm_center, palm_points.at(i)) > 2*std_dev*std_dev)
                            {
                                palm_points.erase(palm_points.begin() + i);
                                i--;
                            }
                        }

                        rough_palm_center = meanStdDeviation(palm_points, std_dev);
                    }
                    while(std_dev > Min_rough_center_std_dev);


                    //The vector contains indices and distances of the defect points from the rough palm center
                    vector<pair<double, int> > distances_vector;

                    for(int i = 0; i < palm_points.size(); i++)
                    {
                        distances_vector.push_back(make_pair(distancePoints(rough_palm_center, palm_points[i]), i));
                    }

                    sort(distances_vector.begin(), distances_vector.end());

                    pair<Point, double> palm_circle;
                    pair<Point, double> tmp;

                    //Find the circle from the 3 points which are closest to the rough palm center. It assumes that defect points (the "valleys" between fingers) lie on a circle around the hand center.
                    //We do it because it's less vulnarable to weird defect points, for which the rough palm center is calculated
                    for(int i = 0; i + 2 < distances_vector.size(); i++)
                    {
                        Point p1 = palm_points.at(distances_vector.at(i + 0).second);
                        Point p2 = palm_points.at(distances_vector.at(i + 1).second);
                        Point p3 = palm_points.at(distances_vector.at(i + 2).second);

                        palm_circle = circleFromPoints(p1, p2, p3);

                        if(palm_circle.second != 0) break;
                    }

                    //We provide band-stop filter to stabilize the final palm_center
                    tmp = meanStdDeviation(palm_centers);

                    if((distancePoints(tmp.first, palm_circle.first) > Upper_center_change_thresh*Upper_center_change_thresh || distancePoints(tmp.first, palm_circle.first) < Lower_center_change_thresh*Lower_center_change_thresh) && palm_circle.second < Maximum_center_radius)
                    {
                        palm_centers.push_back(palm_circle);
                        if(palm_centers.size() > Max_palm_centers_number)
                        {
                            palm_centers.erase(palm_centers.begin());
                        }
                    }

                    Point palm_center;
                    double radius = 0;

                    tmp = meanStdDeviation(palm_centers);

                    palm_center = tmp.first;
                    radius = tmp.second;

                    drawContourAndHull(frame, contour, hull_points, palm_points, palm_center, radius);
                }
            }
        }

        imshow("Camera", frame);
        //imshow("Foreground", fg_mask_mog2);

        if(waitKey(10) >= 0) break;
    }

    return 0;
}
