# Hand detection #

This is the project for Machine Vision classes at Faculty of Mechatronics, Warsaw University of Technology.

The aim was to write a program which would detect and track a hand. 

## Required conditions ##

The program works pretty well provided that following conditions are met:

* The background is relatively static (dynamic background messes everything)
* Works best with bright stray light (for high contrast)
* Only one hand is visible
* All fingers of the hand are visible
* Generally program works better if the face isn't in the image

## Algorithm ##

1. First 500 frames are used to create a good [BackgroundSubstractor model](http://docs.opencv.org/trunk/d1/dc5/tutorial_background_subtraction.html)
2. Binary Threshold
3. Morphological open
4. Finding contours with `findContours()` function
5. Finding the contour with greatest area
6. Finding the [hulls](http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#convexhull) and [defects](http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#convexitydefects) of the contour
7. Count the `rough_palm_center` as a mean of defect points, also count standard deviation
8. Remove all defect points which are father than 1,41*std_deviation from `rough_palm_center`
9. Go back to step 7 until `std_dev > Min_rough_center_std_dev`
10. Make vector of distances of defects to `rough_palm_center`
11. Sort the vector
12. Take the three closest defects and count the center and radius of a circle containing them
13. Check if the center didn't move too far away and whether the radius isn't too big
14. If it's ok, count the mean of 10 last palm centers
15. Resulting circle is the center of palm and size of the hand.

## Contact ##
Piotr Pokorski

piotr.pokorski94@gmail.com