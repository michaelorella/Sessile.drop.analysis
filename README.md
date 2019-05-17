# Python sessile drop analysis
Made by Mathijs van Gorcum during his PhD at the Physics of Fluids group of the University of Twente.
Edited by Michael Orella during his PhD in Chemical Engineering at MIT

This script analyzes black and white image files for measuring contact angles by fitting circles to edges detected using the canny edge detection algorithm in scikit-image. With the fitted circle and baseline, other analyses could be performed, such as contact line speed or volume of the droplet over time (in the case a video file is provided). 

## Prerequisites
The script requires numpy, scipy, matplotlib, and skimage. These can be added by default from the environment.yml file that is provided. Installation can be performed by cloning the repository to your local machine and creating a duplicate environment.

## Running the script
The script is run from the terminal. The first parameter passed to the script must be the filename for the video or images being analyzed. Both full and relative paths are allowed. After that, several flag-based parameters are allowed

-l or --lim : The maximum number of iterations of the circle fitting that should be performed. (DEFAULT = 10)

-b or --baselineThreshold : The number of pixels in from the edge of the crop that will be used to identify the baseline (DEFAULT = 20)

-c or --circleThreshold : The number of pixels above the baseline that we consider to be points on the circle (DEFAULT = 5)

-t or --times : How frequently (in seconds) the video should be analyzed. (DEFAULT = 1)

-s : The initial value for the Gaussian noise filter for the Canny edge detection algorithm (DEFAULT = 5)

-ss or --startSeconds : The time of the video file at which analysis should begin (DEFAULT = 10)

An example of running the script is shown below

```
>>> python analysis.m "../image_test.png" -s 10 -l 3
```

The output from the script will be a table of times, contact angles, drop volumes (in px ** 3), fit circle radii, and (negative) baseline heights

## Contributing
Feel free to send pull requests, critique my awful code or point out any issues.

## License
This project is licensed under the GPLv3 license - see the [LICENSE](https://github.com/michaelorella/Sessile.drop.analysis/blob/master/LICENSE) file for details


## Contributing
Feel free to send pull requests, critique my awful code or point out any issues.

## License
This project is licensed under the GPLv3 license - see the [LICENSE](https://github.com/michaelorella/Sessile.drop.analysis/blob/master/LICENSE) file for details
