# Python sessile drop analysis
Made by Mathijs van Gorcum during his PhD at the Physics of Fluids group of the University of Twente.
Edited by Michael Orella during his PhD in Chemical Engineering at MIT

This script analyzes black and white image files for measuring contact angles by fitting circles to edges detected using the canny edge detection algorithm in scikit-image. With the fitted circle and baseline, other analyses could be performed, such as contact line speed or volume of the droplet over time (in the case a video file is provided). 

## Prerequisites
The script requires numpy, scipy, matplotlib, and skimage.

## Running the script
To run, use analysis.py, with arguments for the filename being analyzed, and numerical thresholds as optional parameters

## Contributing
Feel free to send pull requests, critique my awful code or point out any issues.

## License
This project is licensed under the GPLv3 license - see the [LICENSE](https://github.com/michaelorella/Sessile.drop.analysis/blob/master/LICENSE) file for details
