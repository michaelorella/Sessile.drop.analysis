#Import system for comman line processing
import sys

#Image processing import
import skimage
from skimage import feature
from skimage import io
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool

#Plotting
import matplotlib.pyplot as plt

#Numerical analysis
import numpy as np

#Optimization
import scipy as scipy
import scipy.optimize as opt

function, image, *kwargs = sys.argv

#Set default numerical arguments
lim = 10
baselineThreshold = 20
circleThreshold = 5

#Overwrite these defaults if desired
kwargs = zip ( * [ iter(kwargs) ] * 2 )

for argPair in kwargs:
	if argPair[0] == '-l' or argPair[0] == '--lim':
		lim = int(argPair[1])
	elif argPair[0] == '-b' or argPair[0] == '--baselineThreshold':
		baselineThreshold = int(argPair[1])
	elif argPair[0] == '-c' or argPair[0] == '--circleThreshold':
		circleThreshold = int(argPair[1])

### Strategy for automating contact angle measurements for Lauren+McLain

### Use some toolkit to extract every nth frame from avi file

### At every extracted frame, read it as a grayscale numpy array
# TODO!! - get frames from video
im = io.imread(image,as_gray = True)

### Using scikit-image canny edge detection, find the image edges
edges = feature.canny(im,sigma = 0, low_threshold = 0.1, high_threshold = 0.5)

# Obtain the X,Y coordinates of the True values in this edge image 
# (for processing)
coords = np.array([[i,j] for j,row in enumerate(edges) 
						 for i,x in enumerate(row) if x])

# Get 4-list of points for left, right, top, and bottom crop (in that order)

# Show the image so that user can select crop box
viewer = ImageViewer(im)
rect_tool = RectangleTool(viewer, on_enter = viewer.closeEvent)
viewer.show()
cropPoints = np.array(rect_tool.extents)
cropPoints = np.array(np.round(cropPoints),dtype = int)

# Crop the set of points that are going to be used for analysis
crop = np.array([[x,y] for x,y in coords if (x >= cropPoints[0] and
											 x <= cropPoints[1] and 
											 y >= cropPoints[2] and 
											 y <= cropPoints[3])])

# Get the baseline from the left and right threshold pixels of the image 
# (this is important not to crop too far)
baseline = {'l':np.array([[x,y] for x,y in coords 
								if (x >= cropPoints[0] and 
									x <= cropPoints[0] + baselineThreshold and 
									y >= cropPoints[2] and 
									y <= cropPoints[3] )]),
			'r':np.array([[x,y] for x,y in coords 
								if (x >= cropPoints[1] - baselineThreshold and 
									x <= cropPoints[1] and 
									y >= cropPoints[2] and 
									y <= cropPoints[3] )])}

# Fit the baseline to a line of form y = m*x + b using np.linalg
A = np.ones((baseline['l'].shape[0] + baseline['r'].shape[0],2))
A[:,1] = np.concatenate((baseline['l'][:,0],baseline['r'][:,0]))
c = np.concatenate((baseline['l'][:,1],baseline['r'][:,1]))
b,m = np.linalg.lstsq(A,c, rcond = None)[0]

# Now find the points in the circle
circle = np.array([(x,y) for x,y in crop if
										y - (m*x + b)  <= -circleThreshold])

# Define the loss function that we use for fitting
def dist(param, points):
	*z , r = param
	ar = [(np.linalg.norm(np.array(z) - np.array(point)) - r ) ** 2 
			for point in points]
	return np.sum(ar)

# Get the cropped image width
width = cropPoints[1] - cropPoints[0]

# Try to fit a circle to the points that we have extracted
res = opt.minimize ( lambda x: dist( x , circle ) , 
				     np.concatenate( ( np.mean ( circle, axis = 0) , 
				     				   [width/2] ) ) )

#Get the results
*z , r = res['x']

points = circle

iters = 0

# Keep retrying the fitting while the function value is large, as this 
# indicates that we probably have 2 circles (e.g. there's something light
# in the middle of the image)
while np.sqrt(res['fun']) >= points.shape[0] and iters < lim:
	
	# Extract and fit only those points outside the previously fit circle	
	points = np.array( [ (x,y) for x,y in circle if 
						 (x - z[0]) ** 2 + (y - z[1]) ** 2 >= r ** 2 ] )


	# Fit this new set of points
	res = opt.minimize ( lambda x: dist( x , points ) ,
						 np.concatenate( ( np.mean( points, axis = 0) ,
						 				 [width / 4] ) ) ) 

	# Extract the new fit parameters
	*z , r = res['x']

	# Up the loop count
	iters += 1

# Plot the resulting figure with the fitted lines overlaid
plt.figure(figsize = (5,5))
plt.imshow(im,cmap = 'gray', vmin = 0, vmax = 1)
plt.gca().axis('off')

# Fitted circle
theta = np.linspace(0,2 * np.pi,num = 100)
x = z[0] + r * np.cos(theta)
y = z[1] + r * np.sin(theta)
plt.plot(x,y,'r-')

# Baseline
x = np.array([0,im.shape[1]])
y = m * x + b
plt.plot(x,y,'r-')

plt.draw()

# Now we need to actually get the points of intersection and the angles from
# these fitted curves

# Brute force algebraic way
def rootFun(x,z,r,m,b):
	res = [0,0]
	res[0] = (x[0] - z[0]) ** 2 + (x[1] - z[1])**2 - r**2
	res[1] = x[1] - m*x[0] - b
	return res

# Define the baseline vector
v1 = [-1, -m]

# Get the point to the right of circle center
res = opt.root(lambda x: rootFun(x,z,r,m,b), [z[0] + r , z[1]])
x,y = res['x']

if y != z[1]:
	dydx = -(x - z[0]) / (y - z[1])
	v2 = [1 if dydx <= 0 else -1 , dydx if dydx >= 0 else -dydx]
else:
	v2 = [0 , 1]

phi = np.arccos(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2))
print(f'Contact angle right: {phi*360/2/np.pi : 6.3f}')

# Might be more elegant to do it this way, but also could be more time consuming
# Transform into new coordinate system so baseline is flat (i.e. vector from [1,m] -> [a,0])

plt.show()