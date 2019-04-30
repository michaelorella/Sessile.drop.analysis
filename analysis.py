#Import system for comman line processing
import sys

#Image processing import
import skimage
from skimage import feature
from skimage import io
from skimage.viewer import ImageViewer
from skimage.viewer.canvastools import RectangleTool

#Handle video inputs
#Note, need to make sure that imageio-ffmpeg has been pip installed
import imageio

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
everyNSeconds = 1
sigma = 5

#Get the file type for the image file
parts = image.split('.')
ext = parts[-1]

video = False

if ext == 'avi' or ext == 'mp4':
	video = True
elif ext != 'jpg' and ext != 'png' and ext != 'gif':
	raise ValueError(f'Invalid file extension provided. I can\'t read {ext} files')

#Overwrite these defaults if desired
kwargs = zip ( * [ iter(kwargs) ] * 2 )

for argPair in kwargs:
	if argPair[0] == '-l' or argPair[0] == '--lim':
		lim = int(argPair[1])
	elif argPair[0] == '-b' or argPair[0] == '--baselineThreshold':
		baselineThreshold = int(argPair[1])
	elif argPair[0] == '-c' or argPair[0] == '--circleThreshold':
		circleThreshold = int(argPair[1])
	elif argPair[0] == '-t' or argPair[0] == '--times':
		everyNSeconds = float(argPair[1])
	elif argPair[0] == '-s':
		sigma = argPair[1]


### Strategy for automating contact angle measurements for Lauren+McLain

### Use some toolkit to extract every nth frame from avi file

### At every extracted frame, read it as a grayscale numpy array
# TODO!! - get frames from video

if not video:
	images = [io.imread(image,as_gray = True)]
else:
	images = imageio.get_reader(image)
	fps = images.get_meta_data()['fps']

	#Set the conversion from RGB to grayscale using the scikit-image method (RGB to grayscale page)
	conversion = np.array([0.2125,0.7154,0.0721])

	#Perform the conversion and extract only so many frames to analyze
	images = [ np.dot(im,conversion) for i,im in enumerate(images) 
				if i % (everyNSeconds * np.round(fps) ) == 0 ]
	images = [im / im.max() for im in images]
	images = images[5:]

# Get 4-list of points for left, right, top, and bottom crop (in that order)

# Show the first image in the stack so that user can select crop box
viewer = ImageViewer(images[0])
rect_tool = RectangleTool(viewer, on_enter = viewer.closeEvent)
viewer.show()
cropPoints = np.array(rect_tool.extents)
cropPoints = np.array(np.round(cropPoints),dtype = int)

time = []
angles = []

# Define the loss function that we use for fitting
def dist(param, points):
	*z , r = param
	ar = [(np.linalg.norm(np.array(z) - np.array(point)) - r ) ** 2 
			for point in points]
	return np.sum(ar)

# Define function that calculates residual for intersection between line (m,b) and circle (z,r)
def rootFun(x,z,r,m,b):
	res = [0,0]
	res[0] = (x[0] - z[0]) ** 2 + (x[1] - z[1])**2 - r**2
	res[1] = x[1] - m*x[0] - b
	return res

for j,im in enumerate(images):
	### Using scikit-image canny edge detection, find the image edges
	edges = feature.canny(im,sigma = sigma)

	# Obtain the X,Y coordinates of the True values in this edge image 
	# (for processing)
	coords = np.array([[i,j] for j,row in enumerate(edges) 
							 for i,x in enumerate(row) if x])

	

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

	plt.scatter(circle[:,0],circle[:,1])


	# Get the cropped image width
	width = cropPoints[1] - cropPoints[0]

	# Try to fit a circle to the points that we have extracted, only varying the radius about the
	# center of all the points
	z = np.mean(circle, axis = 0)
	res = opt.minimize ( lambda x: dist( [*z , x] , circle ) , 
					     width/2 )

	#Get the results	
	r = res['x']
	theta = np.linspace(0,2 * np.pi,num = 500)
	x = z[0] + r * np.cos(theta)
	y = z[1] + r * np.sin(theta)

	#print(f"Residual of {res['fun']} before starting")

	iters = 0

	# Keep retrying the fitting while the function value is large, as this 
	# indicates that we probably have 2 circles (e.g. there's something light
	# in the middle of the image)
	while res['fun'] >= circle.shape[0] and iters < lim:
		
		# Extract and fit only those points outside the previously fit circle	
		points = np.array( [ (x,y) for x,y in circle if 
							 (x - z[0]) ** 2 + (y - z[1]) ** 2 >= r ** 2 ] )


		# Fit this new set of points, using the full set of parameters
		res = opt.minimize ( lambda x: dist( x , points ) ,
							 np.concatenate( ( np.mean( points, axis = 0) ,
							 				 [width / 4] ) ) ) 

		# Extract the new fit parameters
		*z , r = res['x']

		# Up the loop count
		#print(f"Residual of {res['fun']} at iteration {iters}")
		
		iters += 1

	# Now we need to actually get the points of intersection and the angles from
	# these fitted curves
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
	angles += [ phi * 360 / 2 / np.pi ]
	time += [ j * everyNSeconds ]

	# Might be more elegant to do it this way, but also could be more time consuming
	# Transform into new coordinate system so baseline is flat (i.e. vector from [1,m] -> [a,0])

# Plot the last resulting figure with the fitted lines overlaid
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

plt.figure(figsize = (5,5))
plt.plot(time,angles)
plt.show()