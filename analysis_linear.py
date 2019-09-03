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
import matplotlib.widgets as widgets

#Numerical analysis
import numpy as np

#Optimization
import scipy as scipy
import scipy.optimize as opt

function, image, *kwargs = sys.argv

#Set default numerical arguments
separationThreshold = 10
baselineThreshold = 20
linThreshold = 20
circleThreshold = 5
everyNSeconds = 1
σ = 5
startSeconds = 10

#Get the file type for the image file
parts = image.split('.')
ext = parts[-1]

video = False

if ext.lower() == 'avi' or ext.lower() == 'mp4':
	video = True
elif ext.lower() != 'jpg' and ext.lower() != 'png' and ext.lower() != 'gif':
	raise ValueError(f'Invalid file extension provided. I can\'t read {ext} files')

#Overwrite these defaults if desired
kwargs = zip ( * [ iter(kwargs) ] * 2 )

for argPair in kwargs:
	if argPair[0] == '-b' or argPair[0] == '--baselineThreshold':
		baselineThreshold = int(argPair[1])
	elif argPair[0] == '-c' or argPair[0] == '--circleThreshold':
		circleThreshold = int(argPair[1])
	elif (argPair[0] == '-t' or argPair[0] == '--times') and video:
		everyNSeconds = float(argPair[1])
	elif argPair[0] == '-s':
		σ = float(argPair[1])
	elif (argPair[0] == '-ss' or argPair[0] == '--startSeconds') and video:
		startSeconds = float(argPair[1])


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
				if ( ( i / np.round(fps) - startSeconds ) % everyNSeconds ) == 0 and
				   ( i / np.round(fps) > startSeconds ) ]
	images = [im / im.max() for im in images]

# Get 4-list of points for left, right, top, and bottom crop (in that order)

# Show the first image in the stack so that user can select crop box
print('Waiting for your input, please crop the image as desired and hit enter')
viewer = ImageViewer(images[0])
rect_tool = RectangleTool(viewer, on_enter = viewer.closeEvent)
viewer.show()
cropPoints = np.array(rect_tool.extents)
cropPoints = np.array(np.round(cropPoints),dtype = int)

time = []
angles = []
volumes = []

# Make sure that the edges are being detected well
edges = feature.canny(images[0],sigma = σ)
fig , ax = plt.subplots(2,1,gridspec_kw = {'height_ratios': [10,1]} , figsize = (8,8))
ax[0].imshow(edges, cmap = 'gray_r', vmin = 0, vmax = 1)
ax[0].set_xlim(cropPoints[:2])
ax[0].set_ylim(cropPoints[2:])
ax[0].axis('off')

sigmaSlide = widgets.Slider(ax[1],r'$\log_{10}\sigma$',-1,1,valinit = np.log10(σ),color = 'gray')

def update(val):
	edges = feature.canny(images[0], sigma = np.power(10,val))
	ax[0].imshow(edges,cmap = 'gray_r',vmin = 0,vmax = 1)
	fig.canvas.draw_idle()

sigmaSlide.on_changed(update)
print('Waiting for your input, please select a desired filter value, and close image when done')
plt.show()
σ = np.power(10,sigmaSlide.val)
print(f'Proceeding with sigma = {σ : 6.2f}')

# Create a set of axes to hold the scatter points for all frames in the videos
plt.figure()
scatAx = plt.axes()

for j,im in enumerate(images):
	### Using scikit-image canny edge detection, find the image edges
	edges = feature.canny(im,sigma = σ)

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

	print(f'Baseline: y = {m}*x + {b}')

	# Now find the points in the circle
	circle = np.array([(x,y) for x,y in crop if
											y - (m*x + b)  <= -circleThreshold])

	scatAx.scatter(circle[:,0],circle[:,1])

	# Look for the greatest distance between points on the baseline
	baselinePoints = np.sort([x for x,y in crop if 
											y - (m*x + b) > -circleThreshold],kind = 'mergesort')
	Δx = np.diff(baselinePoints)
	indices = [i for i,out in enumerate(Δx > separationThreshold) if out]
	print(indices)
	indices = [indices[0] , indices[-1]+1]
	limits = baselinePoints[indices]

	# Get linear points
	linearPoints = {'l':np.array( [ (x,y) for x,y in crop if 
							   (y - (m*x + b) <= -circleThreshold and y - (m*x + b) >= -(circleThreshold + linThreshold)) and
							   ( x <= limits[0] + linThreshold/2 ) and ( x >= limits[0] - linThreshold/2 ) ] ),
					'r':np.array( [ (x,y) for x,y in crop if 
							   (y - (m*x + b) <= -circleThreshold and y - (m*x + b) >= -(circleThreshold + linThreshold)) and
							   ( x <= limits[1] + linThreshold/2 ) and ( x >= limits[1] - linThreshold/2 ) ] ) }

	L = np.ones( ( linearPoints['l'].shape[0] , 2 ) )
	L[:,1] = linearPoints['l'][:,0]
	l = linearPoints['l'][:,1]

	R = np.ones( ( linearPoints['r'].shape[0] , 2 ) )
	R[:,1] = linearPoints['r'][:,0]
	r = linearPoints['r'][:,1]

	params = {'l':np.linalg.lstsq(L,l,rcond=None)[0],
			  'r':np.linalg.lstsq(R,r,rcond=None)[0]}

	lb, lm = params['l']
	rb, rm = params['r']


	# Define baseline vector
	bv = [1, m]/np.linalg.norm([1,m])

	# Define right side vector
	rv = [1, rm]/np.linalg.norm([1,rm])

	# Define left side vector
	lv = [1, lm]/np.linalg.norm([1,lm])

	# Calculate the angle between these two vectors defining the base-line and tangent-line
	ϕ = {'l':180-np.arccos(np.dot(bv,lv))*360/2/np.pi , 'r':180-np.arccos(np.dot(bv,rv))*360/2/np.pi}
	
	# TODO:// Add the actual volume calculation here!


	print(f'At time { j * everyNSeconds }: \t\t Contact angle left (deg): {ϕ["l"] : 6.3f} \t\t Contact angle right (deg): {ϕ["r"] : 6.3f}')
	angles += [ ϕ ]
	time += [ j * everyNSeconds ]

# Plot the last resulting figure with the fitted lines overlaid
plt.figure(figsize = (5,5))
plt.imshow(im,cmap = 'gray', vmin = 0, vmax = 1)
plt.gca().axis('off')

# Baseline
x = np.array([0,im.shape[1]])
y = m * x + b
plt.plot(x,y,'r-')

# Left side line
yl = lm * x + lb
plt.plot(x,yl,'r-')

# Right side line
yr = rm * x + rb
plt.plot(x,yr,'r-')

plt.xlim(cropPoints[0:2])
plt.ylim(cropPoints[-1:-3:-1])

if video:
	fig, ax1 = plt.subplots(figsize = (5,5))
	color = 'black'
	ax1.set_xlabel('Time [s]')
	ax1.set_ylabel('Contact Angle [deg]', fontsize = 10,color = color)
	ax1.plot(time,angles, marker = '.',markerfacecolor = color,markeredgecolor = color,markersize = 10
			 , linestyle = None)
	ax1.tick_params(axis = 'y', labelcolor = color)

	plt.tight_layout()
	plt.draw()

if '\\' in image:
	parts = image.split('\\')
else:
	parts = image.split('/')
path = '/'.join(parts[:-1]) #Leave off the actual file part
filename = path + f'/results_{parts[-1]}.csv'

print(f'Saving the data to {filename}')
with open(filename,'w+') as file:
	file.write(",".join([str(t) for t in time]))
	file.write('\n')
	file.write(",".join([str(s['l']) for s in angles]))
	file.write('\n')
	file.write(",".join([str(s['r']) for s in angles]))

plt.show()