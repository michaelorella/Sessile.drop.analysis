import numpy as np


### Strategy for automating contact angle measurements for Lauren+McLain

### Use some toolkit to extract every nth frame from avi file

### At every extracted frame, read it as a grayscale numpy array

### Using scikit-image canny edge detection, find the image edges

# Crop the edge image to a smaller size using the desired pixel values

# Obtain the X,Y coordinates of the True values in this edge image 
# (for processing)
coords = np.array([[i,j] for j,row in enumerate(edges) 
						 for i,x in enumerate(row) if x])

# Get the baseline from the left and right 20 pixels of the image 
# (this is important not to crop too far)

baseline = {'l':np.array([[x,y] for x,y in coords 
								if (x >= 400 and 
									x <= 420 and 
									y >= 200 and 
									y <= 600 )])
			'r':np.array([[x,y] for x,y in coords 
								if (x >= 400 and 
									x <= 420 and 
									y >= 200 and 
									y <= 600 )])}

# Fit the baseline to a line of form y = m*x + b using np.linalg
A = np.ones((baseline['l'].shape[0] + baseline['r'].shape[0],2))
A[:,1] = np.concatenate((baseline['l'][:,0],baseline['r'][:,0]))
c = np.concatenate((baseline['l'][:,1],baseline['r'][:,1]))
intercept,slope = np.linalg.lstsq(A,c)[0]

# Now find the points in the circle