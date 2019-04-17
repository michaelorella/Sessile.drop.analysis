### Strategy for automating contact angle measurements for Lauren+McLain

### Use some toolkit to extract every nth frame from avi file

### At every extracted frame, read it as a grayscale numpy array

### Using scikit-image canny edge detection, find the image edges

### Once the edges have been detected, fit line to pixels at the left and right side of the image

### Determine where these cross the baseline (which should be computable from the edges also)

### Identify the angle between these two lines