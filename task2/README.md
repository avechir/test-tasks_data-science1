# test-task2
This project performs keypoints detection and image matching for satellite images.

Images used in this code can be found here https://drive.google.com/drive/folders/1SKZJd2Dky1oWoUjrFH645EXxqNVrIyI1?usp=sharing

Images are loaded from a raster file using the rasterio library, reshaped into images, converted to NumPy arrays of unsigned bytes, and resized. 
Then, the function for finding descriptors and keypoints using ORB is called, and after that, matches are found and plotted.
