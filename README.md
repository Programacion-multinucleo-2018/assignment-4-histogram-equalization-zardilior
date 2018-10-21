# Assignment 4: Histogram Equalization

Assignment No 4 for the multi-core programming course. Implement histogram equalization for a gray scale image in CPU and GPU. The result of applying the algorithm to an image with low contrast can be seen in Figure 1:

![Figure 1](Images/histogram_equalization.png)
<br/>Figure 1: Expected Result.

The programs have to do the following:

1. Using Opencv, load and image and convert it to grayscale.
2. Calculate de histogram of the image.
3. Calculate the normalized sum of the histogram.
4. Create an output image based on the normalized histogram.
5. Display both the input and output images.

Test your code with the different images that are included in the *Images* folder. Include the average calculation time for both the CPU and GPU versions, as well as the speedup obtained, in the Readme.

Rubric:

1. Image is loaded correctly.
2. The histogram is calculated correctly using atomic operations.
3. The normalized histogram is correctly calculated.
4. The output image is correctly calculated.
5. For the GPU version, used shared memory where necessary.
6. Both images are displayed at the end.
7. Calculation times and speedup obtained are incuded in the Readme.

# Results
map GPU Test: image Images/dog1.jpeg 65.436440 ms 
Gpu Test: 0.014026 ms 
Omp Cpu Test: image Images/dog2.jpeg 271.965057 ms 
Gpu Test: 0.025777 ms 
map GPU Test: image Images/dog2.jpeg 63.083439 ms 
Gpu Test: 0.014088 ms 
Omp Cpu Test: image Images/dog1.jpeg 208.265594 ms 
Gpu Test: 0.034422 ms 
map GPU Test: image Images/dog1.jpeg 66.736641 ms 
Gpu Test: 0.014567 ms 
Omp Cpu Test: image Images/dog2.jpeg 277.523132 ms 
Gpu Test: 0.023961 ms 
map GPU Test: image Images/dog2.jpeg 63.171249 ms 
Gpu Test: 0.012927 ms 
Omp Cpu Test: image Images/dog3.jpeg 338.286682 ms 
Gpu Test: 0.024755 ms 
map GPU Test: image Images/dog3.jpeg 63.014904 ms 
Gpu Test: 0.013069 ms 
Omp Cpu Test: image Images/histogram_equalization.png 4.371063 ms 
Gpu Test: 0.015678 ms 
map GPU Test: image Images/histogram_equalization.png 0.811725 ms 
Gpu Test: 0.006601 ms 
Omp Cpu Test: image Images/scenery.jpg 13.771117 ms 
Gpu Test: 0.019146 ms 
map GPU Test: image Images/scenery.jpg 1.878367 ms 
Gpu Test: 0.006560 ms 
Omp Cpu Test: image Images/woman2.jpg 207.360275 ms 
Gpu Test: 0.023955 ms 
map GPU Test: image Images/woman2.jpg 51.024738 ms 
Gpu Test: 0.013002 ms 
Omp Cpu Test: image Images/woman3.jpg 306.283783 ms 
Gpu Test: 0.025342 ms 
map GPU Test: image Images/woman3.jpg 51.088223 ms 
Gpu Test: 0.012768 ms 
Omp Cpu Test: image Images/woman.jpg 229.077637 ms 
Gpu Test: 0.024974 ms 
map GPU Test: image Images/woman.jpg 50.850903 ms 
Gpu Test: 0.012981 ms 
