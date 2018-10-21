#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>
#include <chrono>
#include <stdio.h>
using namespace std;
int * gpuHistogramFromImage(cv::Mat input);
int * cpuHistogramFromImage(cv::Mat input);
void printArray(int * array,int length);
__global__ void gpuHistogramFromImageKernel(unsigned char* input, int * histogram, int width, int height, int step);
int compareArrays(int * a1,int * a2, int length);
