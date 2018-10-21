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
int reduceSum(int * array,int length);
int * histogram_equalize(int * histogram,int total,int length);
void cpuMapImageWithHistogram(cv::Mat input,cv::Mat output,int * histogram);
void gpuMapImageWithHistogram(cv::Mat input,cv::Mat output,int * histogram);
__global__ 
void gpuMapImageWithHistogramKernel(unsigned char* input,unsigned char* output, int * histogram, int width, int height, int step);
