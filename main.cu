#include "main.h"
#include <omp.h>
#include <chrono>

int main(int argc,char ** argv){
    cv::Mat input cv::imread(argv[1],CV_LOAD_IMAGEi_GRAYSCALE);

    cv::Mat output(input.rows,input.cols,CV_8UC1); 

    // We normalize the functions and test getting a histogram in cpu vs gpu

    // Get histograms on cpu and gpu time them and print
    auto start_cpu = chrono::high_resolution_clock::now();

    void cpuHistogram = cpuHistogramFromImage(input); 

    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_cpu = end_cpu - start_cpu;

    fprint("Omp Cpu Test %d s",duration.count());

    fprint("Gpu Test %d s",duration.count());

    // Compare histograms

    // get cdf
    // map to closest value of eq frequency
    // map values to new values or apply to image

    cv::imshow("Input",input);
    cv::imshow("Output",output);

    cv::waitKey();

    return 0;
}
int * cpuHistogramFromImage(cv::Mat input){
    // allocate histogram
    int * histogram = calloc(256,sizeof(int));
    // for each pixel 
    #pragma omp parallel for
    for(int i = 0; i<input.step; i++)
    {
        // save value in histograms proper bin
        #pragma omp atomic
        histogram[input.ptr()] ++;
    }
    return histogram;
}
int * gpuHistogramFromImage(cv::Mat input){

}
