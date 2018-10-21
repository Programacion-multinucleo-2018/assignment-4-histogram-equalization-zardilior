#include "main.h"

int main(int argc,char ** argv){
    cv::Mat input = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat output(input.rows,input.cols,CV_16UC1); 

    // We normalize the functions and test getting a histogram in cpu vs gpu

    // Get histograms on cpu and gpu time them and print
    auto start_cpu = chrono::high_resolution_clock::now();

    int * cpuHistogram = cpuHistogramFromImage(input); 

    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration = end_cpu - start_cpu;

    printf("Omp Cpu Test: image %s %f ms \n",argv[1],duration.count());
    printArray(cpuHistogram,256);

    int * gpuHistogram = gpuHistogramFromImage(input);

    // Compare histograms
    printf("Histograms are identical %d",
            compareArrays(cpuHistogram,gpuHistogram,256));

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
    int * histogram = (int * )calloc(256,sizeof(int));
    // for each pixel 
    uchar * data = input.ptr<uchar>(0);
    #pragma omp parallel for
    for(int i = 0; i<input.step*input.rows; i++)
    {
        // save value in histograms proper bin
        #pragma omp atomic
        histogram[data[i]]+=1;
    }
    return histogram;
}
int * gpuHistogramFromImage(cv::Mat input){
    size_t bytes = input.step * input.rows;
    size_t histoSize = 256*sizeof(int);
    unsigned char * d_input;
    int * d_histogram;
    
    int * histogram = (int * )malloc(histoSize);

	cudaMalloc((void**)&d_input,bytes);
	cudaMalloc((void**)&d_histogram,histoSize);

	cudaMemcpy(d_input,input.ptr(),bytes,cudaMemcpyHostToDevice);
	cudaMemset(d_histogram,0,histoSize);
	//Specify a reasonable block size
	const dim3 block(16,32);

	//Calculate grid size to cover the whole image
	const dim3 grid((input.cols)/block.x, (input.rows)/block.y);

    //run kernel and measure time

    auto start_cpu = chrono::high_resolution_clock::now();


	gpuHistogramFromImageKernel<<<grid,block>>>(d_input,d_histogram,input.cols,input.rows,input.step);

    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration = end_cpu - start_cpu;

    printf("Gpu Test: %f ms \n",duration.count());

	cudaMemcpy(histogram,d_histogram,histoSize,cudaMemcpyDeviceToHost);

    printArray(histogram,256);

    cudaDeviceSynchronize();
    cudaFree(d_input);

    return histogram;
}
__global__ void gpuHistogramFromImageKernel(unsigned char* input, int * histogram, int width, int height, int step){
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if ((xIndex < width) && (yIndex < height)){
        const int tid = yIndex * step + xIndex; 
        atomicAdd(&histogram[input[tid]], 1);
    }
}
void printArray(int * array,int length){
    for(int i=0;i<length;i++){
        printf("%d ",array[i]);
    }
    printf("\n");
}
int compareArrays(int * a1,int * a2, int length){
    for(int i=0;i<length;i++){
        if (a1[i]!=a2[i])
            return 0; 
    }
    return 1; 
}
