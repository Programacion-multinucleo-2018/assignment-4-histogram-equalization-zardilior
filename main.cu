#include "main.h"

int main(int argc,char ** argv){
    cv::Mat input = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat output(input.rows,input.cols,CV_8UC1); 
    cv::Mat output2(input.rows,input.cols,CV_8UC1); 

    // We normalize the functions and test getting a histogram in cpu vs gpu

    // Get histograms on cpu and gpu time them and print
    auto start_cpu = chrono::high_resolution_clock::now();

    int * cpuHistogram = cpuHistogramFromImage(input); 

    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration = end_cpu - start_cpu;

    printf("Omp Cpu Test: image %s %f ms \n",argv[1],duration.count());

    //printArray(cpuHistogram,256);

    int * gpuHistogram = gpuHistogramFromImage(input);

    // Compare histograms
    printf("Histograms are identical %d\n",
            compareArrays(cpuHistogram,gpuHistogram,256));

    // sum all up to now cumulative distribution step
    int total = reduceSum(gpuHistogram,256);

    // map to closest value of eq frequency
    int *  finalHistogram = histogram_equalize(gpuHistogram,total,256);
    // printArray(finalHistogram,256);

    // apply to image
    start_cpu = chrono::high_resolution_clock::now();

    cpuMapImageWithHistogram(input,output,finalHistogram); 

    end_cpu = chrono::high_resolution_clock::now();
    duration = end_cpu - start_cpu;

    printf("map GPU Test: image %s %f ms \n",argv[1],duration.count());

    gpuMapImageWithHistogram(input,output2,finalHistogram); 
    
    cv::namedWindow("Input",CV_WINDOW_NORMAL);
    cv::namedWindow("OutputCPU",CV_WINDOW_NORMAL);
    cv::namedWindow("OutputGPU",CV_WINDOW_NORMAL);

    cv::imshow("Input",input);
    cv::imshow("OutputCPU",output);
    cv::imshow("OutputGPU",output2);

    if(!argv[2])
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

    //printArray(histogram,256);

    cudaDeviceSynchronize();
    cudaFree(d_input);
    cudaFree(d_histogram);

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
int reduceSum(int * array,int length){
    int sum = 0;
    for(int i=0;i<length;i++){
        sum+=array[i];
    }
    return sum;
}
int * histogram_equalize(int * histogram,int total,int length){
    int step = total/length;
    int currentCumulative = 0;
    int * nhistogram = (int * )calloc(256,sizeof(int));
    
    for(int i=0;i<length;i++){
        currentCumulative += histogram[i];
        nhistogram[i] = currentCumulative/step;
    }
    return nhistogram;
}
void cpuMapImageWithHistogram(cv::Mat input,cv::Mat output,int * histogram){
    int length = input.rows*input.step;
    uchar * data = input.ptr<uchar>(0);
    uchar * data2 = output.ptr<uchar>(0);
    for(int i=0;i<length;i++){
        data2[i] = histogram[data[i]];
    }
}
void gpuMapImageWithHistogram(cv::Mat input,cv::Mat output,int * histogram){

    size_t bytes = input.step * input.rows;
    size_t histoSize = 256*sizeof(int);

    unsigned char * d_input,* d_output;
    int * d_histogram;

	cudaMalloc((void**)&d_input,bytes);
	cudaMalloc((void**)&d_histogram,histoSize);
	cudaMalloc((void**)&d_output,bytes);

	cudaMemcpy(d_input,input.ptr(),bytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_histogram,histogram,histoSize,cudaMemcpyHostToDevice);

	//Specify a reasonable block size
	const dim3 block(16,32);

	//Calculate grid size to cover the whole image
	const dim3 grid((input.cols)/block.x, (input.rows)/block.y);

    //run kernel and measure time

    auto start_cpu = chrono::high_resolution_clock::now();

	gpuMapImageWithHistogramKernel<<<grid,block>>>(
        d_input,d_output,d_histogram,input.cols,input.rows,input.step);

    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration = end_cpu - start_cpu;

    printf("Gpu Test: %f ms \n",duration.count());

	cudaMemcpy(output.ptr(),d_output,bytes,cudaMemcpyDeviceToHost);

    //printArray(histogram,256);

    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_histogram);
}
__global__ void gpuMapImageWithHistogramKernel(unsigned char* input,unsigned char* output, int * histogram, int width, int height, int step){
    __shared__ int * shHistogram;
    for(int i = 0;i<256;i++){
        shHistogram[i] = histogram[i];
    }
    __syncthreads();

    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if ((xIndex < width) && (yIndex < height)){
        const int tid = yIndex * step + xIndex; 
        output[tid] =static_cast<unsigned char>(shHistogram[input[tid]]);
    }
}
