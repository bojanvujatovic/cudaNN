#include <cmath>
#include <cstdlib> 
#include <iostream>
#include "elasticDeformation.h"
#include "myUtils.h"
#include "MNIST.h"

#include <curand_kernel.h>
#include <ctime>

#include <npp.h>
#include <nppdefs.h>


#define INPUT_VALUE(r, c, n_rows, n_cols, input) ((int)r < 0 || (int)r >= (int)n_rows || (int)c < 0 || (int)c >= (int)n_cols) ? 0.0f : input[(int)(r)*(int)(n_cols) + (int)(c)]

using namespace std;

// kernelwidth must be odd
void elasticDeformation_serial(float* input, float* output,
                               int n_rows, int n_cols,
                               float sigma, float alpha, int kernelWidth, float amplitude)
{
    // Initialise random displacement field
    float* randDisplFieldX = new float[n_rows * n_cols];
    float* randDisplFieldY = new float[n_rows * n_cols];
    for (int i = 0; i < n_rows * n_cols; i++)
    {
        randDisplFieldX[i] = ((float)rand() / RAND_MAX) * 2.0f *amplitude - amplitude;
        //randDisplFieldY[i] = ((double)rand() / RAND_MAX) * 2.0f  - 1.0f;
        randDisplFieldY[i] = randDisplFieldX[i];
    }

    // Initialise Gaussian kernel
    float* kernel = new float[kernelWidth * kernelWidth];
    float sum = 0.0f;


    for (int r = -kernelWidth/2; r <= kernelWidth/2; r++)
        for (int c = -kernelWidth/2; c <= kernelWidth/2; c++)
        {
            float value = expf(-static_cast<float>(c * c + r * r) / (2.f * sigma * sigma));
            int idx = IDX_2D_TO_1D(r + kernelWidth/2, c + kernelWidth/2, kernelWidth);

            kernel[idx] = value;
            sum += value;
        }

    float normalisationFactor = 1.f / sum;
    for (int r = -kernelWidth/2; r <= kernelWidth/2; r++)
        for (int c = -kernelWidth/2; c <= kernelWidth/2; c++)
        {
            int idx = IDX_2D_TO_1D(r + kernelWidth/2, c + kernelWidth/2, kernelWidth);
            kernel[idx] *= normalisationFactor;
        }

    // Convolve radnom displacemet field and kernel
    float* newRandDisplFieldX = new float[n_rows * n_cols];
    float* newRandDisplFieldY = new float[n_rows * n_cols];
    fieldConvolutionAndScaling(randDisplFieldX, newRandDisplFieldX, kernel, n_rows, n_cols, kernelWidth, alpha);
    fieldConvolutionAndScaling(randDisplFieldY, newRandDisplFieldY, kernel, n_rows, n_cols, kernelWidth, alpha);

    /*
    output2DToFile(kernel, kernelWidth, kernelWidth, "kernel.txt");
    output2DToFile(randDisplFieldX, sizeX, sizeY, "randDisplFieldX.txt");
    output2DToFile(randDisplFieldY, sizeX, sizeY, "randDisplFieldY.txt");
    output2DToFile(newRandDisplFieldX, sizeX, sizeY, "newRandDisplFieldX.txt");
    output2DToFile(newRandDisplFieldY, sizeX, sizeY, "newRandDisplFieldY.txt");
    */

    // Apply displacement field
    for (int r = 0; r < n_rows; r++)
        for (int c = 0; c < n_cols; c++)
        {
            int idx = IDX_2D_TO_1D(r, c, n_cols);

            float inputValueTopLeft = INPUT_VALUE(floor(r + newRandDisplFieldX[idx])    , floor(c + newRandDisplFieldY[idx]) + 1, n_rows, n_cols, input);
            float inputValueTopRght = INPUT_VALUE(floor(r + newRandDisplFieldX[idx]) + 1, floor(c + newRandDisplFieldY[idx]) + 1, n_rows, n_cols, input);
            float inputValueBtmLeft = INPUT_VALUE(floor(r + newRandDisplFieldX[idx])    , floor(c + newRandDisplFieldY[idx])    , n_rows, n_cols, input);
            float inputValueBtmRght = INPUT_VALUE(floor(r + newRandDisplFieldX[idx]) + 1, floor(c + newRandDisplFieldY[idx])    , n_rows, n_cols, input);

            float horizontalCoeff = newRandDisplFieldX[idx]- floor(newRandDisplFieldX[idx]);
            float verticalCoeff   = newRandDisplFieldY[idx]- floor(newRandDisplFieldY[idx]);

            float inputValueTop = inputValueTopLeft + horizontalCoeff * (inputValueTopRght - inputValueTopLeft);
            float inputValueBtm = inputValueBtmLeft + horizontalCoeff * (inputValueBtmRght - inputValueBtmLeft);

            output[idx] = inputValueBtm + verticalCoeff * (inputValueTop - inputValueBtm);
        }
    
    
    delete[] kernel;
    delete[] newRandDisplFieldX;
    delete[] newRandDisplFieldY;
    delete[] randDisplFieldX;
    delete[] randDisplFieldY;
}

void fieldConvolutionAndScaling(float* input, float* output,
                                float* kernel,
                                int n_rows, int n_cols, int kernelWidth,
                                float alpha)
{
    for (int r = 0; r < n_rows; r++)
        for (int c = 0; c < n_cols; c++)
        {
            float newValue = 0.f;
          
            for (int kernel_r = -kernelWidth/2; kernel_r <= kernelWidth/2; kernel_r++) 
                for (int kernel_c = -kernelWidth/2; kernel_c <= kernelWidth/2; kernel_c++) 
                {
                    int kernelIdx = IDX_2D_TO_1D(kernel_r + kernelWidth/2, kernel_c + kernelWidth/2, kernelWidth);
                    float inputValue = INPUT_VALUE(r + kernel_r, c + kernel_c, n_rows, n_cols, input); // if outside - 0

                    newValue += inputValue * kernel[kernelIdx];
                }

          output[IDX_2D_TO_1D(r, c, n_cols)] = newValue * alpha; // scaling as well
        }
}

__global__ void initialiseDeformedExamples(float* trainData, char* labels, int N_train,
                                           float* deformedExamples, char* deformedExamplesLabels,
                                           int n_rows, int n_cols, int *randTrainIDs)
{
    int deformedExampleID = blockIdx.x;
    int pixelID = threadIdx.y * n_cols + threadIdx.x;

    int trainExampleID = randTrainIDs[deformedExampleID];

    if(pixelID == 0)
    {
        deformedExamplesLabels[deformedExampleID] = labels[trainExampleID];
    }
    deformedExamples[deformedExampleID * n_rows * n_cols + pixelID] =
            trainData[trainExampleID * n_rows * n_cols + pixelID];
}

__global__ void applyDisplField(float* deformedExamples, float* randDisplField,
                                           int n_rows, int n_cols, int N_deform)
{
    int r = threadIdx.x + n_rows * blockIdx.x;
    int c = threadIdx.y;

    int idx = IDX_2D_TO_1D(r, c, n_cols);

    float inputValueTopLeft = INPUT_VALUE(floor(r + randDisplField[idx])    , floor(c + randDisplField[idx]) + 1, n_rows * N_deform, n_cols, deformedExamples);
    float inputValueTopRght = INPUT_VALUE(floor(r + randDisplField[idx]) + 1, floor(c + randDisplField[idx]) + 1, n_rows * N_deform, n_cols, deformedExamples);
    float inputValueBtmLeft = INPUT_VALUE(floor(r + randDisplField[idx])    , floor(c + randDisplField[idx])    , n_rows * N_deform, n_cols, deformedExamples);
    float inputValueBtmRght = INPUT_VALUE(floor(r + randDisplField[idx]) + 1, floor(c + randDisplField[idx])    , n_rows * N_deform, n_cols, deformedExamples);

    float horizontalCoeff = randDisplField[idx]- floor(randDisplField[idx]);
    float verticalCoeff   = randDisplField[idx]- floor(randDisplField[idx]);

    float inputValueTop = inputValueTopLeft + horizontalCoeff * (inputValueTopRght - inputValueTopLeft);
    float inputValueBtm = inputValueBtmLeft + horizontalCoeff * (inputValueBtmRght - inputValueBtmLeft);

    float value = inputValueBtm + verticalCoeff * (inputValueTop - inputValueBtm);
    __syncthreads();
    deformedExamples[idx] = value;

}

void elasticDeformation_parallel(float* trainData, char* labels, int N_train,
                                 float* deformedExamples,
                                 char* deformedExamplesLabels, float* randDisplField,
                                 int n_rows, int n_cols, int N_deform, int i_main)
{

    int *randTrainIDs;
    cudaMallocManaged((void **)&randTrainIDs, sizeof(int)* N_deform);

    for(int i = 0; i < N_deform; i++)
    {
        randTrainIDs[i] = i_main % N_train + i;
    }

    initialiseDeformedExamples<<<N_deform, dim3(n_rows, n_cols)>>>
                                    (trainData, labels, N_train,
                                     deformedExamples, deformedExamplesLabels,
                                     n_rows, n_cols, randTrainIDs);
    cudaDeviceSynchronize();

    applyDisplField<<<N_deform, dim3(n_rows, n_cols)>>>
                                    (deformedExamples, randDisplField,
                                     n_rows, n_cols, N_deform);
    cudaDeviceSynchronize();

    cudaFree(randTrainIDs);
}
