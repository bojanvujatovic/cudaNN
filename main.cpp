#include <iostream>
#include <ctime> 
#include <cstdlib>
#include "MNIST.h"    
#include "elasticDeformation.h"

#include"NN.h"
#include<vector>
#include<cmath>
#include"myUtils.h"
#include <limits>

#include <unistd.h>
#include <stdio.h>
#include <fstream>
#include <iomanip>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <nvToolsExt.h>

#include <npp.h>
#include <nppdefs.h>

#include <vector>
#include <algorithm>
#include <set>

using namespace std;

void MNIST_deformation(int n_layersInput, int* n_unitsInput, int N_trainCurveInput, int N_deformInput, int validationPeriod, string directory);
void MNIST_deformation(string file, int N_trainCurveInput, int N_deformInput, int validationPeriod, string directory);

int main(void)
{ 
    srand((unsigned)time(NULL));

    int n_unitsA[] = {784, 2000, 10};
    int n_unitsB[] = {784, 1500, 500, 10};
    int n_unitsC[] = {784, 1500, 1000, 500, 10};

    MNIST_deformation(3, n_unitsA, 15000000, 5000, 5000, "/home/bojan/NN/Testing/748x2000x10");
    //MNIST_deformation(4, n_unitsB, 20000000, 5000, 5000, "/home/bojan/NN/Testing/748x1500x500x10");
    //MNIST_deformation(5, n_unitsC, 35000000, 5000, 5000, "/home/bojan/NN/Testing/748x1500x1000x500x10");


    return 0;
}

void MNIST_deformation(int n_layersInput, int* n_unitsInput, int N_trainCurveInput, int N_deformInput, int validationPeriod, string directory)
{
    // nvtx
    nvtxRangeId_t r1, r2;
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };

    int N_train = 0;
    int N_test = 0;
    int n_rows = 0;
    int n_cols = 0;

    ////////////////////////////////////////////////////////
    eventAttrib.color = colors[3]; eventAttrib.message.ascii = "Loading MNIST";
    r1 = nvtxRangeStartEx(&eventAttrib);
    float* trainData = readMNISTDataNormalised1DUnified(&N_train, &n_rows, &n_cols, "/home/bojan/NN/MNISTData/train-images.idx3-ubyte");
    char* trainLabels = readMNISTLabelsUnified(&N_train, "/home/bojan/NN/MNISTData/train-labels.idx1-ubyte");
    float* testData = readMNISTDataNormalised1DUnified(&N_test, &n_rows, &n_cols, "/home/bojan/NN/MNISTData/t10k-images.idx3-ubyte");
    char* testLabels = readMNISTLabelsUnified(&N_test, "/home/bojan/NN/MNISTData/t10k-labels.idx1-ubyte");
    nvtxRangeEnd(r1);
    ////////////////////////////////////////////////////////
    cudaDeviceSynchronize();

    ////////////////////////////////////////////////////////
    eventAttrib.color = colors[4]; eventAttrib.message.ascii = "NN constructor";
    r1 = nvtxRangeStartEx(&eventAttrib);
    NN nn1(n_layersInput, n_unitsInput, 0.01f);
    nvtxRangeEnd(r1);
    ////////////////////////////////////////////////////////

    float* deformedExamples;
    char* deformedExamplesLabels;
    cudaCheckErrors(cudaMallocManaged((void **)&deformedExamples, sizeof(float) *n_rows * n_cols * N_deformInput));
    cudaCheckErrors(cudaMallocManaged((void **)&deformedExamplesLabels, sizeof(char) * N_deformInput));
    cudaDeviceSynchronize();
    float* randDisplField;


    for (int i = 0; i < N_trainCurveInput; i++)
    {

        if (!(i % N_train))
        {
            if (i > 0) cudaFree(randDisplField);
            randDisplField = generateRandDisplFieldUnified_parallel(N_deformInput * n_rows, n_cols,
                                                                    10.0f, 10.0f, 35, 1.5,nn1.cublasHandle);
            cudaDeviceSynchronize();
        }

        ////////////////////////////////////////////////////////
        eventAttrib.color = colors[6]; eventAttrib.message.ascii = "Deformation";
        r1 = nvtxRangeStartEx(&eventAttrib);


        if(!(i % N_deformInput))
            elasticDeformation_parallel(trainData, trainLabels, N_train,
                                        deformedExamples,
                                        deformedExamplesLabels, randDisplField,
                                        n_rows, n_cols, N_deformInput, i);

        nvtxRangeEnd(r1);
        ///////////////////////////////////////////////////////
        cudaDeviceSynchronize();


        ////////////////////////////////////////////////////////
        eventAttrib.color = colors[i%2]; eventAttrib.message.ascii = "Train";
        r1 = nvtxRangeStartEx(&eventAttrib);

        nn1.Train(deformedExamples + (i%N_deformInput) * n_rows * n_cols,
                  deformedExamplesLabels + (i%N_deformInput),
                  1);
        cudaDeviceSynchronize();
        nvtxRangeEnd(r1);
        ////////////////////////////////////////////////////////


        if (!(i%validationPeriod))
        {
            ////////////////////////////////////////////////////////
            eventAttrib.color = colors[2]; eventAttrib.message.ascii = "Validate/save";
            r1 = nvtxRangeStartEx(&eventAttrib);
            float error = nn1.Validate(testData, testLabels, N_test);
            cout << left << setw(13) << ((float)i)/(float)N_trainCurveInput << " "
                 << left << setw(13) << error << endl;

            if (error <= 0.02f)
                nn1.Save(directory.c_str(), "CVerror" + intToString(error*1000.0f) + "promile.txt");
            cudaDeviceSynchronize();
            nvtxRangeEnd(r1);
            ////////////////////////////////////////////////////////
        }

    }

    printf("OK\n");

    cudaFree(randDisplField);
    cudaFree(deformedExamples);
    cudaFree(deformedExamplesLabels);
    cudaFree(trainData);
    cudaFree(trainLabels);

}

void MNIST_deformation(string file, int N_trainCurveInput, int N_deformInput, int validationPeriod, string directory)
{
    // nvtx
    nvtxRangeId_t r1, r2;
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };

    int N_train = 0;
    int N_test = 0;
    int n_rows = 0;
    int n_cols = 0;

    ////////////////////////////////////////////////////////
    eventAttrib.color = colors[3]; eventAttrib.message.ascii = "Loading MNIST";
    r1 = nvtxRangeStartEx(&eventAttrib);
    float* trainData = readMNISTDataNormalised1DUnified(&N_train, &n_rows, &n_cols, "/home/bojan/NN/MNISTData/train-images.idx3-ubyte");
    char* trainLabels = readMNISTLabelsUnified(&N_train, "/home/bojan/NN/MNISTData/train-labels.idx1-ubyte");
    float* testData = readMNISTDataNormalised1DUnified(&N_test, &n_rows, &n_cols, "/home/bojan/NN/MNISTData/t10k-images.idx3-ubyte");
    char* testLabels = readMNISTLabelsUnified(&N_test, "/home/bojan/NN/MNISTData/t10k-labels.idx1-ubyte");
    nvtxRangeEnd(r1);
    ////////////////////////////////////////////////////////
    cudaDeviceSynchronize();

    ////////////////////////////////////////////////////////
    eventAttrib.color = colors[4]; eventAttrib.message.ascii = "NN constructor";
    r1 = nvtxRangeStartEx(&eventAttrib);
    NN nn1(file.c_str());
    nvtxRangeEnd(r1);
    ////////////////////////////////////////////////////////

    float* deformedExamples;
    char* deformedExamplesLabels;
    cudaCheckErrors(cudaMallocManaged((void **)&deformedExamples, sizeof(float) *n_rows * n_cols * N_deformInput));
    cudaCheckErrors(cudaMallocManaged((void **)&deformedExamplesLabels, sizeof(char) * N_deformInput));
    cudaDeviceSynchronize();
    float* randDisplField;


    for (int i = 0; i < N_trainCurveInput; i++)
    {

        if (!(i % N_train))
        {
            if (i > 0) cudaFree(randDisplField);
            randDisplField = generateRandDisplFieldUnified_parallel(N_deformInput * n_rows, n_cols,
                                                                    10.0f, 30.0f, 35, 1.5,nn1.cublasHandle);
            cudaDeviceSynchronize();
        }

        ////////////////////////////////////////////////////////
        eventAttrib.color = colors[6]; eventAttrib.message.ascii = "Deformation";
        r1 = nvtxRangeStartEx(&eventAttrib);


        if(!(i % N_deformInput))
            elasticDeformation_parallel(trainData, trainLabels, N_train,
                                        deformedExamples,
                                        deformedExamplesLabels, randDisplField,
                                        n_rows, n_cols, N_deformInput, i);

        nvtxRangeEnd(r1);
        ///////////////////////////////////////////////////////
        cudaDeviceSynchronize();


        ////////////////////////////////////////////////////////
        eventAttrib.color = colors[i%2]; eventAttrib.message.ascii = "Train";
        r1 = nvtxRangeStartEx(&eventAttrib);

        nn1.Train(deformedExamples + (i%N_deformInput) * n_rows * n_cols,
                  deformedExamplesLabels + (i%N_deformInput),
                  1);
        cudaDeviceSynchronize();
        nvtxRangeEnd(r1);
        ////////////////////////////////////////////////////////


        if (!(i%validationPeriod))
        {
            ////////////////////////////////////////////////////////
            eventAttrib.color = colors[2]; eventAttrib.message.ascii = "Validate/save";
            r1 = nvtxRangeStartEx(&eventAttrib);
            float error = nn1.Validate(testData, testLabels, N_test/3);
            cout << left << setw(13) << ((float)i)/(float)N_trainCurveInput << " "
                 << left << setw(13) << error << endl;

            if (error <= 0.03f)
                nn1.Save(directory.c_str(), "CVerror" + intToString(error*1000.0f) + "promile.txt");
            cudaDeviceSynchronize();
            nvtxRangeEnd(r1);
            ////////////////////////////////////////////////////////
        }

    }

    printf("OK\n");

    cudaFree(randDisplField);
    cudaFree(deformedExamples);
    cudaFree(deformedExamplesLabels);
    cudaFree(trainData);
    cudaFree(trainLabels);

}
