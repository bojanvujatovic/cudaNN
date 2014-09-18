#include <fstream>
#include <iostream> 
#include "MNIST.h"
#include "myUtils.h"

#include <cuda_runtime.h>

using namespace std;


float* readMNISTDataNormalised1DUnified(int* N, int* n_rows, int* n_cols, string filename)
{
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file.is_open()) cout << "ERROR READING FILE" << endl;
    
    int magic_number;        
    unsigned char temp;
    
    file.read((char*)&magic_number, 4);
    magic_number = reverseInt(magic_number);
    
    file.read((char*)N, 4);
    *N= reverseInt(*N);        
    
    file.read((char*)n_rows, 4);
    *n_rows= reverseInt(*n_rows);        
    
    file.read((char*)n_cols, 4);
    *n_cols= reverseInt(*n_cols);
    
    //double* trainData = new double[(*n_rows) * (*n_cols) * (*N)];
	float* trainData;
	cudaMallocManaged((void **)&trainData, sizeof(float)* (*n_rows) * (*n_cols) * (*N));
	cudaDeviceSynchronize();

    for(int i=0; i < *N ; i++)   
        for(int r = 0; r < *n_rows; r++)
            for(int c = 0; c < *n_cols; c++)                
            {                                       
                file.read((char *)&temp, 1);

                trainData[IDX_2D_TO_1D(i, IDX_2D_TO_1D(r, c, *n_cols), (*n_rows) * (*n_cols))] =
                        (float)temp/255.0f;
            }
    
    file.close();
    
    return trainData;
}

char* readMNISTLabelsUnified(int* N, string filename)
{
    std::ifstream file(filename.c_str(), std::ios::binary);
    
    int magic_number;        
    char temp;
    
    file.read((char*)&magic_number, 4);        
    magic_number = reverseInt(magic_number);
    
    file.read((char*)N, 4);
    *N = reverseInt(*N);
    
    //char* labels = new char[*N];
    char* labels;
    cudaMallocManaged((void **)&labels, sizeof(char)* (*N));
    cudaDeviceSynchronize();
	
    for(int i = 0; i < *N; i++)
    {                                       
        file.read(&temp, 1);

        labels[i] = (char)(temp); 
    }               
    
    file.close();
    
    return labels;
}
