#ifndef UTILS_H
#define UTILS_H

#include <png.h>
#include <string>
#include <cmath>
#include <cstdlib>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IDX_2D_TO_1D(r, c, n_cols) ((r) * (n_cols)) + (c)
#define SIGMOID(x) 1.f / (1.f + expf(-(x)))
#define SIGMOID_GRADIENT(x) (SIGMOID(x))*(1 - (SIGMOID(x)))

#define float_PRINT_WIDTH 13

#define cudaCheckErrors(err)    __cudaCheckError(err, __FILE__, __LINE__ )
inline void __cudaCheckError(cudaError err, const char *file, const int line )
{
    if ( cudaSuccess != err )
    {
        printf("cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return;
}

using namespace std;

void outputFloatMatrixToFile(float* array, int n_rows, int n_cols, string filename);
void outputCharVectorToFile(char* array, int n_rows, string filename);
int writePNG(string filename, int width, int height, float* buffer, string title);

string intToString(int n);

void matrixMultiply_serial(float* A, int n_rows_A, int n_cols_A,
                           float* B, int n_rows_B, int n_cols_B,
                           float* C, int n_rows_C, int n_cols_C);
void matrixMultiply_parallel(float* A, int n_rows_A, int n_cols_A,
                           float* B, int n_rows_B, int n_cols_B,
                           float* C, int n_rows_C, int n_cols_C,
                           cublasHandle_t cublasHandle);
void matrixMultiplyAndTranspose_serial(float* A, int n_rows_A, int n_cols_A,
                              float* B, int n_rows_B, int n_cols_B,
                              float* C, int n_rows_C, int n_cols_C);
void matrixMultiplyAndTranspose_parallel(float* A, int n_rows_A, int n_cols_A,
                              float* B, int n_rows_B, int n_cols_B,
                              float* C, int n_rows_C, int n_cols_C,
                              cublasHandle_t cublasHandle);
void Transpose_serial(float* matrix, int n_rows, int n_cols);

float* generateRandDisplFieldUnified_parallel(int n_rows, int n_cols,
                                      float sigma, float alpha, int kernelWidth, float amplitude,
                                      cublasHandle_t cublasHandle);
                           
void applySigmoidToMatrixElements_serial(float* matrix, int n_rows, int n_cols);
double calculateClassificationError_serial(float* output, char* labels, int N, int d);
double calculateMSE_serial(float* output, char* labels, int N, int d);

#endif
