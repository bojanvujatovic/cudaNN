#include <sstream>
#include <fstream>
#include "myUtils.h" 
#include "elasticDeformation.h"

#include <iostream>
#include <iomanip>
#include <cfloat>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <npp.h>
#include <nppdefs.h>

using namespace std; 

int writePNG(string filename, int width, int height, float* buffer, string title)
{
    int code = 0;
    FILE *fp;
    png_structp png_ptr;
    png_infop info_ptr;
    png_bytep row;

    // Open file for writing (binary mode)
    fp = fopen(filename.c_str(), "wb");
    if (fp == NULL)
    {
        cout << "Could not open file " << filename << " for writing" << endl;
        code = 1;
        goto finalise;
    }

    // Initialize write structure
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL)
    {
        cout << "Could not allocate write struct" << endl;
        code = 1;
        goto finalise;
    }

    // Initialize info structure
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        cout << "CCould not allocate info struct" << endl;
        code = 1;
        goto finalise;
    }

    // Setup Exception handling
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        cout << "Error during png creation" << endl;
        code = 1;
        goto finalise;
    }

    png_init_io(png_ptr, fp);

    // Write header (8 bit colour depth)
    png_set_IHDR(png_ptr, info_ptr, width, height,
            8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    // Set title
    if (!title.empty()) {
        png_text title_text;
        title_text.compression = PNG_TEXT_COMPRESSION_NONE;
        title_text.key = NULL;
        title_text.text = strdup(title.c_str());;
        png_set_text(png_ptr, info_ptr, &title_text, 1);
    }

    png_write_info(png_ptr, info_ptr);

    // Allocate memory for one row
    row = new png_byte[width];

    // Write image data
    int x, y;
    for (y=0 ; y<height ; y++)
    {
        for (x=0 ; x<width ; x++)
        {
            row[x] = (unsigned char)(buffer[IDX_2D_TO_1D(y, x, width)] * 255.0f);
        }
        png_write_row(png_ptr, row);
    }

    // End write
    png_write_end(png_ptr, NULL);

    finalise:
    if (fp != NULL) fclose(fp);
    if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    if (row != NULL) delete[] row;

    return code;
}

int reverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
string intToString(int n)
{
    ostringstream temp;
    temp << n;
    return temp.str();
}

void outputFloatMatrixToFile(float* array, int n_rows, int n_cols, string filename)
{
    ofstream file(filename.c_str());
    
    if (!file.is_open()) cout << "ERROR READING FILE" << endl;

    for (int r = 0; r < n_rows; r++) 
    {
        for (int c = 0; c < n_cols; c++) 
        {
        	if (array[IDX_2D_TO_1D(r, c, n_cols)] >= 0.0f)
        	{
        		file << " " ;
        		file << left << setw(float_PRINT_WIDTH - 1) << array[IDX_2D_TO_1D(r, c, n_cols)];
        	}
        	else
        		file << left << setw(float_PRINT_WIDTH) << array[IDX_2D_TO_1D(r, c, n_cols)];
        }
        file << endl;
    }
    
    file.close();
}

void outputCharVectorToFile(char* array, int n_rows, string filename)
{
    ofstream file(filename.c_str());

    if (!file.is_open()) cout << "ERROR READING FILE" << endl;

    for (int r = 0; r < n_rows; r++)
    {
        file << (int)array[r] << endl;
    }

    file.close();
}

void matrixMultiply_serial(float* A, int n_rows_A, int n_cols_A,
                           float* B, int n_rows_B, int n_cols_B,
                           float* C, int n_rows_C, int n_cols_C)
{
    for (int i = 0; i < n_rows_A; i++)
        for (int j = 0; j < n_cols_B; j++)
        {
            float sum = 0;
            for (int k = 0; k < n_cols_A ; k++)
                sum += A[IDX_2D_TO_1D(i, k, n_cols_A)] * B[IDX_2D_TO_1D(k, j, n_cols_B)];
     
            C[IDX_2D_TO_1D(i, j, n_cols_C)] = sum;
        }
}

void matrixMultiply_parallel(float* A, int n_rows_A, int n_cols_A,
                           float* B, int n_rows_B, int n_cols_B,
                           float* C, int n_rows_C, int n_cols_C,
                           cublasHandle_t cublasHandle)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;


    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n_cols_B, n_rows_A, n_cols_A, &alpha, B, n_cols_B, A, n_cols_A, &beta, C, n_cols_C);
    cudaDeviceSynchronize();
}

void matrixMultiplyAndTranspose_serial(float* A, int n_rows_A, int n_cols_A,
                              float* B, int n_rows_B, int n_cols_B,
                              float* C, int n_rows_C, int n_cols_C)
{
    for (int i = 0; i < n_rows_A; i++)
    for (int j = 0; j < n_cols_B; j++)
        {
            float sum = 0;
            for (int k = 0; k < n_cols_A ; k++)
                sum += A[IDX_2D_TO_1D(i, k, n_cols_A)] * B[IDX_2D_TO_1D(k, j, n_cols_B)];

            C[IDX_2D_TO_1D(j, i, n_cols_C)] = sum;
        }
}

void matrixMultiplyAndTranspose_parallel(float* A, int n_rows_A, int n_cols_A,
                              float* B, int n_rows_B, int n_cols_B,
                              float* C, int n_rows_C, int n_cols_C,
                              cublasHandle_t cublasHandle)
{
    const float alpha = 1.0;
    const float beta  = 0.0;

    cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, n_cols_C, n_rows_C, n_rows_B, &alpha, A, n_cols_A, B, n_cols_B, &beta, C, n_cols_C);
    cudaDeviceSynchronize();
}

float* generateRandDisplFieldUnified_parallel(int n_rows, int n_cols,
                                      float sigma, float alpha, int kernelWidth, float amplitude,
                                      cublasHandle_t cublasHandle)
{
    float* randDisplFieldInput;
    float* randDisplFieldOutput;

    const float constAlpha = alpha;

    cudaCheckErrors(cudaMallocManaged((void **)&randDisplFieldInput, sizeof(float) * (n_rows + kernelWidth - 1) * (n_cols + kernelWidth - 1)));
    cudaCheckErrors(cudaMallocManaged((void **)&randDisplFieldOutput, sizeof(float) *n_rows * n_cols));
    cudaDeviceSynchronize();

    float* kernel;
    cudaCheckErrors(cudaMallocManaged((void **)&kernel, sizeof(float)* kernelWidth * kernelWidth));
    cudaDeviceSynchronize();
    float sum = 0.0f;

    for (int r = -kernelWidth/2; r <= kernelWidth/2; r++)
        for (int c = -kernelWidth/2; c <= kernelWidth/2; c++)
        {
            double value = expf(-static_cast<double>(c * c + r * r) / (2.f * sigma * sigma));
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

    for (int i = 0; i < (n_rows + kernelWidth - 1) * (n_cols + kernelWidth - 1); i++) {// inicijaliziraj jednom randDeisplacementField - ionako
        randDisplFieldInput[i] = ((double)rand() / RAND_MAX) * 2.0 *amplitude  - amplitude; //se znamenke stalno mijenjaju, svaki put je prijmenjeno na drugu
    }

    NppiSize outputROI = {n_cols, n_rows};
    NppiSize kernelSize = {kernelWidth, kernelWidth};
    NppiPoint anchor = {kernelWidth/2, kernelWidth/2};

    /*
    int r =  nppiFilter_32f_C1R((const Npp32f *)randDisplFieldInput, sizeof(float) * (n_cols + kernelWidth - 1),
                       randDisplFieldOutput, n_cols * sizeof(float),
                       outputROI,
                       (const Npp32f *)kernel, kernelSize, anchor); */
    fieldConvolutionAndScaling(randDisplFieldInput, randDisplFieldOutput,
                                kernel,
                                n_rows, n_cols, kernelWidth,
                                1.0);
    cudaDeviceSynchronize();

    cublasSscal(cublasHandle, n_rows * n_cols, &constAlpha, randDisplFieldOutput, 1);
    cudaDeviceSynchronize();

    cudaFree(randDisplFieldInput);

    return randDisplFieldOutput;
}

void Transpose_serial(float* matrix, int n_rows, int n_cols)
{
	float* temp = new float[n_rows * n_cols];

	for (int i = 0; i < n_rows * n_cols; i++) temp[i] = matrix[i];

	for (int r = 0; r < n_rows; r++)
		for (int c = 0; c < n_cols; c++)
			matrix[IDX_2D_TO_1D(c, r, n_rows)] = temp[IDX_2D_TO_1D(r, c, n_cols)];

	delete[] temp;
}

                           
void applySigmoidToMatrixElements_serial(float* matrix, int n_rows, int n_cols)
{
    for(int i = 0; i < n_rows * n_cols; i++)
        matrix[i] = SIGMOID(matrix[i]);
}

double calculateClassificationError_serial(float* output, char* labels, int N, int d)
{
    int missmatch = 0;
    int maxLabel;
    
    for (int r = 0; r < N; r++)
    {
        float maxValue = -FLT_MAX;
        for (int c = 0; c < d; c++)
        {
            if (output[IDX_2D_TO_1D(r, c, d)] > maxValue)
            {
                
                maxValue = output[IDX_2D_TO_1D(r, c, d)];
                maxLabel = c;
            }
        }
        if (maxLabel != (int)labels[r]) missmatch++;
    }
    
    
    return (float)missmatch/N;
}

double calculateMSE_serial(float* output, char* labels, int N, int d)
{
    float sum = 0;

    for (int r = 0; r < N; r++)
    {
        for (int c = 0; c < d; c++)
        {
        	float a = SIGMOID(output[IDX_2D_TO_1D(r, c, d)]);
        	float b = (float)(c == (int)labels[r]);


            sum += (a - b) * (a - b);
        }
    }


    return (float)sum/(2.0f * N);
}


