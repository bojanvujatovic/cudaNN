#include <cublas.h>
#include <iostream>
#include "myUtils.h"
#include "NN.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <iomanip>
#include <unistd.h>
#include <fstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <nvToolsExt.h>

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };

NN::NN(int n_layers, int* n_units, float learningRate) : n_layers(n_layers),
                                                         n_units(n_units),
                                                         learningRate(learningRate)
{
    n_weights = 0;
    for (int i = 0; i < n_layers - 1; i++)
        n_weights += (1 + n_units[i]) * n_units[i + 1];// +1 for bias

    //weights = new double[n_weights];
    cudaMallocManaged((void **)&weights, sizeof(double)* n_weights);
    for(int i = 0; i < n_weights; i++)
        weights[i] = RAND_INITIALISE_WEIGHT; // TODO parallel

    cudaMallocManaged((void **)&weightGradients, sizeof(double)* n_weights);

    cublasStatus_t cublasStatus = cublasCreate(&cublasHandle);

    // nvtx
    eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;

    N_trained = 0;
}

NN::NN(string filename)
{
    ifstream file(filename.c_str());
    if (!file.is_open()) cout << "ERROR READING FILE" << endl;

    file >> n_layers;

    n_units = new int[n_layers];

    for (int i = 0; i < n_layers; i++)
        file >> n_units[i];

    file >> n_weights
         >> learningRate
         >> N_trained;

    cudaMallocManaged((void **)&weights, sizeof(float)* n_weights);

    for (int i = 0; i < n_layers - 1; i++)
    {
        float* array = getWeightsForLayer(i);
        int n_rows = n_units[i] + 1;
        int n_cols = n_units[i + 1];

        for (int r = 0; r < n_rows; r++)
            for (int c = 0; c < n_cols; c++)
                file >> array[IDX_2D_TO_1D(r, c, n_cols)];
    }

    cudaMallocManaged((void **)&weightGradients, sizeof(float)* n_weights);
    cublasStatus_t cublasStatus = cublasCreate(&cublasHandle);

    file.close();

    // nvtx
    eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
}

NN::~NN()
{
    cudaFree(weights);
    cudaFree(weightGradients);

    cublasDestroy(cublasHandle);
}

void NN::FP(float* inputLayer, float** activations, int N)
{
    float* input = NULL;
    float* output = NULL;
    float* weights_i = NULL;
    
    activations[0] = inputLayer;

    for(int i = 0; i < n_layers - 1; i++)
    {
        int d_prev_i = n_units[i] + 1;
        int d_i      = n_units[i + 1];
        
        ////////////////////////////////////////////////////////
        eventAttrib.color = colors[2]; eventAttrib.message.ascii = "FP-INPUTmalloc_addbiased";
        r1 = nvtxRangeStartEx(&eventAttrib);


        if (input != NULL) cudaFree(input);
        cudaDeviceSynchronize();
        cudaMallocManaged((void **)&input, sizeof(float)* N * d_prev_i);


        if(i == 0) addBiasedUnit(inputLayer, input, N, d_prev_i - 1); // first layer input is the data itself
        else       addBiasedUnit(output, input, N, d_prev_i - 1); // else read from the previous output
        nvtxRangeEnd(r1);
        ////////////////////////////////////////////////////////
        
        weights_i = getWeightsForLayer(i);
        

        ////////////////////////////////////////////////////////
        eventAttrib.color = colors[3]; eventAttrib.message.ascii = "FP-OUTPUTmalloc";
        r1 = nvtxRangeStartEx(&eventAttrib);
        if (output != NULL) cudaFree(output);
        cudaCheckErrors(cudaMallocManaged((void **)&output, sizeof(float)* N * d_i));
        nvtxRangeEnd(r1);
        ////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////
        string text = "FP-OUTPUTmultipy: " + intToString(N) + "x" + intToString(d_prev_i) + "x" + intToString(d_i);
        eventAttrib.color = colors[6]; eventAttrib.message.ascii = text.c_str();
        r1 = nvtxRangeStartEx(&eventAttrib);
        if (N * d_prev_i * d_i > 20100)
            matrixMultiply_parallel(input, N, d_prev_i, weights_i, d_prev_i, d_i, output, N, d_i, cublasHandle);
        else
            matrixMultiply_serial(input, N, d_prev_i, weights_i, d_prev_i, d_i, output, N, d_i);
        cudaDeviceSynchronize();
        nvtxRangeEnd(r1);
        ////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////
        eventAttrib.color = colors[4]; eventAttrib.message.ascii = "FP-copying_activations";
        r1 = nvtxRangeStartEx(&eventAttrib);
        cudaMallocManaged((void **)&activations[i + 1], sizeof(float)* N * d_i);
        // TODO parallel
        for (int j = 0; j < N * d_i; j++)
        	activations[i + 1][j] = output[j];
        nvtxRangeEnd(r1);
        ////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////
        eventAttrib.color = colors[5]; eventAttrib.message.ascii = "FP-sigmoid";
        r1 = nvtxRangeStartEx(&eventAttrib);
        applySigmoidToMatrixElements_serial(output, N, d_i);
        nvtxRangeEnd(r1);
        ////////////////////////////////////////////////////////
    }

    cudaFree(input);
    cudaFree(output);
    cudaDeviceSynchronize();
}

void NN::BP(float** activations, char* labels, int N)
{
    float** deltas = new float*[n_layers];

    for(int i = n_layers - 1; i >= 1; i--)
    {
        int d_i = n_units[i];

        ////////////////////////////////////////////////////////
        eventAttrib.color = colors[2]; eventAttrib.message.ascii = "BP-DELTA";
        r1 = nvtxRangeStartEx(&eventAttrib);
        cudaMallocManaged((void **)&deltas[i], sizeof(float)* N * d_i);
        cudaDeviceSynchronize();

        // TODO: sa dot productom i sigmoid gradient i matrix product
        if(i == n_layers - 1) finalLayerDeltaCalculation_serial(activations[i], deltas[i], labels, N);
        else                  deltaCalculation_serial(activations[i], deltas[i + 1], deltas[i], i, N);
        cudaDeviceSynchronize();
        nvtxRangeEnd(r1);
        ////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////
        eventAttrib.color = colors[3]; eventAttrib.message.ascii = "BP-GRADIENTS";
        r1 = nvtxRangeStartEx(&eventAttrib);
        updateWeightGradients_serial(deltas[i], activations[i - 1], i, N);
        cudaDeviceSynchronize();
        nvtxRangeEnd(r1);
        ////////////////////////////////////////////////////////

    }

    for (int i = 1; i < n_layers; i++)
    	cudaFree(deltas[i]);
    delete[] deltas;
    cudaDeviceSynchronize();

    //gradientChecking(activations[0], labels, N);

    ////////////////////////////////////////////////////////
    eventAttrib.color = colors[4]; eventAttrib.message.ascii = "BP-UPDATEWEIGHTS";
    r1 = nvtxRangeStartEx(&eventAttrib);
    updateWeights_serial();
    cudaDeviceSynchronize();
    nvtxRangeEnd(r1);
    ////////////////////////////////////////////////////////

}

void NN::gradientChecking(float* data, char* labels, int N)
{
	float* weightGradientsApprox = new float[n_weights];
	float** activations;

	for (int i = 0; i < n_weights; i++)
	{
		weights[i] += EPSILON;
		activations = new float*[n_layers];
	    FP(data, activations, N);
	    float error2 = calculateMSE_serial(activations[n_layers - 1], labels, N, n_units[n_layers - 1]);

	    for(int j = 1; j < n_layers; j++) // ne delete prvu - to su data
	        cudaFree(activations[j]);
	    delete [] activations;

	    weights[i] -= 2*EPSILON;
	    activations = new float*[n_layers];
	    FP(data, activations, N);
	    float error1 = calculateMSE_serial(activations[n_layers - 1], labels, N, n_units[n_layers - 1]);

	    for(int j = 1; j < n_layers; j++) // ne delete prvu - to su data
	        cudaFree(activations[j]);
	    delete [] activations;

	    weightGradientsApprox[i] = (error2  -error1) / (2*EPSILON);

	    cout << left << setw(13) << weightGradientsApprox[i]  << " "
	         << left << setw(13) << weightGradients[i] << endl ;

	    weights[i] += EPSILON; // vrati na staro
	}
}

void NN::deltaCalculation_serial(float* layerActivation, float* deltaPrev, float* delta, int i, int N)
{
    //double* temp = new double[(n_units[i] + 1) * N];
    float* temp;
    cudaMallocManaged((void **)&temp, sizeof(float)* N * (n_units[i] + 1));

    ////////////////////////////////////////////////////////
    eventAttrib.color = colors[4]; eventAttrib.message.ascii = "1 - matMul";
    r2 = nvtxRangeStartEx(&eventAttrib);
    if ((n_units[i] + 1) * n_units[i + 1] * N > 20100)
        matrixMultiply_parallel(getWeightsForLayer(i), n_units[i] + 1, n_units[i + 1],
                               deltaPrev, n_units[i + 1], N,
                               temp, n_units[i] + 1, N, cublasHandle);
    else
        matrixMultiply_serial(getWeightsForLayer(i), n_units[i] + 1, n_units[i + 1],
                               deltaPrev, n_units[i + 1], N,
                               temp, n_units[i] + 1, N);
    cudaDeviceSynchronize();
     nvtxRangeEnd(r2);
    ////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////
    eventAttrib.color = colors[4]; eventAttrib.message.ascii = "2 - rest";
    r2 = nvtxRangeStartEx(&eventAttrib);
    for (int r = 0; r < N; r++)
        for (int c = 0; c < n_units[i]; c++)
        {
            float layerActivationValue = layerActivation[IDX_2D_TO_1D(r, c, n_units[i])];
            delta[IDX_2D_TO_1D(c, r, N)] = temp[IDX_2D_TO_1D(c+1, r, N)] * SIGMOID_GRADIENT(layerActivationValue); // +1 da se preskoci prvi clan
        }
    nvtxRangeEnd(r2);
    ////////////////////////////////////////////////////////
    
    cudaFree(temp);
}

void NN::finalLayerDeltaCalculation_serial(float* outputLayerActivation, float* delta, char* labels, int N)
{
    for (int r = 0; r < N; r++)
        for (int c = 0; c < n_units[n_layers - 1]; c++)
        {
            float outputLayerActivationValue = outputLayerActivation[IDX_2D_TO_1D(r, c, n_units[n_layers - 1])];
            float a = SIGMOID(outputLayerActivationValue);
            float b = (double)(c == (int)labels[r]);
            delta[IDX_2D_TO_1D(c, r, N)] = SIGMOID_GRADIENT(outputLayerActivationValue) * (SIGMOID(outputLayerActivationValue) - (double)(c == (int)labels[r]));
        }
}

void NN::Train(float* data, char* labels, int N)
{
    float** activations = new float*[n_layers];

    FP(data, activations, N);
    BP(activations, labels, N);

    N_trained++;

    for(int i = 1; i < n_layers; i++) // ne delete prvu - to su data
        cudaFree(activations[i]);
    delete [] activations;
    cudaDeviceSynchronize();
}

void NN::updateWeightGradients_serial(float* delta, float *prevActivations, int i, int N)
{
	float* gradients = getWeightGradientsForLayer(i - 1);

	////////////////////////////////////////////////////////
    eventAttrib.color = colors[4]; eventAttrib.message.ascii = "input-UPDATEWEIGHTS";
    r2 = nvtxRangeStartEx(&eventAttrib);

	float* sigmoidActivations;
	cudaMallocManaged((void **)&sigmoidActivations, sizeof(float)* N * n_units[i - 1]);
	for (int j = 0; j < N* n_units[i - 1];j++)
		sigmoidActivations[j]=prevActivations[j];
    nvtxRangeEnd(r2);
    ////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////
    eventAttrib.color = colors[5]; eventAttrib.message.ascii = "sigmoid-UPDATEWEIGHTS";
    r2 = nvtxRangeStartEx(&eventAttrib);
	if(i > 1) // ako su activations od prvog layer - ne primjenjuj sigmoid
		applySigmoidToMatrixElements_serial(sigmoidActivations, N, n_units[i - 1]);

	float* biasedActivation;
    cudaMallocManaged((void **)&biasedActivation, sizeof(float)* N * (n_units[i - 1] + 1));

    addBiasedUnit(sigmoidActivations, biasedActivation, N, n_units[i - 1]);
    nvtxRangeEnd(r2);
    ////////////////////////////////////////////////////////
	// TODO: da se pamte sigmoidi, ne računaju


    ////////////////////////////////////////////////////////
    eventAttrib.color = colors[6]; eventAttrib.message.ascii = "gradientsMultiply-UPDATEWEIGHTS";
    r2 = nvtxRangeStartEx(&eventAttrib);
    if (n_units[i] * N * (n_units[i - 1] + 1) > 20100)
        matrixMultiplyAndTranspose_parallel(delta, n_units[i], N,
                              biasedActivation, N, n_units[i - 1] + 1,
                              gradients, n_units[i - 1] + 1, n_units[i], cublasHandle);
    else
        matrixMultiplyAndTranspose_serial(delta, n_units[i], N,
                          biasedActivation, N, n_units[i - 1] + 1,
                          gradients, n_units[i - 1] + 1, n_units[i]);
	cudaDeviceSynchronize();
	nvtxRangeEnd(r2);
    ////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////
    eventAttrib.color = colors[6]; eventAttrib.message.ascii = "normalisation-UPDATEWEIGHTS";
    r2 = nvtxRangeStartEx(&eventAttrib);

    if (N > 1)
    {
        const float alpha = 1.0/(float)N;
        cublasSscal(cublasHandle, (n_units[i - 1] + 1) * n_units[i], &alpha, gradients, sizeof(float));
    }
    nvtxRangeEnd(r2);
    ////////////////////////////////////////////////////////


    cudaFree(biasedActivation);
    cudaFree(sigmoidActivations);
}

void NN::updateWeights_serial(void)
{
    const float negLearningRate = -learningRate;

    cublasSaxpy(cublasHandle, n_weights, &negLearningRate, weightGradients, 1, weights, 1);
    cudaDeviceSynchronize();
}

void NN::TrainLearningCurve(float* data, char* labels, int N, float* dataValidate, char* labelsValidate, int NValidate, bool printError)
{

	int size_chunk = 1;
	int n_chunks = N / size_chunk;

    for (int i = 0; i < n_chunks; i++)
    {
    	float* newData = data + size_chunk * n_units[0] * i;
    	char* newLabels = labels + size_chunk * i;

    	Train(newData, newLabels, size_chunk);


    	if(printError)
    	cout // << left << setw(13)<< intToString(i) + "/" + intToString(n_chunks) << endl;
    		 // << left << setw(13) << Validate(data,         labels        , (i+1)*size_chunk)
    		 << left << setw(13) << Validate(dataValidate, labelsValidate, NValidate) << endl;
    }
}

float NN::Validate(float* data, char* labels, int N)
{
	float** activations = new float*[n_layers];

    FP(data, activations, N);

    float error = calculateClassificationError_serial(activations[n_layers - 1], labels, N, n_units[n_layers - 1]);
    
    for(int i = 1; i < n_layers; i++) // ne delete prvu - to su data
        cudaFree(activations[i]);
    delete [] activations;
    
    return error;
}

void NN::addBiasedUnit(float* input, float* output, int n_rows_input, int n_cols_input)
{
    for(int r = 0; r < n_rows_input; r++)
    {
        output[IDX_2D_TO_1D(r, 0, n_cols_input + 1)] = 1.0f;
        for(int c = 0; c < n_cols_input; c++)
            output[IDX_2D_TO_1D(r, c + 1, n_cols_input + 1)] = input[IDX_2D_TO_1D(r, c, n_cols_input)];
    }
}

float* NN::getWeightsForLayer(int layer)
{
    float* returnWeights = weights;
    
    for(int i = 1; i <= layer; i++)
        returnWeights += (n_units[i - 1] + 1) * n_units[i]; // move the weights_i to point to the next matrix
        
    return returnWeights;
}

float* NN::getWeightGradientsForLayer(int layer)
{
    float* returnWeights = weightGradients;

    for(int i = 1; i <= layer; i++)
        returnWeights += (n_units[i - 1] + 1) * n_units[i]; // move the weights_i to point to the next matrix

    return returnWeights;
}

void NN::printWeights(string dirName)
{
	struct stat st = {0};
	if (stat(dirName.c_str(), &st) == -1) {
	    mkdir(dirName.c_str(), 0777);
	}

	for (int i = 0; i < n_layers - 1; i++)
		outputFloatMatrixToFile(getWeightsForLayer(i), n_units[i] + 1, n_units[i + 1], dirName + "/" + intToString(i) + ".txt");
}

void NN::printWeightGradients(string dirName)
{
	struct stat st = {0};
	if (stat(dirName.c_str(), &st) == -1) {
	    mkdir(dirName.c_str(), 0777);
	}

	for (int i = 0; i < n_layers - 1; i++)
		outputFloatMatrixToFile(getWeightGradientsForLayer(i), n_units[i] + 1, n_units[i + 1], dirName + "/" + intToString(i) + ".txt");
}

void NN::printActivations(string dirName, float** activations, int N)
{
	struct stat st = {0};
	if (stat(dirName.c_str(), &st) == -1) {
	    mkdir(dirName.c_str(), 0777);
	}

	for (int i = 0; i < n_layers; i++)
	{
		outputFloatMatrixToFile(activations[i], N, n_units[i], dirName + "/" + intToString(i) + ".txt");
	}
}

void NN:: Save(string dirName, string fileName)
{
    struct stat st = {0};
    if (stat(dirName.c_str(), &st) == -1) {
        mkdir(dirName.c_str(), 0777);
    }

    fileName = dirName + "/" + fileName;

    ofstream file(fileName.c_str());
    if (!file.is_open())
        cout << "ERROR READING FILE" << endl;

    file << n_layers << endl;

    for (int i = 0; i < n_layers; i++)
        file << n_units[i] << " ";
    file << endl;

    file << n_weights << endl
         << learningRate << endl
         << N_trained << endl;

    for (int i = 0; i < n_layers - 1; i++)
    {
        float* array = getWeightsForLayer(i);
        int n_rows = n_units[i] + 1;
        int n_cols = n_units[i + 1];

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
    }

    file.close();
}
