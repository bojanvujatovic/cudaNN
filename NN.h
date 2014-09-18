#ifndef NN_H
#define NN_H

#include <cmath>
#include <cstdlib> 
#include <cublas_v2.h>

#include <nvToolsExt.h>

#define RAND_INITIALISE_WEIGHT ((float)rand() / RAND_MAX) * 1.f  - 0.5f
//#define RAND_INITIALISE_WEIGHT 1.0f

#define EPSILON 0.0001f

class NN
{
  private:
    int n_layers;
    int* n_units;
    int n_weights;
    
    long int N_trained;



    float learningRate;
    
    float* weights;
    float* weightGradients;

    nvtxRangeId_t r1, r2;
    nvtxEventAttributes_t eventAttrib;


    void addBiasedUnit(float* input, float* output, int width, int height); // TODO NPP
    float* getWeightsForLayer(int layer);
    float* getWeightGradientsForLayer(int layer);
    void FP(float* inputLayer, float** activations, int N);
    void BP(float** activations, char* labels, int N);
    void updateWeightGradients_serial(float* delta, float* prevActivations, int i, int N);
    void updateWeights_serial(void);
    void finalLayerDeltaCalculation_serial(float* outputLayer, float* delta, char* labels, int N);
    void deltaCalculation_serial(float* layerActivation, float* deltaPrev, float* delta, int i, int N);
    
    void gradientChecking(float* data, char* labels, int N);

    void printWeights(string filename);
    void printWeightGradients(string filename);
    void printActivations(string dirName, float** activations, int N);
    
  public:
    cublasHandle_t cublasHandle;

    NN(int n_layers, int* n_units, float learningRate);
    NN(string filename);
    ~NN();
    
    void Train(float* data, char* labels, int N);
    void TrainLearningCurve(float* data, char* labels, int N, float* dataValidate, char* labelsValidate, int NValidate, bool printError = true);
    float Validate(float* data, char* labels, int N);
    char* Predict(float* data, int N); // TODO
    
    void Save(string dirName, string fileName);

};

#endif
