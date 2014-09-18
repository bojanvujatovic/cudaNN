#ifndef ELASTIC_DEFORMATION_H
#define ELASTIC_DEFORMATION_H

void elasticDeformation_serial(float* input, float* output, int n_rows, int n_cols, double sigma, double alpha, int kernelWidth, float amplitude);

void fieldConvolutionAndScaling(float* field, float* output, float* kernel, int n_rows, int n_cols, int kernelWidth, float alpha);

void elasticDeformation_parallel(float* trainData, char* labels, int N_train,
                                 float* deformedExamples,
                                 char* deformedExamplesLabels, float* randDisplField,
                                 int n_rows, int n_cols, int N_deform, int i);



#endif
