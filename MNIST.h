#ifndef MNIST_H
#define MNIST_H

#include <string>
#include "myUtils.h"

using namespace std;

int reverseInt(int i);
float* readMNISTDataNormalised1DUnified(int* N_train, int* n_rows, int* n_cols, string filename);
char* readMNISTLabelsUnified(int *N, string filename);

#endif
