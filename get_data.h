#ifndef __GET_DATA_H__
#define __GET_DATA_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define N_SAMPLES_1 60000
#define N_DIMS 784
#define N_CLASSES 10
#define N_TEST_SAMPLES_1 10000
#define N_TEST_SAMPLES 10000
#define N_SAMPLES 60000



// Function to read training and test data and store them appropriately
void read_csv_file(double** data, double* y_temp, double** y, char* dataset);

// Function to scale the dataset
void scale_data(double** data, char* dataset);

// Function to normalize the dataset
void normalize_data(double** X_train, double** X_test);

// Function to get data in batches
void fetch_batch(double** batch_data, double** batch_labels, double batch_size, int batch_index, double** img, double** lbl);

#endif
