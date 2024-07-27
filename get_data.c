#include "get_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
// #define N_SAMPLES 60000
// #define N_DIMS 784
// #define N_CLASSES 10
// #define N_TEST_SAMPLES 10000


// Function to read training and test data and store them appropriately
void read_csv_file(double** data, double* label_i, double** lbl_vec, char* dataset){
    printf("in read_csv_file\n");
    FILE *file;
    if(strcmp(dataset, "train") == 0){
       // printf("dataset = train\n");
        file = fopen("mnist_train.csv", "r");
    }
    else if(strcmp(dataset, "test") == 0){
        file = fopen("mnist_test.csv", "r");
    }
    if(file == NULL){
        printf("Error reading the file!");
        exit(1);
    }
    char buffer[3200];  //784 pixel * (3 char + 1 comma) + 1 label char + 1 comma = 3138 -> one line of csv file
    
    int i = 0;
    //printf("before while\n");
    while(fgets(buffer, sizeof(buffer), file)){     //legge da file tanti 3200 char alla volta e salva in buffer
        char* token = strtok(buffer, ",");  //saves in array 'token' first value of buffer ( ',' is delimitator)
        
        // printf("token: '%s'\n",token);  
        char *endptr;
        long int num;

        // // Convert the string to a long integer
        // num = strtol(token, &endptr, 10);
        // if (endptr == token) {
        //     printf("No digits were found.\n");
        // } else if (*endptr != '\0') {
        //     printf("Invalid character: %c\n", *endptr);
        // } else {
        //     printf("The number is: %ld\n", num);
        // }

        // Convert the string to a long integer
       
        int j = 0;
        while(token != NULL){
            
             if(j == 0){  
                // printf("atoi token str: %s\n", (token));   
                // printf("atoi token: %f\n",strtof(token,NULL));  
                     
                //label_i[i] = (double)atoi(token);  //transforms char in int (primo token, label)
                label_i[i] = strtof(token,NULL);
                // printf("label_i[%d]: %f \n", i, label_i[i]);
            }
            else{
                data[i][j-1] = strtof(token,NULL);
                //data[i][j-1] = (double)atoi(token);   //salva valori pixel in matrice (ma non il primo che è la label)
                // printf("data[%d][%d]: %f\n",i,j-2, data[i][j-2]);
            }
            j++;
            token = strtok(NULL, ",");    //continua a processare i token della riga letta
        }
        i++;
    }
   // printf("after while\n");
    fclose(file);
   
    int total_samples;
    if(strcmp(dataset, "train") == 0){
        total_samples = N_SAMPLES;      //60000
    }
    else if(strcmp(dataset, "test") == 0){
        total_samples = N_TEST_SAMPLES; //10000
    }
    for(int i=0;i<total_samples;i++){
        for(int j=0;j<N_CLASSES;j++){
            if((int)label_i[i] == j){    //se label i-esima tra tutte quelle del file == j
                // printf("(int)label_i[%d] == j: %d - %d\n",i,(int)label_i[i],j);
                lbl_vec[i][j] = 1.0;          //y di sample i class j (target.one_hot)
            }
            else{
                lbl_vec[i][j] = 0.0;
            }
        }
    }
     
      for(int i=0;i<total_samples;i++){
        // printf("[%s]all read tokens:label_i[%d]: %f \n",dataset,i,label_i[i]);
        for(int j=0;j<N_CLASSES;j++){
            
                // printf("lbl_vec[%d][j]: %f\n",i,lbl_vec[i][j]);          //y di sample i class j (target.one_hot)
           
        }
        // printf("\n");
    }
    ///printf("end of read_csv_file\n");
}

// Function to scale the dataset
void scale_data(double** data, char* dataset){
    printf("in scale_data\n");
    int total_samples;
    if(strcmp(dataset, "train") == 0){
        total_samples = N_SAMPLES;
    }
    else if(strcmp(dataset, "test") == 0){
        total_samples = N_TEST_SAMPLES;
    }
    for(int i=0;i<total_samples;i++){
        for(int j=0;j<N_DIMS;j++){  //N_DIMS=784
            data[i][j] = (double)data[i][j]/(double)255.0;   //scale between 0 and 1 every pixel
        }
    }
}

void fetch_batch(double** batch_data, double** batch_labels, double batch_size, int batch_index, double** img, double** lbl){
    int i;
    printf("in fetch batch\n");
    for (i = 0; i < batch_size; i++) {
        int data_index = batch_index * batch_size + i;
        if (data_index >= N_SAMPLES){
            printf("There are no more samples!\n");
            break;  // Prevent out-of-bounds access
        }

        for (int j = 0; j < N_DIMS; j++) {
            batch_data[i][j] = img[data_index][j];
            
        }
        for(int j=0;j<N_CLASSES;j++){
            batch_labels[i][j] = lbl[data_index][j];
        }

    }
    //batch_index += batch_index + batch_size;
    printf("end of fetch_batch\n");
}

// Function to normalize the dataset
void normalize_data(double** X_train, double** X_test){    //X_train -> data (pixel values)
    printf("in normalize_data\n");
    double* mean = malloc(N_DIMS*sizeof(double));
    double total = N_SAMPLES;   //60000
    for(int i=0;i<N_DIMS;i++){
        double sum = 0.0;
        for(int j=0;j<N_SAMPLES;j++){
            sum += X_train[j][i];
        }
        mean[i] = sum/total;    //media su ogni dimensione (sommando tutti i sample)
    }
    double* sd = malloc(N_DIMS*sizeof(double));
    for(int i=0;i<N_DIMS;i++){ //784
        double sum = 0.0;
        for(int j=0;j<N_SAMPLES;j++){
            sum += pow(X_train[j][i] - mean[i], 2);     //standard deviation
        }
        sd[i] = sqrt(sum/total);
    }
    for(int i=0;i<N_DIMS;i++){
        for(int j=0;j<N_SAMPLES;j++){
            if(sd[i]>0.0001){
                X_train[j][i] = (double)(X_train[j][i] - mean[i])/(double)sd[i];
            }
        }
        for(int j=0;j<N_TEST_SAMPLES;j++){
            if(sd[i]>0.0001){
                X_test[j][i] = (double)(X_test[j][i] - mean[i])/(double)sd[i];
            }
        }
    }
    free(sd);
    free(mean);
    printf("end of normalize\n");
}