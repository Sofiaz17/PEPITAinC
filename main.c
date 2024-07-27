#include "activation.h"
#include "get_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#define NUM_LAYERS 3
#define NUM_NEU_L1 784
#define NUM_NEU_L2 128
#define NUM_NEU_L2_1 10
#define NUM_NEU_L3 10
#define BATCH_SIZE_1 3
#define BATCH_SIZE 128
#define EPOCHS 50

int neu_per_lay[] = {NUM_NEU_L1, NUM_NEU_L2, NUM_NEU_L3};

const double mom_gamma = 0.9;
int batch_index = 0;
int exp_number = 0;

void matrix_multiply(double* A, double* B, double* C, int A_rows, int A_cols, int B_cols) {
// Initialize the result matrix C to zero
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            C[i * B_cols + j] = 0.0;
        }
    }

    // Perform the matrix multiplication
    for (int i = 0; i < A_rows; i++) {
        for (int k = 0; k < A_cols; k++) {
            for (int j = 0; j < B_cols; j++) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

void matrix_transpose(double* A, double* B, int rows, int cols) {
    // A is rows x cols
    // B will be cols x rows (transpose of A)

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

void matrix_subtract(double* A, double* B, double* C, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            C[i * cols + j] = -A[i * cols + j] + B[i * cols + j];
        }
    }
}

int find_max(int array[], int size) {
    if (size <= 0) {
        // Handle the case where the array size is non-positive
        fprintf(stderr, "Array size must be greater than 0\n");
        return 1;
    }

    int max_value = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
        }
    }
    return max_value;
}

typedef struct NeuralNet{
    int n_layers;
    int* n_neurons_per_layer;
    double*** w;
    double** b;
    double*** momentum_w;
    //double*** momentum2_w;
    double** momentum_b;
    //double** momentum2_b;
    double** error;
    double** actv_in;
    double** actv_out;
    double* targets;
}NeuralNet;
#include <math.h>
#include <stdlib.h>

double randn1() {
    static int hasSpare = 0;
    static double spare;
    if (hasSpare) {
        hasSpare = 0;
        return spare;
    }

    hasSpare = 1;
    double u, v, s;
    do {
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return u * s;
}

double randn (double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;

  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }

  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);

  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (double) X1);
}


void initialize_net(NeuralNet* nn){
    int i,j,k;

    if(nn->n_layers == 0){
        printf("No layers in Neural Network...\n");
        return;
    }

    printf("Initializing net...\n");

    for(i=0;i<nn->n_layers-1;i++){

        for(j=0;j<nn->n_neurons_per_layer[i];j++){          //££

            for(k=0;k<nn->n_neurons_per_layer[i+1];k++){     //££

                // double nin = nn->n_neurons_per_layer[i+1];
                // double limit = sqrtf(6.0f/nin);
                // double scale = rand() / (double) RAND_MAX;

                double nin = nn->n_neurons_per_layer[i]; // fan_in
                double stddev = sqrtf(6.0f / nin);
                nn->w[i][j][k] = stddev * randn(0.0,1.0); // randn() should generate a random number from a standard normal distribution
                
                // if(i==0){
                //     printf("w[%d][%d][%d]: %f\n",i,j,k,nn->w[i][j][k]);
                // }
                // if(k==1){
                //     printf("w[%d][%d][%d]: %.20f\n",i,j,k,nn->w[i][j][k]);
                // }

                //CHECK VALUE
                // Initialize Output Weights for each neuron
                // nn->w[i][j][k] = -limit + scale * (limit + limit);
                nn->momentum_w[i][j][k] = 0.0;
                //printf("w[%d][%d][%d] defined \n",i,j,k);
            }
            //printf("\n");
           
            nn->b[i][j] = 0.0;
            nn->momentum_b[i][j] = 0.0;
                //printf("bias[%d][%d]\n",i,j);
            
        }
        //printf("\n");
    }
    
    // for (j=0; j<nn->n_neurons_per_layer[nn->n_layers-1]; j++){
    //     //printf("nn->n_neurons_per_layer[nn->n_layers-1]: %d\n",nn->n_neurons_per_layer[nn->n_layers-1]);
    //     //printf("nn->n_layers-1: %d\n",nn->n_layers-1);
    //     nn->b[nn->n_layers-1][j] = 0.0;
    //     //printf("bias[n_layers-1][%d]\n",nn->n_layers-1,j);
    // }

    // for(int i=0;i<nn->n_layers;i++){
    //     for(int j=0;j<nn->n_neurons_per_layer[i];j++){    //+1??                    //££
    //         //printf("actv_in = 0.0\n");
    //         nn->actv_in[i][j] = 0.0;
    //     }
    // }

    printf("net initialized\n");

}

void free_NN(NeuralNet* nn);


// Function to create a neural network and allocate memory
NeuralNet* newNet(){
    printf("in newNet\n");
    int i,j;
    //space for nn
    NeuralNet* nn = malloc(sizeof(struct NeuralNet));
    nn->n_layers = NUM_LAYERS;
    //space for layers
    nn->n_neurons_per_layer = (int*)malloc(nn->n_layers * sizeof(int));

    //initialize layer with num neurons
    for(i=0; i<nn->n_layers; i++){
        nn->n_neurons_per_layer[i] = neu_per_lay[i];
    }

    //space for weight matrix and weight momentum (first dimension)->layer
    nn->w = (double***)malloc((nn->n_layers-1)*sizeof(double**));
    nn->momentum_w = (double***)malloc((nn->n_layers-1)*sizeof(double**));
    //space for bias matrix and bias momentum (first dimension)->layer
    nn->b = (double**)malloc((nn->n_layers-1)*sizeof(double*));
    nn->momentum_b = (double**)malloc((nn->n_layers-1)*sizeof(double*));
    

    for(int i=0;i<nn->n_layers-1;i++){
        //weight matrix and momentum (second dimension)->neurons of curr layer
        nn->w[i] = (double**)malloc((nn->n_neurons_per_layer[i])*sizeof(double*));   //+1?? (per tutti in for)  //££
        nn->momentum_w[i] = (double**)malloc((nn->n_neurons_per_layer[i])*sizeof(double*));                   //££
        //bias matrix and mometum (second dimension)->neurons of curr layer
    
        nn->b[i] = (double*)malloc((nn->n_neurons_per_layer[i])*sizeof(double));                    //££
        nn->momentum_b[i] = (double*)malloc((nn->n_neurons_per_layer[i])*sizeof(double));                    //££
        
        //space for weight matrix and weight momentum (third dimension)->neurond of next layer
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){  //+1??
            nn->w[i][j] = malloc((nn->n_neurons_per_layer[i+1])*sizeof(double));     //+1??             //££
            nn->momentum_w[i][j] = malloc((nn->n_neurons_per_layer[i+1])*sizeof(double));                    //££
        }
    }

    //nn->b[nn->n_layers-1] =  (double*)malloc((nn->n_neurons_per_layer[nn->n_layers-1])*sizeof(double));
    
    //space for error matrix for each neuron in each layer(layer dimension) 
    nn->error = (double**)malloc((nn->n_layers)*sizeof(double*));
    //space for input and output to activation functions (layer dimension)
    nn->actv_in = (double**)malloc((nn->n_layers)*sizeof(double*));
    nn->actv_out = (double**)malloc((nn->n_layers)*sizeof(double*));
    
    for(int i=0;i<nn->n_layers;i++){
        //space for error matrix for each neuron in each layer(neuron dimension) 
        nn->error[i] = (double*)malloc((nn->n_neurons_per_layer[i])*sizeof(double)); //+1??                    //££
        //space for input and output to activation functions (neuron dimension)
        nn->actv_in[i] = (double*)malloc((nn->n_neurons_per_layer[i])*sizeof(double));                    //££
        nn->actv_out[i] = (double*)malloc((nn->n_neurons_per_layer[i])*sizeof(double));                    //££
    }
    //space for desired outputs (one hot vector)
    nn->targets = malloc((nn->n_neurons_per_layer[nn->n_layers-1])*sizeof(double));    //+1??                    //££
    
    // Initialize the weights
    initialize_net(nn);

    printf("end newNet\n");
    //return SUCCESS_CREATE_ARCHITECTURE;
    return nn;
}

// Function to free the dynamically allocated memory
void free_NN(struct NeuralNet* nn){
    printf("in free_NN\n");
    if(!nn) return;
    for(int i=0;i<nn->n_layers-1;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££
            free(nn->w[i][j]);
            free(nn->momentum_w[i][j]);
           // free(nn->momentum2_w[i][j]);
        }
        free(nn->w[i]);
        free(nn->momentum_w[i]);
       // free(nn->momentum2_w[i]);
        free(nn->b[i]);
        free(nn->momentum_b[i]);
        //free(nn->momentum2_b[i]);
    }
    free(nn->w);
    free(nn->momentum_w);
    //free(nn->momentum2_w);
    free(nn->b);
    free(nn->momentum_b);
    //free(nn->momentum2_b);
    for(int i=0;i<nn->n_layers;i++){
        free(nn->actv_in[i]);
        free(nn->actv_out[i]);
        free(nn->error[i]);
    }
    free(nn->actv_in);
    free(nn->actv_out);
    free(nn->error);
    free(nn->targets);
    free(nn->n_neurons_per_layer);
    free(nn);
}


// Function for forward propagation step
void forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss){
    // printf("in forward prop\n");
    //initialize input to actv for every layer
    for(int i=0;i<nn->n_layers;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){    //+1??                    //££
            //printf("actv_in = 0.0\n");
            nn->actv_in[i][j] = 0.0;
        }
    }
    for(int i=1;i<nn->n_layers;i++){
        // printf("i: %d\n", i);
        // Compute the weighted sum -> add bias to every input
        for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££ + 0
            nn->actv_in[i][j] += 1.0 * nn->b[i-1][j];       //why *1.0?
            // printf("FP bias,actv_in[%d][%d]: %f\n",i,j,nn->actv_in[i][j]);
            //printf("actv_in + bias\n");
        }
        //  printf("nn->n_neurons_per_layer[i-1]: %d\n", nn->n_neurons_per_layer[i-1]);
        //  printf("nn->n_neurons_per_layer[i]: %d\n", nn->n_neurons_per_layer[i]);

        //add previous weighted output
        for(int k=0;k<nn->n_neurons_per_layer[i-1];k++){
            for(int j=0;j<nn->n_neurons_per_layer[i];j++){                       //££ + 0
                nn->actv_in[i][j] += nn->actv_out[i-1][k] * nn->w[i-1][k][j];
                // printf("FP actv_in[%d][%d]: %f\nnn->actv_out[%d][%d]: %f\nnn->w[%d][%d][%d]\n",i,j,nn->actv_in[i][j],i-1,k,nn->actv_out[i-1][k],i-1,k,j,nn->w[i-1][k][j]);
                //printf("[i:%d][k:%d][j:%d] actv_in con w\n", i,k,j);
                //printf("[i:%d][k:%d][j:%d]\n", i,k,j);
            }
             //printf("[i:%d][k:%d]\n", i,k);
            //printf("[i:%d][k:%d]\n", i,k);
            //printf("nn->n_neurons_per_layer[i-1]\n", nn->n_neurons_per_layer[i-1]);
            // printf("\n");
        }
        // Apply non-linear activation function to the weighted sums
        //if last layer, apply softmax
        if(i == nn->n_layers-1){
            // printf("i==n_layers-1\n");
            if(strcmp(loss, "mse") == 0){
                for(int j=0;j<nn->n_neurons_per_layer[i];j++){    //+1??                    //££ + 0
                    nn->actv_out[i][j] = sigmoid(nn->actv_in[i][j]);
                }
            }
            else if(strcmp(loss, "ce") == 0){
                // double max_input_to_softmax = (double)INT_MIN;
                // for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££ + 0
                //     if(fabs(nn->actv_in[i][j]) > max_input_to_softmax){
                //         max_input_to_softmax = fabs(nn->actv_in[i][j]);
                //     }
                // }
                // double deno = 0.0;
                // for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££ + 0
                //     nn->actv_in[i][j] /= max_input_to_softmax;
                //     deno += exp(nn->actv_in[i][j]);
                // }
                // for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££ + 0
                //     nn->actv_out[i][j] = (double)exp(nn->actv_in[i][j])/(double)deno;
                    
                // }
             
                // double deno = 0.0;
                // for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££ + 0
                //     deno += exp(nn->actv_in[i][j]);
                // }
                // for(int j=0;j<nn->n_neurons_per_layer[i];j++){                    //££ + 0
                //     nn->actv_out[i][j] = (double)exp(nn->actv_in[i][j])/(double)deno;   
                // }
                double max_input_to_softmax = (double)INT_MIN;
                    for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                        if(nn->actv_in[i][j] > max_input_to_softmax){
                            max_input_to_softmax = nn->actv_in[i][j];
                        }
                    }
                    double deno = 0.0;
                    for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                        nn->actv_in[i][j] -= max_input_to_softmax;
                        deno += exp(nn->actv_in[i][j]);
                    }
                    for(int j=0;j<nn->n_neurons_per_layer[i];j++){
                        nn->actv_out[i][j] = exp(nn->actv_in[i][j]) / deno;
                        // printf("FP,actv_out: %f\n", nn->actv_out[i][j]);
                    }
                // printf("ce loss\n");
            }
        } //if other layers, apply something else
        else{
            for(int j=0;j<nn->n_neurons_per_layer[i];j++){        //+1??                    //££ + 0 
                if(strcmp(activation_fun, "sigmoid") == 0){
                    nn->actv_out[i][j] = sigmoid(nn->actv_in[i][j]);
                }
                else if(strcmp(activation_fun, "tanh") == 0){
                    nn->actv_out[i][j] = tanh(nn->actv_in[i][j]);
                }
                else if(strcmp(activation_fun, "relu") == 0){
                    nn->actv_out[i][j] = relu(nn->actv_in[i][j]);
                    //nn->actv_out[i][j] = nn->actv_in[i][j];
                    // printf("[%d],actv_out: %f\n", nn->actv_out[i][j]);
                    //printf("relu activation\n");
                }
                else{
                    nn->actv_out[i][j] = relu(nn->actv_in[i][j]);
                }
            }
        }
    }
}


// Function to calculate loss
double calc_loss(struct NeuralNet* nn, char* loss){
    // printf("in calc loss\n");
    int i;
    double running_loss = 0.0;
    int last_layer = nn->n_layers-1;
    for(i=0;i<nn->n_neurons_per_layer[last_layer];i++){       //+1???                    //££ + 0
        if(strcmp(loss, "mse") == 0){
            running_loss += (nn->actv_out[last_layer][i] - nn->targets[i]) * (nn->actv_out[last_layer][i] - nn->targets[i]);
        }
        else if(strcmp(loss, "ce") == 0){
            double epsilon = 1e-10;
            running_loss -= nn->targets[i]*(log(nn->actv_out[last_layer][i] + epsilon));
            
        }
        // printf("running loss[%d]: %f\n",i, running_loss);
	}
    if(strcmp(loss, "mse") == 0){
        running_loss /= BATCH_SIZE;
    }
    return running_loss;
}


void shuffle(int* arr, size_t n) {
    if (n > 1) {
        // Seed the random number generator to ensure different results on each run
        srand(time(NULL));

        for (size_t i = n - 1; i > 0; i--) {
            // Generate a random index j such that 0 <= j <= i
            size_t j = rand() % (i + 1);

            // Swap arr[i] with arr[j]
            int t = arr[i];
            arr[i] = arr[j];
            arr[j] = t;
        }
    }
}

// Function to train the model for 1 epoch
void model_train(struct NeuralNet* nn, double** X_train, double** y_train, double* y_train_temp, double** X_test, double** y_test, double* y_test_temp,
                 char* activation_fun, char* loss_fun, char* opt, double learning_rate){
    printf("in model train\n");
    // Create an array for generating random permutation of training sample indices
    // int shuffler_train[N_SAMPLES];    //better to do on whole dataset???
    // for(int i=0;i<N_SAMPLES;i++){
    //     shuffler_train[i] = i;
    // }
    // shuffle(shuffler_train, N_SAMPLES);

    //FIRST FORWARD PASS FOR ACTIVATIONS (dropout mask)
   
    // Start training the model for 1 epoch and simultaneously calculate the training error and accuracy
  
    //double* test_accs = (double*)malloc(EPOCHS*sizeof(double));
    double test_accs[EPOCHS];
    //double* train_losses = (double*)malloc(EPOCHS*sizeof(double));
    double train_losses[EPOCHS];
    //double* train_accs = (double*)malloc(EPOCHS*sizeof(double));
    double train_accs[EPOCHS];
    double curr_loss = 0.0;

    // double* B = (double*)malloc(N_CLASSES*N_DIMS*sizeof(double));
    double B[N_CLASSES*N_DIMS];
    
    int nin = 28*28;
    double sd = sqrtf(6.0f/(double)nin);
    for(int i=0;i<N_CLASSES;i++){
        // printf("B:\n");
        for(int j=0;j<N_DIMS;j++){
            double rand_num = ((double)rand() / RAND_MAX);
            B[i * N_DIMS + j] = (rand_num * 2 * sd - sd) * 0.05;    //0.05
            //B[i * N_DIMS + j] = rand_num*2*sd*0.05;
             
            //  printf("%f ",B[i * N_DIMS + j] );
        }
        // printf("\n");
    }

    int max_num_neu = find_max(neu_per_lay, NUM_LAYERS);

    double* outputs = (double*)malloc(BATCH_SIZE*N_CLASSES*sizeof(double));
    //double outputs[BATCH_SIZE*N_CLASSES];
    double* targets = (double*)malloc(BATCH_SIZE*N_CLASSES*sizeof(double));
    //double targets[BATCH_SIZE*N_CLASSES];
    double* inputs = (double*)malloc(BATCH_SIZE*N_DIMS*sizeof(double));
    //double inputs[BATCH_SIZE*N_DIMS];
    double** layers_act = (double**)malloc((NUM_LAYERS-1)*sizeof(double*));     //n_lay-1, batch, neu
    for(int i=0; i<NUM_LAYERS-1;i++){
        layers_act[i] = (double*)malloc(BATCH_SIZE*nn->n_neurons_per_layer[i+1]*sizeof(double));
    }
    double* error = (double*)malloc(BATCH_SIZE*N_CLASSES*sizeof(double));
    //double error[BATCH_SIZE*N_CLASSES];
    double* error_input = (double*)malloc(BATCH_SIZE*N_DIMS*sizeof(double));
    //double error_input[BATCH_SIZE*N_DIMS];
    double* mod_inputs = (double*)malloc(BATCH_SIZE*N_DIMS*sizeof(double));
    //double mod_inputs[BATCH_SIZE*N_DIMS];
    double* mod_outputs = (double*)malloc(BATCH_SIZE*N_CLASSES*sizeof(double));
    //double mod_outputs[BATCH_SIZE*N_CLASSES];
    double** mod_layers_act = (double**)malloc((NUM_LAYERS-1)*sizeof(double*));     //n_lay-1, batch, neu
    for(int i=0; i<NUM_LAYERS-1;i++){
        mod_layers_act[i] = (double*)malloc(BATCH_SIZE*nn->n_neurons_per_layer[i+1]*sizeof(double));
    }
    double* mod_error = (double*)malloc(BATCH_SIZE*N_CLASSES*sizeof(double));
    //double mod_error[BATCH_SIZE*N_CLASSES];
    double* delta_w = (double*)malloc(max_num_neu*max_num_neu*sizeof(double));
    double** delta_w_all = (double**)malloc((NUM_LAYERS-1)*sizeof(double*));
    // for(int l=0; l<NUM_LAYERS-1;l++){
    //     delta_w_all[l] = (double*)malloc(nn->n_neurons_per_layer[l+1]*nn->n_neurons_per_layer[l]*sizeof(double));
    // }
    double* mod_error_T = (double*)malloc(N_CLASSES*BATCH_SIZE*sizeof(double));
    //double mod_error_T[N_CLASSES*BATCH_SIZE];
    double** delta_lay_act = (double**)malloc((NUM_LAYERS-1)*sizeof(double*));
    for(int i=0; i<NUM_LAYERS-1;i++){
        delta_lay_act[i] = (double*)malloc(BATCH_SIZE*nn->n_neurons_per_layer[i+1]*sizeof(double));
    }
    double** delta_lay_act_T = (double**)malloc((NUM_LAYERS-1)*sizeof(double*));
    for(int i=0; i<NUM_LAYERS-1;i++){
        delta_lay_act_T[i] = (double*)malloc(nn->n_neurons_per_layer[i+1]*BATCH_SIZE*sizeof(double));
    }
    printf("memory allocated\n");
    
  
    for(int epoch=0;epoch<EPOCHS;epoch++){      //remember not real batch size but tot number of images used (one per time)
        double loss = 0.0;
        
        if(epoch == 50){
            learning_rate *= 0.2;
        }
        int shuffler_train[N_SAMPLES];
        for(int i=0;i<N_SAMPLES;i++){
            shuffler_train[i] = i;
        }

        shuffle(shuffler_train, N_SAMPLES);

        double running_loss = 0.0;
        int batch_count = 0;
        double total = 0.0;
        double correct = 0.0;
        
        int idx = -1;
        double max_val = (double)INT_MIN;
        int in_cnt_train = 0;
        int in_cnt_train_2 = 0;

        //alloca batch array
        for(int batch_num=0;batch_num<floor(N_SAMPLES/BATCH_SIZE);batch_num++){
            //assegna batch
            printf("[%d]TRAIN BATCH %d\n", epoch, batch_num);
                
            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
                //do masks
            
                for(int j=0;j<nn->n_neurons_per_layer[0];j++){          //££ + 0 
                    // nn->actv_out[0][j] = X_train[shuffler_train[batch_elem+batch_count*BATCH_SIZE]][j];      //assign input ??????i?????
                    nn->actv_out[0][j] = X_train[shuffler_train[in_cnt_train]][j];
                    // printf("[%d][elem:%d] actv_out[0][%d]: %f\n",epoch,batch_elem, j,nn->actv_out[0][j]);
                    // printf(" batch_num[%d]-nn->actv_out[0][%d] = X_train[%d]][%d]\n",batch_num,j,shuffler_train[in_cnt_train],j);
                    // if(j==300){
                        // printf("index example: %d\n", shuffler_train[in_cnt_train]);
                        // printf("[batch:%d][elem:%d]X_train:%f\n",batch_num,batch_elem,X_train[shuffler_train[in_cnt_train]][j]);
                    // }
                }
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
                    nn->targets[j] = y_train[shuffler_train[in_cnt_train]][j];        //assign target labels (one hot)
                    // printf("[epoch:%d][elem:%d]targets[%d]: %f\n",epoch,batch_elem, j,nn->targets[j]);
                    // if(j==300)
                    // printf("[batch:%d][elem:%d]y_train:%f\n",batch_num,batch_elem,y_train[shuffler_train[in_cnt_train]][j]);
                 }
                //  printf("y_train_label: %f\n", y_train_temp[in_cnt_train]);
                for(int in_neu=0;in_neu<N_DIMS;in_neu++){
                    inputs[batch_elem*N_DIMS+in_neu] = nn->actv_out[0][in_neu];
                    // printf("[elem:%d]inputs[%d]: %f\n",batch_elem, batch_elem*N_DIMS+in_neu,inputs[batch_elem*N_DIMS+in_neu]);
                    // printf("input indexes: %d\n", batch_elem*N_DIMS+in_neu);
                    // printf("inputs[%d]: %f\n",batch_elem*N_DIMS+in_neu,inputs[batch_elem*N_DIMS+in_neu]);
                    // printf("[elem:%d]actv_out[0][%d]: %f\n", batch_elem,in_neu, nn->actv_out[0][in_neu]);
                }
               
                forward_propagation(nn, activation_fun, loss_fun);
               

                //array to save output for every batch
                for(int out_neu=0;out_neu<N_CLASSES;out_neu++){
                    outputs[batch_elem*N_CLASSES+out_neu] = nn->actv_out[nn->n_layers-1][out_neu]; 
                    targets[batch_elem*N_CLASSES+out_neu] = nn->targets[out_neu];
                    // printf("[elem:%d]outputs[%d]: %f\n",batch_elem, batch_elem*N_CLASSES+out_neu,outputs[batch_elem*N_CLASSES+out_neu]);
                    // printf("[epoch:%d][elem:%d][1]actv_out[%d][%d]: %f\n", epoch,batch_elem,nn->n_layers-1,out_neu, nn->actv_out[nn->n_layers-1][out_neu]);
                    // printf("[elem:%d]targets[%d]: %f\n",batch_elem, batch_elem*N_CLASSES+out_neu,targets[batch_elem*N_CLASSES+out_neu]);
                    // printf("[elem:%d]nn->targets[%d]: %f\n", batch_elem,out_neu, nn->targets[out_neu]);
         
                }
                // printf("\n");
                for(int lay=0;lay<NUM_LAYERS-1;lay++){
                    for(int neu=0;neu<nn->n_neurons_per_layer[lay+1];neu++){
                        layers_act[lay][batch_elem*nn->n_neurons_per_layer[lay+1]+neu] = nn->actv_out[lay+1][neu]; //?????????
                        // printf("[ep:%d][btc:%d]layers_act[%d][%d]: %f\n",epoch,batch_num,lay,batch*nn->n_neurons_per_layer[lay+1]+neu,layers_act[lay][batch*nn->n_neurons_per_layer[lay+1]+neu]);
                    }
                }
                ++in_cnt_train;
            }
            total += BATCH_SIZE;
            loss = calc_loss(nn, loss_fun);
            // printf("batch[%d] first forward pass\n",batch_num);
         
            // printf("batch[%d] lay_act\n",batch_num);
            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
                for(int j=0;j<N_CLASSES;j++){
                    //error[batch_elem*N_CLASSES+j] = outputs[batch_elem*N_CLASSES+j] - nn->targets[j];
                    error[batch_elem*N_CLASSES+j] = outputs[batch_elem*N_CLASSES+j] - targets[batch_elem*N_CLASSES+j];
                    // printf("[ep:%d][btc:%d]outputs[%d]: %f\n",epoch,batch_num,batch_elem*N_CLASSES+j,outputs[batch_elem*N_CLASSES+j]);
                    // printf("[ep:%d][btc:%d]nn->targets[%d]: %f\n",epoch,batch_num,j,nn->targets[j]);
                    // printf("[ep:%d][btc:%d]error[%d][%d]: %f\n",epoch,batch_num,batch_elem,j,error[batch_elem*N_CLASSES+j]);
                }
            }
            // printf("batch[%d] error\n",batch_num);
            
            // for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
            //     for(int j=0;j<N_CLASSES;j++){
            //         error_input[batch_elem*N_CLASSES+j] =  error[batch_elem*N_CLASSES+j] * B[batch_elem*N_CLASSES+j] 
                     
            matrix_multiply(error, B, error_input, BATCH_SIZE, N_CLASSES, N_DIMS);

            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
                for(int j=0;j<N_DIMS;j++){
                    // printf("[ep:%d][btc:%d]error_input[%d][%d]: %f\n",epoch,batch_num,batch_elem,j,error_input[batch_elem*N_CLASSES+j]);
                }
            }

            // //matrix multiplication
            // for(int i = 0; i < BATCH_SIZE; i++) {
            //     for(int j = 0; j < N_DIMS; j++) {
            //         error_input[i*N_DIMS+j] = 0.0;
            //         for (int k = 0; k < N_CLASSES; k++) {
            //             error_input[i*N_DIMS+j] += error[i*N_CLASSES+k] * B[k*N_DIMS+j];
            //         }
            //     }
            // }
           
            for(int i = 0; i < BATCH_SIZE; i++) {
                for(int j = 0; j < N_DIMS; j++) {
                    mod_inputs[i*N_DIMS+j] = inputs[i*N_DIMS+j] + error_input[i*N_DIMS+j];
                }
            }

            // printf("[ep:%d][btc:%d]mod_inputs[1200]: %f\n",epoch,batch_num,mod_inputs[1200]);
            // printf("[ep:%d][btc:%d]inputs[1200]: %f\n",epoch,batch_num,inputs[1200]);
            // printf("[ep:%d][btc:%d]error_input[1200]: %f\n",epoch,batch_num,error_input[1200]);
            // printf("batch[%d] mod_inputs\n",batch_num);
           
           
            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
                //do masks
                max_val = (double)INT_MIN;
                idx = -1;
            
                for(int j=0;j<nn->n_neurons_per_layer[0];j++){          //££ + 0 
                    nn->actv_out[0][j] = mod_inputs[batch_elem*nn->n_neurons_per_layer[0] + j];      //assign input
                }
                
                forward_propagation(nn, activation_fun, loss_fun);
                 
                    for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){
                       
                        // printf("[%d]actv_out: %f\n", batch_elem,nn->actv_out[nn->n_layers-1][j]);
                    }
                //array to save output for every batch
                for(int out_neu=0;out_neu<N_CLASSES;out_neu++){
                    mod_outputs[batch_elem*N_CLASSES+out_neu] = nn->actv_out[nn->n_layers-1][out_neu]; 
                    // printf("2[epoch:%d][elem:%d]actv_out[%d][%d]: %f\n", epoch,batch_elem,nn->n_layers-1,out_neu, nn->actv_out[nn->n_layers-1][out_neu]);
                }
                //printf("\n");

                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
                    if(batch_elem==0){
                        // printf("[%d][%d][2]nn->actv_out[%d][%d] : %f \n",epoch,batch_elem,nn->n_layers-1,j,nn->actv_out[nn->n_layers-1][j] );
                    }
                    if(nn->actv_out[nn->n_layers-1][j] > max_val){      //trova neurone con output maggiore
                        max_val = nn->actv_out[nn->n_layers-1][j];
                        idx = j;       //££ + 0 j-1
                        
                    }
                }
                // if(batch_elem==1){
                //     printf("[%d]train idx: %d\n",batch_elem,idx);
                //     printf("[%d]label: %d\n",batch_elem,(int)y_train_temp[shuffler_train[in_cnt_train_2]]);
                // }
             

                // printf("[batch:%d][elem:%d]y_train_temp: %.2f\n", batch_num, batch_elem,y_train_temp[shuffler_train[in_cnt_train_2]]);
                // printf("index of max val computed: %d\n", idx);
                // printf("index 2: %d\n",shuffler_train[in_cnt_train_2]);
                
                if(idx == (int)y_train_temp[shuffler_train[in_cnt_train_2]]){   //checks train prediction
                    correct++;
                }
                for(int lay=0;lay<NUM_LAYERS-1;lay++){
                    for(int neu=0;neu<nn->n_neurons_per_layer[lay+1];neu++){
                        mod_layers_act[lay][batch_elem*nn->n_neurons_per_layer[lay+1]+neu] = nn->actv_out[lay+1][neu]; //?????????
                    }
                }
                ++in_cnt_train_2;
            }
            // printf("batch[%d] second forward pass\n",batch_num);
            
           
            // printf("batch[%d] mod_layers_act\n",batch_num);
            //mod activations
          
            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){
                for(int j=0;j<N_CLASSES;j++){
                    mod_error[batch_elem*N_CLASSES+j] = mod_outputs[batch_elem*N_CLASSES+j] - targets [batch_elem*N_CLASSES+j];
                    // printf("TARGETS[%d]: %F\n", j,nn->targets[j]);
                    // printf("[ep:%d][btc:%d]mod_error[%d][%d]: %f\n",epoch,batch_num,batch_elem,j,mod_error[batch_elem*N_CLASSES+j]);
                    
                }
                // printf("\n");
            }
            // printf("batch[%d] mod_error\n",batch_num);

            for(int row=0;row<N_CLASSES;row++){
                for(int col=0;col<BATCH_SIZE;col++){
                    mod_error_T[col*N_CLASSES+row] = -mod_error[row*BATCH_SIZE+col]; //neg
                }
            }
            // printf("batch[%d] mod_error_T\n",batch_num);

            for(int l=0;l<NUM_LAYERS-1;l++){    //LOOOP???????? 
                // neg_delta_lay_act_T[l] = -(layers_act[l] - mod_layers_act[l]);
                matrix_subtract(layers_act[l], mod_layers_act[l], delta_lay_act[l], BATCH_SIZE, nn->n_neurons_per_layer[l+1]);
                matrix_transpose(delta_lay_act[l], delta_lay_act_T[l], BATCH_SIZE, nn->n_neurons_per_layer[l+1]);    //neg
                
               
                if(l == NUM_LAYERS-2){
                    if((NUM_LAYERS-1) > 1){
                        matrix_multiply(mod_error_T, mod_layers_act[l-1], delta_w, N_CLASSES, BATCH_SIZE, nn->n_neurons_per_layer[l]);
                    }
                    else {
                        matrix_multiply(mod_error_T, mod_inputs, delta_w, N_CLASSES, BATCH_SIZE, N_DIMS);
                    }
                } 
                else if(l==0){
                    matrix_multiply(delta_lay_act_T[l], mod_inputs, delta_w, nn->n_neurons_per_layer[l+1], BATCH_SIZE, N_DIMS);
                } 
                else if(l>0 && l<NUM_LAYERS-2){
                    matrix_multiply(delta_lay_act_T[l], mod_layers_act[l-1], delta_w, nn->n_neurons_per_layer[l+1], BATCH_SIZE, nn->n_neurons_per_layer[l]);
                }

                delta_w_all[l] = delta_w;

            }
            // for(int i=0;i<NUM_LAYERS-1;i++){
            //     for(int j=0;j<max_num_neu;j++){
            //         for(int k=0;k<max_num_neu;k++){
            //             printf("delta_w_all[%d][%d][%d]: %f\n",i,j*max_num_neu,k,delta_w_all[i][j*max_num_neu+k]);
            //         } 
            //     }
            // }
            // printf("batch[%d] delta computed\n",batch_num);
            for(int k=0;k<nn->n_layers-1;k++){
                for(int i=0;i<nn->n_neurons_per_layer[k];i++){                    //££ + 0
                    for(int j=0;j<nn->n_neurons_per_layer[k+1];j++){    

                        if(strcmp(opt, "sgd") == 0){
                            nn->w[k][i][j] -= learning_rate * (delta_w_all[k][j*max_num_neu+i]/BATCH_SIZE);  //WHY BATCH_SIZE???
                        }
                        else if(strcmp(opt, "momentum") == 0){
                            // nn->momentum_w[k][i][j] = mom_gamma * nn->momentum_w[k][i][j] + (1.0-mom_gamma) * (delta_w_all[k][j*nn->n_neurons_per_layer[k]+i]/BATCH_SIZE) * learning_rate;
                            //nn->momentum_w[k][i][j] = mom_gamma * nn->momentum_w[k][i][j] + (1.0-mom_gamma) * (delta_w_all[k][j*nn->n_neurons_per_layer[k]+i]/BATCH_SIZE) * learning_rate;
                            nn->momentum_w[k][i][j] = mom_gamma * nn->momentum_w[k][i][j] + (delta_w_all[k][j*max_num_neu+i]/BATCH_SIZE) * learning_rate;
                            // if(k==1 && j==5 || k==1 && j==idx){
                            if(k==1 && j==7 || k==1 && j== idx){
                                // printf("[%d][%d]momentum[%d][%d][%d]: %.20f\n",epoch,batch_num,k,i,j,nn->momentum_w[k][i][j]);
                             }
                            
                            nn->w[k][i][j] -= nn->momentum_w[k][i][j];
                            //printf("[%d][%d]w[%d][%d][%d]: %f\n",epoch,batch_num,k,i,j,nn->w[k][i][j]);
                        }
                    }
                }
    
            }
            // printf("batch[%d] w updated\n",batch_num);
            
            running_loss += loss;  
            // printf("[epoch:%d][batch:%d]running_loss:%f\n",epoch,batch_num,running_loss);
            batch_count += 1;
        }

        curr_loss = running_loss / (double)batch_count;
        printf("[%d, %5d] loss: %.3f\n", epoch, batch_count, curr_loss);
        train_losses[epoch] = curr_loss;
        printf("Train correct: %.2f\n", correct);
        printf("Train total: %.2f\n", total);
        printf("Train accuracy epoch [%d]: %.4f %%\n", epoch, 100 * correct / total);
        train_accs[epoch] = 100 * correct / total;

        printf("TESTING...\n");

        correct = 0.0;
        total = 0.0;
        batch_count = 0;

        int shuffler_test[N_TEST_SAMPLES];
        for(int i=0;i<N_TEST_SAMPLES;i++){
            shuffler_test[i] = i;
        }

        shuffle(shuffler_test, N_TEST_SAMPLES);

        int in_cnt_test = 0;

        for(int batch_num=0;batch_num<floor(N_TEST_SAMPLES/BATCH_SIZE);batch_num++){
        
            // printf("[%d]TEST BATCH %d\n", epoch,batch_num);

            for(int batch_elem=0;batch_elem<BATCH_SIZE;batch_elem++){

                max_val = (double)INT_MIN;
                idx = -1;
            
            
                for(int j=0;j<nn->n_neurons_per_layer[0];j++){          //££ + 0 
                    // nn->actv_out[0][j] = X_test[shuffler_test[j+batch_count*BATCH_SIZE]][j];      //assign input ??????i?????
                    nn->actv_out[0][j] = X_test[shuffler_test[in_cnt_test]][j]; 
                }
                
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
                    nn->targets[j] = y_test[shuffler_test[in_cnt_test]][j];        //assign target labels (one hot)
                }
                
                forward_propagation(nn, activation_fun, loss_fun);
            
                for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
                    if(nn->actv_out[nn->n_layers-1][j] > max_val){
                        max_val =nn->actv_out[nn->n_layers-1][j];
                        idx = j;        //££ + 0 j-1
                        // printf("test idx: %d\n", idx);
                    }
                }
                
                
                if(idx == (int)y_test_temp[shuffler_test[in_cnt_test]]){
                    correct++;
                }
                ++in_cnt_test;
            }
            total += BATCH_SIZE;
            // printf("batch[%d] test finished\n",batch_num);
            batch_count += 1;
        }
        printf("Test accuracy epoch [%d]: %f %%\n",epoch, 100 * correct / total);
        test_accs[epoch] = 100 * correct / total;

        
        
    }
    
    printf("FINISHED TRAINING\n");

    FILE *file = fopen("PEPITA_C_implem.txt", "w");
    printf("open file\n");

    for(int epoch=0;epoch<EPOCHS;epoch++){
        fprintf(file, "EPOCH %d\n", epoch);
        fprintf(file, "Train loss epoch [%d]: %lf\n", epoch, train_losses[epoch]);
        fprintf(file, "Train accuracy epoch [%d]: %lf\n", epoch, train_accs[epoch]);
        fprintf(file, "Test accuracy epoch [%d]: %lf\n", epoch, test_accs[epoch]);
    }
    
    fclose(file);
    printf("close file\n");

    printf("Freeing memory...\n");
    // free(test_accs);
    // free(train_losses);
    // free(train_accs);
    // free(B);
    free(outputs);
    free(targets);
    free(inputs);
    for(int i=0; i<NUM_LAYERS-1;i++){
        free(layers_act[i]);
    }
    free(layers_act);
    free(error);
    free(error_input);
    free(mod_inputs);
    free(mod_outputs);
    for(int i=0; i<NUM_LAYERS-1;i++){
        free(mod_layers_act[i]);
    }
    free(mod_layers_act);
    free(mod_error);
   
    // for(int l=0; l<NUM_LAYERS-1;l++){
    //     free(delta_w_all[l]);
    // }
    free(delta_w_all);
    free(delta_w);
    free(mod_error_T);
    for(int i=0; i<NUM_LAYERS-1;i++){
        free(delta_lay_act[i]);
    }
    free(delta_lay_act);
    for(int i=0; i<NUM_LAYERS-1;i++){
        free(delta_lay_act_T[i]);
    }
    free(delta_lay_act_T);  
}
 
                 



// // Function to test the model
// double* model_test(struct NeuralNet* nn, double** X_test, double** y_test, double* y_test_temp, char* activation_fun, char* loss){
//     printf("in model test\n");
//     int correct = 0;
//     double running_loss = 0.0;
//     double curr_loss = 0.0;
//     for(int i=0;i<BATCH_SIZE;i++){
//         int idx = -1;
//         double max_val = (double)INT_MIN;
//         for(int j=0;j<nn->n_neurons_per_layer[0];j++){        //££ + 0
//             nn->actv_out[0][j] = X_test[i][j];
//         }
//         for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
//             nn->targets[j] = y_test[i][j];
//         }
//         forward_propagation(nn, activation_fun, loss);
//         running_loss += calc_loss(nn, loss);
            
//         for(int j=0;j<nn->n_neurons_per_layer[nn->n_layers-1];j++){       //££ + 0
//             if(nn->actv_out[nn->n_layers-1][j] > max_val){
//                 max_val =nn->actv_out[nn->n_layers-1][j];
//                 idx = j;        //££ + 0 j-1
//             }
//         }
//         if(idx == (int)y_test_temp[i]){
//             correct++;
//         }
//     }
//     curr_loss = running_loss / (double)BATCH_SIZE;
//     //running_loss /= (double)N_TEST_SAMPLES;
//     double accuracy = (double)correct*100/(double)BATCH_SIZE;
//     static double metrics[2];
//     metrics[0] = curr_loss;
//     metrics[1] = accuracy;
//     return metrics;
// }


int main(){

    // Used for setting a random seed
    srand(time(NULL));


    // Initialize neural network architecture parameters
    // int n_layers = 3;
    // int n_neurons_per_layer[] = {784, 128, 10};

    // Create and initialize the neural network
    struct NeuralNet* nn = newNet();
    //init_nn(nn);
 
    // Initialize the learning rate, optimizer, loss, and other hyper-parameters
    double learning_rate = 0.001;
    double init_lr = 1e-4;
    char* activation_fun = "relu";
    char* loss = "ce";
    char* opt = "momentum";
   
    double** img_train;
    double** lbl_train;
    double* lbl_train_temp;
    double** img_test;
    double** lbl_test;
    double* lbl_test_temp;
    // double** batch_train_data; 
    // double** batch_train_labels;
    // double** batch_test_data; 
    // double** batch_test_labels;
    

    // double* train_losses = (double*)malloc(epochs*sizeof(double));
    // double* train_accuraces = (double*)malloc(epochs*sizeof(double));
    // double* test_losses = (double*)malloc(epochs*sizeof(double));
    // double* test_accuraces = (double*)malloc(epochs*sizeof(double));


    img_train = (double**) malloc(N_SAMPLES*sizeof(double*));
    for(int i=0;i<N_SAMPLES;i++){
        img_train[i] = (double*)malloc(N_DIMS*sizeof(double));
    }
    lbl_train = malloc(N_SAMPLES * sizeof(double*));
    for(int i=0;i<N_SAMPLES;i++){
        lbl_train[i] = malloc(N_CLASSES * sizeof(double));
    }
    lbl_train_temp = malloc(N_SAMPLES*sizeof(double));
    read_csv_file(img_train, lbl_train_temp, lbl_train, "train");

    // for(int j=0;j<N_SAMPLES;j++){
    //     for(int i=0;i<N_DIMS;i++){
    //         printf("X_train before scaling: img[%d][%d]: %f\n",j,i,img_train[j][i]);
    //     }
    // }
    scale_data(img_train, "train");
    // for(int j=0;j<N_SAMPLES;j++){
    //     for(int i=0;i<N_DIMS;i++){
    //         printf("X_train after scaling: img[%d][%d]: %f\n",j,i,img_train[j][i]);
    //     }
    // }

    img_test = malloc(N_TEST_SAMPLES*sizeof(double*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        img_test[i] = malloc(N_DIMS*sizeof(double));
    }
    lbl_test = malloc(N_TEST_SAMPLES * sizeof(double*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        lbl_test[i] = malloc(N_CLASSES * sizeof(double));
    }
    lbl_test_temp = malloc(N_TEST_SAMPLES*sizeof(double));
    read_csv_file(img_test, lbl_test_temp, lbl_test, "test");
    printf("heading to scale_data\n");
    scale_data(img_test, "test");
    //normalize_data(img_train, img_test);

    //   for(int j=0;j<N_SAMPLES;j++){
    //     for(int i=0;i<N_DIMS;i++){
    //         printf("X_train after normalizing: img[%d][%d]: %f\n",j,i,img_train[j][i]);
    //     }
    // }

    // for(int i=0; i<N_SAMPLES;i++){
    //     printf("label_num: %f\n", lbl_train_temp[i]);
    // }
    

    model_train(nn,img_train,lbl_train,lbl_train_temp,img_test,lbl_test,lbl_test_temp,activation_fun,loss,opt,learning_rate);

    // batch_train_data = (double**) malloc(BATCH_SIZE*sizeof(double*));
    // for(int i=0;i<BATCH_SIZE;i++){
    //     batch_train_data[i] = (double*)malloc(N_DIMS*sizeof(double));
    // }
    // //printf("batch_train_data allocated\n");
    // batch_train_labels = malloc(BATCH_SIZE * sizeof(double*));
    // for(int i=0;i<BATCH_SIZE;i++){
    //     batch_train_labels[i] = malloc(N_CLASSES * sizeof(double));
    // }
    // //printf("batch_train_lbl allocated\n");
    // batch_test_data = (double**) malloc(BATCH_SIZE*sizeof(double*));
    // for(int i=0;i<BATCH_SIZE;i++){
    //     batch_test_data[i] = (double*)malloc(N_DIMS*sizeof(double));
    // }   
    // // printf("batch_test_data allocated\n");
    // batch_test_labels = malloc(BATCH_SIZE * sizeof(double*));
    // for(int i=0;i<BATCH_SIZE;i++){
    //     batch_test_labels[i] = malloc(N_CLASSES * sizeof(double));
    // }
    // // printf("batch_test_lbl allocated\n");
  
    

    // Initialize file to store metrics info for each epoch
    // FILE* file = fopen("BP_C_implementation.txt", "w");
    // printf("file opened\n");
    // fprintf(file, "train_loss,train_acc,test_loss,test_acc\n");
    // double curr_train_loss;
    // double curr_train_acc;
    // double curr_test_loss;
    // double curr_test_acc;

  

    // // Train the model for given number of epoch and test it after every epoch
    // for(int itr=0;itr<epochs;itr++){
    //     printf("in epoch for\n");
       
    //     int batch_count = 0;
    //     for(int btr=0;btr<floor(N_SAMPLES/BATCH_SIZE);btr++){
    //         printf("in batch for\n");
    //         // fetch_batch(batch_train_data, batch_train_labels, BATCH_SIZE, batch_index, img_train, lbl_train);

    //         // double* train_metrics = model_train(nn, batch_train_data, batch_train_labels, lbl_train_temp, activation_fun, loss, opt, learning_rate, num_samples_to_train, itr+1);
           
           

    //         // curr_train_loss = train_metrics[0] / batch_count;
    //         // curr_train_acc = train_metrics[1] / batch_count; //????????
    //         // train_losses[itr+1] = curr_train_loss;
    //         // train_accuraces[itr+1] = curr_train_acc;

    //         fprintf(file, "%lf,", train_losses[itr+1]);
    //         fprintf(file, "%lf,", train_accuraces[itr+1]);

    //         // printf("TRAIN...\n");
    //         // printf("Epoch, batch count: [%d, %5d] -> ", itr+1,batch_count);
    //         // printf("Train loss: %lf, ", train_losses[itr+1]);
    //         // printf("Train Accuracy: %lf, ", train_accuraces[itr+1]);
    
    //         // learning_rate = init_lr * exp(-0.1 * (itr+1));
    //         // batch_index += BATCH_SIZE;
    //     }
    //     for(int btr=0;btr<floor(N_TEST_SAMPLES/BATCH_SIZE);btr++){
    //         // printf("in batch test for\n");
    //         // fetch_batch(batch_test_data, batch_test_labels, BATCH_SIZE, batch_index, img_test, lbl_test);

        

    //         // double* test_metrics = model_test(nn, img_test, lbl_test, lbl_test_temp, activation_fun, loss);
    //         // double test_loss = test_metrics[0];
    //         // double test_acc = test_metrics[1];

    //         // curr_test_loss = test_metrics[0] / batch_count;
    //         // curr_test_acc = test_metrics[1] / batch_count; //????????
    //         // test_losses[itr+1] = curr_test_loss;
    //         // test_accuraces[itr+1] = curr_test_acc;

    //         fprintf(file, "%lf,", test_losses[itr+1]);
    //         fprintf(file, "%lf\n", test_accuraces[itr+1]);

    //         // printf("TEST...\n");
    //         // printf("Epoch, batch count: [%d, %5d] -> ", itr+1,batch_count);
    //         // printf("Test loss: %lf, ", test_losses[itr+1]);
    //         // printf("Test Accuracy: %lf\n", test_accuraces[itr+1]);

    //         // batch_index += BATCH_SIZE;
    //     }
    // }

    // // Close the file
    // fclose(file);

    // Free the dynamically allocated memory
    free_NN(nn);
    // free(train_losses);
    // free(train_accuraces);
    // free(test_losses);
    // free(test_accuraces);
    
   
    for(int i=0;i<N_SAMPLES;i++){
        free(img_train[i]);
        free(lbl_train[i]);
    }
    free(img_train);
    free(lbl_train);
    free(lbl_train_temp);
    for(int i=0;i<N_TEST_SAMPLES;i++){
        free(img_test[i]);
        free(lbl_test[i]);
    }
    free(img_test);
    free(lbl_test);
    free(lbl_test_temp);
  
    return 0;
}