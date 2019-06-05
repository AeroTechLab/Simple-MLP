#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "multi_layer_perceptron.h"

typedef struct { double min, max; } Limits;

struct _MLPerceptronData
{
  double* inputsList;						// x'
  double* hiddenOutputsList;				// y'
  double** inputWeightsTable;				// Wji
  double** outputWeightsTable;				// Wkj
  Limits* inputLimitsList;
  Limits* outputLimitsList;
  size_t inputsNumber, outputsNumber;
  size_t hiddenNeuronsNumber;
};


MLPerceptron MLPerceptron_InitNetwork( size_t inputsNumber, size_t outputsNumber, size_t hiddenNeuronsNumber )
{
  MLPerceptronData* newNetwork = (MLPerceptronData*) malloc( sizeof(MLPerceptronData) );
  
  // Allocate memory for network inputs and hidden neurons outputs (extra memory for -1 threshold/offset)
  newNetwork->inputsList = (double*) calloc( inputsNumber + 1, sizeof(double) );
  newNetwork->hiddenOutputsList = (double*) calloc( hiddenNeuronsNumber + 1, sizeof(double) );
  // Allocate memory for input weights (extra memory for threshold/offset "weight" applied to -1 input)
  newNetwork->inputWeightsTable = (double**) calloc( hiddenNeuronsNumber, sizeof(double*) );
  for( size_t neuronIndex = 0; neuronIndex < hiddenNeuronsNumber; neuronIndex++ )
    newNetwork->inputWeightsTable[ neuronIndex ] = (double*) calloc( inputsNumber + 1, sizeof(double) );
  // Allocate memory for output weights (extra memory for threshold/offset "weight" applied to -1 hidden "output")
  newNetwork->outputWeightsTable = (double**) calloc( outputsNumber, sizeof(double*) );
  for( size_t neuronIndex = 0; neuronIndex < outputsNumber; neuronIndex++ )
    newNetwork->outputWeightsTable[ neuronIndex ] = (double*) calloc( hiddenNeuronsNumber + 1, sizeof(double) );
  // Allocate memory for network inputs/outputs maximum and minimum values
  newNetwork->inputLimitsList = (Limits*) calloc( inputsNumber, sizeof(Limits) );
  newNetwork->outputLimitsList = (Limits*) calloc( outputsNumber, sizeof(Limits) );
  
  newNetwork->inputsNumber = inputsNumber;
  newNetwork->outputsNumber = outputsNumber;
  newNetwork->hiddenNeuronsNumber = hiddenNeuronsNumber;
  
  return (MLPerceptron) newNetwork;
}

void MLPerceptron_EndNetwork( MLPerceptron network )
{
  if( network == NULL ) return;
  
  free( network->hiddenOutputsList );
  
  for( size_t neuronIndex = 0; neuronIndex < network->hiddenNeuronsNumber; neuronIndex++ )
    free( network->inputWeightsTable[ neuronIndex ] );
  free( network->inputWeightsTable );
  
  for( size_t neuronIndex = 0; neuronIndex < network->hiddenNeuronsNumber; neuronIndex++ )
    free( network->outputWeightsTable[ neuronIndex ] );
  free( network->outputWeightsTable );
  
  free( network->inputLimitsList );
  free( network->outputLimitsList );
  
  free( network );
}

static void GetInputLimits( const double** inputSamplesTable, size_t samplesNumber, size_t inputsNumber, Limits* inputLimitsList )
{
  memset( inputLimitsList, 0, inputsNumber * sizeof(Limits) );
  // Find min/max input values
  for( size_t sampleIndex = 0; sampleIndex < samplesNumber; sampleIndex++ )
  {
    const double* inputSamplesList = inputSamplesTable[ sampleIndex ];
    for( size_t inputIndex = 0; inputIndex < inputsNumber; inputIndex++ )
    {
      if( inputSamplesList[ inputIndex ] < inputLimitsList[ inputIndex ].min ) inputLimitsList[ inputIndex ].min = inputSamplesList[ inputIndex ];
      if( inputSamplesList[ inputIndex ] > inputLimitsList[ inputIndex ].max ) inputLimitsList[ inputIndex ].max = inputSamplesList[ inputIndex ];
    }
  }
}

static void NormalizeValues( double* valuesList, size_t valuesNumber, Limits* valueLimitsList )
{
  for( size_t valueIndex = 0; valueIndex < valuesNumber; valueIndex++ )
  {
    double valueRange = valueLimitsList[ valueIndex ].max - valueLimitsList[ valueIndex ].min;
    valuesList[ valueIndex ] = ( valuesList[ valueIndex ] - valueLimitsList[ valueIndex ].min ) / valueRange;
  }
}

static void GetOutputLimits( const double** outputSamplesTable, size_t samplesNumber, size_t outputsNumber, Limits* outputLimitsList )
{
  memset( outputLimitsList, 0, outputsNumber * sizeof(Limits) );
  // Find min/max output values
  for( size_t sampleIndex = 0; sampleIndex < samplesNumber; sampleIndex++ )
  {
    const double* outputSamplesList = outputSamplesTable[ sampleIndex ];
    for( size_t outputIndex = 0; outputIndex < outputsNumber; outputIndex++ )
    {
      if( outputSamplesList[ outputIndex ] < outputLimitsList[ outputIndex ].min ) outputLimitsList[ outputIndex ].min = outputSamplesList[ outputIndex ];
      if( outputSamplesList[ outputIndex ] > outputLimitsList[ outputIndex ].max ) outputLimitsList[ outputIndex ].max = outputSamplesList[ outputIndex ];
    }
  }
}

static void DenormalizeValues( double* valuesList, size_t valuesNumber, Limits* valueLimitsList )
{
  for( size_t valueIndex = 0; valueIndex < valuesNumber; valueIndex++ )
  {
    double valueRange = valueLimitsList[ valueIndex ].max - valueLimitsList[ valueIndex ].min;
    valuesList[ valueIndex ] = valuesList[ valueIndex ] * valueRange + valueLimitsList[ valueIndex ].min;
  }
}

double MLPerceptron_Train( MLPerceptron network, const double** inputSamplesTable, const double** outputSamplesTable, size_t samplesNumber )
{
  const size_t MAX_EPOCHS_NUMBER = 100;
  const double TOLERANCE = 1e-6;
  const double LEARNING_RATE = 0.1, MAX_MOMENTUM = 0.4;
  
  double* hiddenActivationsList = (double*) calloc( network->hiddenNeuronsNumber, sizeof(double) );		// Hidden layer internal neuron activations
  double* outputActivationsList = (double*) calloc( network->outputsNumber, sizeof(double) );			// Output neurons activations
  double* outputGradientsList = (double*) calloc( network->outputsNumber, sizeof(double) );				// Output neurons output gradients
  double* fittingErrorsList = (double*) calloc( network->outputsNumber, sizeof(double) );				// Deviation between current epochCount outputs and reference samples
  double momentum = 0.0;
  double averageSquareError = 0.0;																				// Average epoch deviation from samples (square error)
  
  //srand(time(NULL));
  
  // Random initialization of hidden layer and output layer weights
  for( size_t neuronIndex = 0; neuronIndex < network->hiddenNeuronsNumber; neuronIndex++ )
  {
    for( size_t inputIndex = 0; inputIndex < network->inputsNumber; inputIndex++ )
      network->inputWeightsTable[ neuronIndex ][ inputIndex ] = (rand() % 1000) / 1000.0;
  }
  for( size_t neuronIndex = 0; neuronIndex < network->hiddenNeuronsNumber; neuronIndex++ )
  {
    for( size_t outputIndex = 0; outputIndex < network->outputsNumber; outputIndex++ )
      network->outputWeightsTable[ neuronIndex ][ outputIndex ] = (rand() % 1000) / 1000.0;
  }
  
  GetInputLimits( inputSamplesTable, samplesNumber, network->inputsNumber, network->inputLimitsList );
  GetOutputLimits( outputSamplesTable, samplesNumber, network->outputsNumber, network->outputLimitsList );
  
  // Adjust weights (train) the network until error is below tolerance or maximum number of epochs is reached
  for( size_t epochCount = 0; epochCount < MAX_EPOCHS_NUMBER; epochCount++ ) 
  {    
    momentum = 0.9 * momentum + 0.1 * MAX_MOMENTUM;
    // Adjust weights and sum resulting square error for all samples (reference input/output sets)
    for( size_t sampleIndex = 0; sampleIndex < samplesNumber; sampleIndex++ )
    {         
      const double* inputSamplesList = inputSamplesTable[ sampleIndex ];
      const double* outputSamplesList = outputSamplesTable[ sampleIndex ];
      // Normalize and add -1 input at the end of inputs list
      memcpy( network->inputsList, inputSamplesList, network->inputsNumber * sizeof(double) );
      NormalizeValues( network->inputsList, network->inputsNumber, network->inputLimitsList );
      network->inputsList[ network->inputsNumber ] = -1.0;
      // First/hidden layer: calculate hidden neurons internal activations and outputs
      for( size_t neuronIndex = 0; neuronIndex < network->hiddenNeuronsNumber; neuronIndex++ ) 
      {
        // Hidden neuron internal activation: u = Wji * x - theta1 = Wji' * x'
        hiddenActivationsList[ neuronIndex ] = 0.0;
        for( size_t inputIndex = 0; inputIndex < network->inputsNumber + 1; inputIndex++ )
          hiddenActivationsList[ neuronIndex ] += network->inputWeightsTable[ neuronIndex ][ inputIndex ] * network->inputsList[ inputIndex ];
        // Hidden layer/neuron output (activation function): y' = sigmoid(u)
        network->hiddenOutputsList[ neuronIndex ] = 1.0 / ( 1.0 + exp( -hiddenActivationsList[ neuronIndex ] ) );
      }
      network->hiddenOutputsList[ network->hiddenNeuronsNumber ] = -1.0;	// Add threshold input
      // Normalize output samples list
      memcpy( fittingErrorsList, outputSamplesList, network->outputsNumber * sizeof(double) );
      NormalizeValues( fittingErrorsList, network->outputsNumber, network->outputLimitsList );
      // Second/output layer: calculate network outputs
      for( size_t outputIndex = 0; outputIndex < network->outputsNumber; outputIndex++ )
      {
        // Output neuron internal activation: u' = Wkj * y' - theta2 = Wkj' * y'
        outputActivationsList[ outputIndex ] = 0.0;
        for( size_t neuronIndex = 0; neuronIndex < network->hiddenNeuronsNumber + 1; neuronIndex++ )
          outputActivationsList[ outputIndex ] += network->outputWeightsTable[ neuronIndex ][ outputIndex ] * network->hiddenOutputsList[ neuronIndex ];
        // Exit/output neuron output (activation function): y = sigmoid(u')
        double output = 1.0 / ( 1.0 + exp( -outputActivationsList[ outputIndex ] ) );
        
        // Current/epoch network output error and square error sum
        fittingErrorsList[ outputIndex ] = fittingErrorsList[ outputIndex ] - output;
        averageSquareError += pow( fittingErrorsList[ outputIndex ], 2 );
      }
      // Backpropagation traning: output layer
      for( size_t outputIndex = 0; outputIndex < network->outputsNumber; outputIndex++ )
      {
        double outputDerivative = exp( -outputActivationsList[ outputIndex ] ) / pow( 1.0 + exp( -outputActivationsList[ outputIndex ] ), 2 );  	// sigmoid function derivative at output activation 
        outputGradientsList[ outputIndex ] = fittingErrorsList[ outputIndex ] * outputDerivative;				// output weight gradient: grad = output_error * output_derivative
        // output_weight = hidden_output_weight + LEARNING_RATE * grad * hidden_output
        for( size_t neuronIndex = 0; neuronIndex < network->hiddenNeuronsNumber + 1; neuronIndex++ )
          network->outputWeightsTable[ neuronIndex ][ outputIndex ] += ( 1 + momentum ) * LEARNING_RATE * outputGradientsList[ outputIndex ] * network->hiddenOutputsList[ neuronIndex ];
      }      
      // Backpropagation traning: hidden layer
      for( size_t neuronIndex = 0; neuronIndex < network->hiddenNeuronsNumber; neuronIndex++ ) 
      {
        // Local gradient calculation: grad = sum( output_delta * output_weight ) * hidden_output_derivative
        double hiddenOutputGradient = 0.0;
        for( size_t outputIndex = 0; outputIndex < network->outputsNumber; outputIndex++ )
          hiddenOutputGradient += outputGradientsList[ outputIndex ] * network->outputWeightsTable[ network->hiddenNeuronsNumber ][ outputIndex ];
        hiddenOutputGradient *= exp( -hiddenActivationsList[ neuronIndex ] ) / pow( 1.0 + exp( -hiddenActivationsList[ neuronIndex ] ), 2 );		// sigmoid function derivative at hidden activation
        // weight = weight + LEARNING_RATE * grad * input
        for( size_t inputIndex = 0; inputIndex < network->inputsNumber + 1; inputIndex++ )
          network->inputWeightsTable[ neuronIndex ][ inputIndex ] += ( 1 + momentum ) * LEARNING_RATE * hiddenOutputGradient * network->inputsList[ inputIndex ];
      }
    }
    
    averageSquareError = averageSquareError / samplesNumber;
    
    if( averageSquareError < TOLERANCE ) break;
  }
  
  free( hiddenActivationsList );
  free( outputActivationsList );
  free( outputGradientsList );
  free( fittingErrorsList );
  
  return averageSquareError;
}

double MLPerceptron_Validate( MLPerceptron network, const double** inputSamplesTable, const double** outputSamplesTable, size_t samplesNumber )
{
  double averageSquareError = 0.0;										// Average epoch deviation from samples (square error)
  
  double* outputsList = (double*) calloc( network->outputsNumber, sizeof(double) );
  double* outputRefsList = (double*) calloc( network->outputsNumber, sizeof(double) );
  
  for( size_t sampleIndex = 0; sampleIndex < samplesNumber; sampleIndex++ )
  {
    const double* inputSamplesList = inputSamplesTable[ sampleIndex ];
    const double* outputSamplesList = outputSamplesTable[ sampleIndex ];
    
    MLPerceptron_ProcessInput( network, inputSamplesList, outputsList );
    NormalizeValues( outputsList, network->outputsNumber, network->outputLimitsList );
    
    memcpy( outputRefsList, outputSamplesList, network->outputsNumber * sizeof(double) );
    NormalizeValues( outputRefsList, network->outputsNumber, network->outputLimitsList );
    
    for( size_t outputIndex = 0; outputIndex < network->outputsNumber; outputIndex++ )
      averageSquareError += pow( outputRefsList[ outputIndex ] - outputsList[ outputIndex ], 2 );
  }
  
  averageSquareError = averageSquareError / samplesNumber;
  
  free( outputsList );
  free( outputRefsList );
        
  return averageSquareError;
}

void MLPerceptron_ProcessInput( MLPerceptron network, const double* inputsList, double* outputsList )
{
  if( network == NULL ) return;
  
  // Normalize and add -1 input at the end of inputs list
  memcpy( network->inputsList, inputsList, network->inputsNumber * sizeof(double) );
  NormalizeValues( network->inputsList, network->inputsNumber, network->inputLimitsList );
  network->inputsList[ network->inputsNumber ] = -1.0;
  
  // First/hidden layer: calculate hidden neurons internal activations and outputs
  for( size_t neuronIndex = 0; neuronIndex < network->hiddenNeuronsNumber; neuronIndex++ ) 
  {
    // Hidden neuron internal activation: u = Wji * x - theta1 = Wji' * x'
    network->hiddenOutputsList[ neuronIndex ] = 0.0;
    for( size_t inputIndex = 0; inputIndex < network->inputsNumber + 1; inputIndex++ )
      network->hiddenOutputsList[ neuronIndex ] += network->inputWeightsTable[ neuronIndex ][ inputIndex ] * network->inputsList[ inputIndex ];
    // Hidden layer/neuron output (activation function): y' = sigmoid(u)
    network->hiddenOutputsList[ neuronIndex ] = 1.0 / ( 1.0 + exp( -network->hiddenOutputsList[ neuronIndex ] ) );
  }
  network->hiddenOutputsList[ network->hiddenNeuronsNumber ] = -1.0;	// Add threshold input
  
  // Second/output layer: calculate network outputs
  for( size_t outputIndex = 0; outputIndex < network->outputsNumber; outputIndex++ )
  {
    // Output neuron internal activation: u' = Wkj * y' - theta2 = Wkj' * y'
    outputsList[ outputIndex ] = 0.0;
    for( size_t neuronIndex = 0; neuronIndex < network->hiddenNeuronsNumber + 1; neuronIndex++ )
      outputsList[ outputIndex ] += network->outputWeightsTable[ neuronIndex ][ outputIndex ] * network->hiddenOutputsList[ neuronIndex ];
    // Exit/output neuron output (activation function): y = sigmoid(u')
    outputsList[ outputIndex ] = 1.0 / ( 1.0 + exp( -outputsList[ outputIndex ] ) );
  }
  
  DenormalizeValues( outputsList, network->outputsNumber, network->outputLimitsList );
} 
