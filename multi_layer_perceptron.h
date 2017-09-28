#ifndef MULTI_LAYER_PERCEPTRONS_H
#define MULTI_LAYER_PERCEPTRONS_H

typedef struct _MLPerceptronData MLPerceptronData;
typedef MLPerceptronData* MLPerceptron;

MLPerceptron MLPerceptron_InitNetwork( size_t, size_t, size_t );
void MLPerceptron_EndNetwork( MLPerceptron );
double MLPerceptron_Train( MLPerceptron, double**, double**, size_t );
void MLPerceptron_ProcessInput( MLPerceptron, double*, double* );

#endif // MULTI_LAYER_PERCEPTRONS_H