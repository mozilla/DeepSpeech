#ifndef OUTPUT_H_
#define OUTPUT_H_

#include <vector>

/* Struct for the beam search output, containing the tokens based on the vocabulary indices, and the timesteps
 * for each token in the beam search output
 */
struct Output {
    double confidence;
    std::vector<unsigned int> tokens;
    std::vector<unsigned int> timesteps;
};

#endif  // OUTPUT_H_
