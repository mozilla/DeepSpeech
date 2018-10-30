#ifndef OUTPUT_H_
#define OUTPUT_H_

/* Struct for the beam search output, containing the tokens based on the vocabulary indices, and the timesteps
 * for each token in the beam search output
 */
struct Output {
    std::vector<int> tokens, timesteps;
};

#endif  // OUTPUT_H_
