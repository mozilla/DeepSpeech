package org.mozilla.deepspeech.exception.cuda;

import org.jetbrains.annotations.NotNull;

/**
 * Exception representing the error state of {@link org.mozilla.deepspeech.cuda.Cuda.CudaError#CUDA_ERROR_INVALID_VALUE}
 */
public class CudaInvalidValueError extends CudaException {

    /**
     * @param function the prototype of the cuda function that returned the error
     */
    public CudaInvalidValueError(@NotNull String function) {
        super("Invalid value error returned by cuda function \"" + function + "\"");
    }

}
