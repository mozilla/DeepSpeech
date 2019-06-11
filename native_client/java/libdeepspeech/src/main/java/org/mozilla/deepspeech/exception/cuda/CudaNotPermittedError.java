package org.mozilla.deepspeech.exception.cuda;

import org.jetbrains.annotations.NotNull;

/**
 * Exception representing the error state of {@link org.mozilla.deepspeech.cuda.Cuda.CudaError#CUDA_ERROR_NOT_PERMITTED}
 */
public class CudaNotPermittedError extends CudaException {

    /**
     * @param function the prototype of the function that returned the error
     */
    public CudaNotPermittedError(@NotNull String function) {
        super("Cuda function \"" + function + "\" returned CUDA_ERROR_NOT_PERMITTED! Attempted operation is not permitted!");
    }

}
