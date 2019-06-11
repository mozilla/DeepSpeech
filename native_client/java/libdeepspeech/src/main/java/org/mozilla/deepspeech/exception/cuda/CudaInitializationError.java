package org.mozilla.deepspeech.exception.cuda;

/**
 * Exception representing the error state of {@link org.mozilla.deepspeech.cuda.Cuda.CudaError#CUDA_ERROR_INITIALIZATION_ERROR}
 */
public class CudaInitializationError extends CudaException {

    public CudaInitializationError() {
        super("CUDA driver and runtime could not be initialized!");
    }

}
