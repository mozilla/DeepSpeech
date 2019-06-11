package org.mozilla.deepspeech.exception.cuda;

/**
 * Exception representing the error state of {@link org.mozilla.deepspeech.cuda.Cuda.CudaError#CUDA_ERROR_INSUFFICIENT_DRIVER}
 */
public class CudaInsufficientDriverError extends CudaException {

    public CudaInsufficientDriverError() {
        super("The installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Please install an updated NVIDIA display driver to allow the application to run!");
    }

}
