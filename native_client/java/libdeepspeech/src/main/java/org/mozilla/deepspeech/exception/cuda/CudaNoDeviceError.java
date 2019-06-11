package org.mozilla.deepspeech.exception.cuda;

/**
 * Exception representing the error state of {@link org.mozilla.deepspeech.cuda.Cuda.CudaError#CUDA_ERROR_NO_DEVICE}
 */
public class CudaNoDeviceError extends CudaException {

    public CudaNoDeviceError() {
        super("No CUDA-capable devices were detected by the installed CUDA driver!");
    }

}
