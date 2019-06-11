package org.mozilla.deepspeech.exception.cuda;

/**
 * Exception representing the error state of {@link org.mozilla.deepspeech.cuda.Cuda.CudaError#INVALID_DEVICE}
 */
public class CudaInvalidDeviceError extends CudaException {

    /**
     * @param invalidDevice the invalid device
     */
    public CudaInvalidDeviceError(int invalidDevice) {
        super("The device ordinal \"" + invalidDevice + "\" does not correspond to a valid CUDA device!");
    }

}
