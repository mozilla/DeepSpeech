package org.mozilla.deepspeech.exception.cuda;

import org.jetbrains.annotations.NotNull;

public class UnknownCudaError extends CudaException {

    /**
     * @param enumError the unknown error represented by the integer enum value as defined by the native CUDA api.
     * @param function  the prototype of the cuda function that returned the invalid error
     */
    public UnknownCudaError(int enumError, @NotNull String function) {
        super("Unknown cuda error \"" + enumError + "\" returned by cuda function: \"" + function + "\"");
    }
}
