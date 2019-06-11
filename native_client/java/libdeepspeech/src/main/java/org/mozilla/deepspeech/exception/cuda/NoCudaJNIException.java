package org.mozilla.deepspeech.exception.cuda;

/**
 * Exception representing the error state of the DeepSpeech JNI library not being compiled to use cuda
 */
public class NoCudaJNIException extends CudaException {

    public NoCudaJNIException() {
        super("DeepSpeech JNI library is either not loaded or not built to use cuda!");
    }

}
