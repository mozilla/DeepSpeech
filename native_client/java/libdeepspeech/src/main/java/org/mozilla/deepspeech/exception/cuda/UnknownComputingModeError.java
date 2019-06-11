package org.mozilla.deepspeech.exception.cuda;

/**
 * Exception thrown an unknown computing mode enum is retrieved
 */
public class UnknownComputingModeError extends CudaException {

    /**
     * @param unknownEnum the unknown enum value
     */
    public UnknownComputingModeError(int unknownEnum) {
        super("An unknown computing mode enum value was retrieved: " + unknownEnum);
    }
}
