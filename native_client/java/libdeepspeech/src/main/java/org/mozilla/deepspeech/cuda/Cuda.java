package org.mozilla.deepspeech.cuda;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.mozilla.deepspeech.cuda.device.Device;
import org.mozilla.deepspeech.doc.CallByReference;
import org.mozilla.deepspeech.doc.Calls;
import org.mozilla.deepspeech.doc.NativeType;
import org.mozilla.deepspeech.exception.cuda.CudaInitializationError;
import org.mozilla.deepspeech.exception.cuda.CudaInsufficientDriverError;
import org.mozilla.deepspeech.exception.cuda.CudaInvalidDeviceError;
import org.mozilla.deepspeech.exception.cuda.CudaInvalidValueError;
import org.mozilla.deepspeech.exception.cuda.CudaNoDeviceError;
import org.mozilla.deepspeech.exception.cuda.CudaNotPermittedError;
import org.mozilla.deepspeech.exception.cuda.NoCudaJNIException;
import org.mozilla.deepspeech.exception.cuda.UnknownComputingModeError;
import org.mozilla.deepspeech.exception.cuda.UnknownCudaError;
import org.mozilla.deepspeech.utils.NativeAccess;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.mozilla.deepspeech.utils.BufferUtils.getBufferAddress;

/**
 * Provides access to a few CUDA functions to ensure that the CUDA runtime works properly
 */
public class Cuda {

    /**
     * CUDA Error types
     */
    public enum CudaError {

        /**
         * The API call returned with no errors. In the case of query calls, this also means that the operation being queried is complete.
         */
        SUCCESS(0),

        /**
         * This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
         */
        INVALID_VALUE(1),

        /**
         * The API call failed because the CUDA driver and runtime could not be initialized.
         */
        INITIALIZATION_ERROR(3),

        /**
         * This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library.
         * This is not a supported configuration.
         * Users should install an updated NVIDIA display driver to allow the application to run.
         */
        INSUFFICIENT_DRIVER(35),

        /**
         * This indicates that no CUDA-capable devices were detected by the installed CUDA driver.
         */
        NO_DEVICE(100),

        /**
         * This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device.
         */
        INVALID_DEVICE(101),

        /**
         * This error indicates the attempted operation is not permitted.
         */
        NOT_PERMITTED(800);

        /**
         * The integer enum value that represents the error in the native cuda api
         */
        private int nativeEnumValue;

        CudaError(int nativeEnumValue) {
            this.nativeEnumValue = nativeEnumValue;
        }

        public int getNativeEnumValue() {
            return nativeEnumValue;
        }

        /**
         * @param nativeEnumValue the native integer enum value of the error as defined by the native cuda api
         * @return the according {@link CudaError} enum value or null if no according enum constant is found
         */
        @Nullable
        public static CudaError fromNativeEnum(int nativeEnumValue) {
            CudaError[] errors = values();
            for (CudaError error : errors) {
                if (error.nativeEnumValue == nativeEnumValue) {
                    return error;
                }
            }
            return null;
        }

        /**
         * Throws the parent parent exception of the error
         *
         * @param causeFunction the prototype of the native function that caused the error
         */
        public void throwException(@NotNull String causeFunction) {
            switch (this) {
                case INITIALIZATION_ERROR:
                    throw new CudaInitializationError();
                case NO_DEVICE:
                    throw new CudaNoDeviceError();
                case INVALID_VALUE:
                    throw new CudaInvalidValueError(causeFunction);
                case INSUFFICIENT_DRIVER:
                    throw new CudaInsufficientDriverError();
                case NOT_PERMITTED:
                    throw new CudaNotPermittedError(causeFunction);
            }
        }
    }

    /**
     * Returns which device is currently being used.
     *
     * @return the current device for the calling host thread
     * @throws UnknownCudaError            if the native cudaGetDevice function returns an unknown error
     * @throws CudaInitializationError     if CUDA driver and runtime could not be initialized
     * @throws CudaNoDeviceError           if no CUDA-capable devices were detected by the installed CUDA driver
     * @throws CudaInvalidValueError       if the invalid value error is returned by the native cudaGetDevice function
     * @throws CudaInsufficientDriverError if the installed NVIDIA CUDA driver is older than the CUDA runtime library
     * @throws CudaNotPermittedError       if the attempted operation is not permitted
     * @throws CudaInvalidDeviceError      if an invalid device was specified
     * @throws UnknownComputingModeError   if the gpu has an unknown computing mode
     */
    @NotNull
    @Calls("cudaGetDevice")
    @NativeType("cudaDeviceProp")
    public static Device getDevice() throws UnknownCudaError, CudaInitializationError, CudaNoDeviceError, CudaInvalidValueError, CudaInsufficientDriverError, CudaNotPermittedError, CudaInvalidDeviceError, UnknownComputingModeError {
        try {
            ByteBuffer devicePointer = ByteBuffer.allocateDirect(NativeAccess.NATIVE_INT_SIZE).order(ByteOrder.nativeOrder());
            {
                int errorEnum = nCudaGetDevice(devicePointer);
                CudaError cudaError = CudaError.fromNativeEnum(errorEnum);
                String function = "cudaError_t cudaGetDevice(int *)";
                if (cudaError == null)
                    throw new UnknownCudaError(errorEnum, function);
                cudaError.throwException(function);
            }
            int device = devicePointer.get(0);
            {
                @NativeType("void **")
                ByteBuffer buffer = ByteBuffer.allocateDirect(NativeAccess.NATIVE_POINTER_SIZE).order(ByteOrder.nativeOrder()); // Where the pointer pointing to the data exchange buffer is written to.
                int errorEnum = nCudaGetDeviceProperties(buffer, device);

                CudaError cudaError = CudaError.fromNativeEnum(errorEnum);
                String function = "cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)";
                if (cudaError == null)
                    throw new UnknownCudaError(errorEnum, function);
                cudaError.throwException(function);
                return new Device(NativeAccess.getNativePointer(getBufferAddress(buffer))); // *buf is now freed
            }
        } catch (UnsatisfiedLinkError e) {
            e.printStackTrace();
            throw new NoCudaJNIException();
        }
    }

    /**
     * Unexposed unsafe function that should not be used. Use {@link #getDevice()} instead!
     *
     * @see #getDevice()
     */
    @Calls("cudaGetDevice")
    @NativeType("cudaError_t")
    private static native int nCudaGetDevice(@NotNull
                                             @NativeType("int *")
                                             @CallByReference ByteBuffer devicePointer);

    /**
     * Unexposed unsafe function that should not be used. Use {@link #getDevice()} instead!
     *
     * @param dataOut where the pointer to the dynamically allocated data exchange buffer is stored into. This is not the direct struct memory as the data layout could change in future versions of the cuda library.
     * @see #getDevice()
     */
    @Calls("cudaGetDevice")
    @NativeType("cudaError_t")
    private static native int nCudaGetDeviceProperties(@NotNull
                                                       @NativeType("void **")
                                                       @CallByReference ByteBuffer dataOut,
                                                       @NativeType("jint") int device);
}
