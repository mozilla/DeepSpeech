package org.mozilla.deepspeech.cuda.device;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.mozilla.deepspeech.doc.NativeType;
import org.mozilla.deepspeech.exception.cuda.UnknownComputingModeError;
import org.mozilla.deepspeech.nativewrapper.DynamicStruct;
import org.mozilla.deepspeech.utils.NativeAccess;

import java.util.Arrays;

import static org.mozilla.deepspeech.utils.NativeAccess.NATIVE_INT_SIZE;
import static org.mozilla.deepspeech.utils.UtilFunctions.humanReadableByteCount;

public class Device extends DynamicStruct.InstantlyDisposed {

    /**
     * The size of the data exchange buffer
     */
    public static final int EXCHANGE_BUFFER_SIZE =
            256 + // 256 chars for name
                    16 + // 16 bytes for uuid
                    8 + // totalGlobalMem long
                    8 + // sharedMemPerBlock long
                    NATIVE_INT_SIZE + // regsPerBlock int
                    NATIVE_INT_SIZE + // warpSize int
                    8 + // memPitch long
                    NATIVE_INT_SIZE + // maxThreadsPerBlock int
                    NATIVE_INT_SIZE * 3 + // maxThreadsDim int[3]
                    NATIVE_INT_SIZE * 3 + // maxGridSize int[3]
                    8 + // totalConstMem long
                    NATIVE_INT_SIZE + // major int
                    NATIVE_INT_SIZE + // minor int
                    NATIVE_INT_SIZE + // clockRate int
                    8 + // textureAlignment long
                    NATIVE_INT_SIZE + // deviceOverlap boolean (int)
                    NATIVE_INT_SIZE + // multiProcessorCount int
                    NATIVE_INT_SIZE + // kernelExecTimeoutEnabled boolean (int)
                    NATIVE_INT_SIZE + // integrated boolean (int)
                    NATIVE_INT_SIZE + // canMapHostMemory boolean (int)
                    NATIVE_INT_SIZE // computeMode (int) (enum)
            ;

    /**
     * The name of the device
     */
    @NotNull
    private final String name;

    /**
     * 16-byte unique identifier represented by a java UUID
     */
    @NotNull
    private final byte[] uuid;

    /**
     * The total amount of global memory available on the device in bytes
     */
    private final long totalGlobalMemory;

    /**
     * The maximum amount of shared memory available to a thread block in bytes; this amount is shared by all thread blocks simultaneously resident on a multiprocessor
     */
    private final long sharedMemoryPerBlock;

    /**
     * The maximum number of 32-bit registers available to a thread block; this number is shared by all thread blocks simultaneously resident on a multiprocessor
     */
    private final int registerPerBlock;

    /**
     * The warp size in threads
     */
    private final int warpSize;

    /**
     * The maximum pitch in bytes allowed by the memory copy functions that involve memory regions allocated through cudaMallocPitch()
     */
    private final long memoryPitch;

    /**
     * Maximum number of threads per block
     */
    private final int maxThreadsPerBlock;

    /**
     * Contains the maximum size of each dimension of a block.
     * Array-length: 3
     */
    private final int[] maxThreadsDim;

    /**
     * Maximum size of each dimension of a grid
     * Array-length: 3
     */
    private final int[] maxGridSize;

    /**
     * The total amount of constant memory available on the device in bytes
     */
    private final long totalConstantMemory;

    /**
     * the major and minor revision numbers defining the device's compute capability
     */
    private final int major, minor;

    /**
     * Clock frequency in kilohertz
     */
    private final int clockRate;

    /**
     * The alignment requirement; texture base addresses that are aligned to textureAlignment bytes do not need an offset applied to texture fetches
     */
    private final long textureAlignment;

    /**
     * Device can concurrently copy memory and execute a kernel.
     * Is true if the device can concurrently copy memory between host and device while executing a kernel, or false if not.
     */
    private final boolean deviceOverlap;

    /**
     * Number of multiprocessors on device.
     */
    private final int multiProcessorCount;

    /**
     * Is true if there is a run time limit for kernels executed on the device, or false if not.
     */
    private final boolean kernelExecTimeoutEnabled;

    /**
     * Is true if the device is an integrated (motherboard) GPU and false if it is a discrete (card) component.
     */
    private final boolean integrated;

    /**
     * Is true if the device can map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer(), or false if not;
     */
    private final boolean canMapHostMemory;

    /**
     * The compute mode of the Device
     */
    @NotNull
    private final ComputeMode computeMode;

    /**
     * Constructs a device instance from the exchange buffer
     *
     * @param bufferPointer the pointer pointing to the memory of the exchange buffer.
     *                      An exchange buffer is used because the memory layout of the cudaDeviceProp struct is not guaranteed.
     *                      The bufferPointer is treated like it was a struct pointer as it has a consistent memory layout just like a struct.
     *                      Note that the memory of the buffer pointer is freed after the copy action completes.
     */
    public Device(@NativeType("void *") long bufferPointer) {
        super(bufferPointer, EXCHANGE_BUFFER_SIZE);
        long offset = 0;
        this.name = getStructString(offset);
        offset += 256;
        this.uuid = new byte[16];
        for (int i = 0; i < 16; i++) {
            this.uuid[i] = (byte) getStructChar(offset);
            offset++;
        }
        this.totalGlobalMemory = getStructLong(offset);
        offset += 8; // sizeof(long)
        this.sharedMemoryPerBlock = getStructLong(offset);
        offset += 8; // sizeof(long)
        this.registerPerBlock = getStructInt(offset);
        offset += NATIVE_INT_SIZE; // sizeof(int)
        this.warpSize = getStructInt(offset);
        offset += NATIVE_INT_SIZE; // sizeof(int)
        this.memoryPitch = getStructLong(offset);
        offset += 8; // sizeof(long)
        this.maxThreadsPerBlock = getStructInt(offset);
        offset += NATIVE_INT_SIZE; // sizeof(int)
        this.maxThreadsDim = new int[3];
        for (int i = 0; i < 3; i++) {
            this.maxThreadsDim[i] = getStructInt(offset);
            offset += NATIVE_INT_SIZE; // sizeof(int)
        }
        this.maxGridSize = new int[3];
        for (int i = 0; i < 3; i++) {
            this.maxGridSize[i] = getStructInt(offset);
            offset += NATIVE_INT_SIZE; // sizeof(int)
        }
        this.totalConstantMemory = getStructLong(offset);
        offset += 8; // sizeof(long)
        this.major = getStructInt(offset);
        offset += NATIVE_INT_SIZE; // sizeof(int)
        this.minor = getStructInt(offset);
        offset += NATIVE_INT_SIZE; // sizeof(int)
        this.clockRate = getStructInt(offset);
        offset += NATIVE_INT_SIZE; // sizeof(int)
        this.textureAlignment = getStructLong(offset);
        offset += 8; // sizeof(long)
        this.deviceOverlap = getStructBoolean(offset);
        offset += NATIVE_INT_SIZE; // sizeof(int) keep in mind that a c++ bool or c style booleans are booth integers and have the same amount of bytes as an integer. In java a boolean is only one byte. This does not apply for c / c++
        this.multiProcessorCount = getStructInt(offset);
        offset += NATIVE_INT_SIZE; // sizeof(int)
        this.kernelExecTimeoutEnabled = getStructBoolean(offset);
        offset += NATIVE_INT_SIZE; // sizeof(int) --> c/c++ boolean
        this.integrated = getStructBoolean(offset);
        offset += NATIVE_INT_SIZE; // sizeof(int) --> c/c++ boolean
        this.canMapHostMemory = getStructBoolean(offset);
        offset += NATIVE_INT_SIZE;
        int nativeComputingModeEnum = getStructInt(offset);
        ComputeMode computeMode = ComputeMode.fromNativeEnum(nativeComputingModeEnum);
        if (computeMode == null)
            throw new UnknownComputingModeError(nativeComputingModeEnum);
        this.computeMode = computeMode;
        this.deallocateStruct(bufferPointer);
    }

    @Override
    protected void deallocateStruct(long pointer) {
        NativeAccess.freeMemory(pointer);
    }

    /**
     * CUDA device compute modes
     */
    public enum ComputeMode {

        /**
         * Default compute mode (Multiple threads can use ::cudaSetDevice() with this device)
         */
        DEFAULT(0),

        /**
         * Compute-exclusive-thread mode (Only one thread in one process will be able to use ::cudaSetDevice() with this device)
         */
        EXCLUSIVE(1),

        /**
         * Compute-prohibited mode (No threads can use ::cudaSetDevice() with this device)
         */
        PROHIBITED(2),

        /**
         * Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device)
         */
        EXCLUSIVE_PROCESS(3);

        /**
         * The integer value representing the enum value in the native cuda api
         */
        private final int nativeEnumValue;

        ComputeMode(int nativeEnumValue) {
            this.nativeEnumValue = nativeEnumValue;
        }

        public int getNativeEnumValue() {
            return nativeEnumValue;
        }

        /**
         * @param nativeEnumValue the native integer enum value of the compute mode as defined by the native cuda api
         * @return the according {@link ComputeMode} enum value or null if no according enum constant is found
         */
        @Nullable
        public static ComputeMode fromNativeEnum(int nativeEnumValue) {
            ComputeMode[] modes = values();
            for (ComputeMode mode : modes) {
                if (mode.nativeEnumValue == nativeEnumValue)
                    return mode;
            }
            return null;
        }
    }


    @NotNull
    public String getName() {
        return name;
    }

    @NotNull
    public byte[] getUUID() {
        return uuid;
    }

    public long getTotalGlobalMemory() {
        return totalGlobalMemory;
    }

    public long getSharedMemoryPerBlock() {
        return sharedMemoryPerBlock;
    }

    @Override
    public String toString() {
        return "Device {" +
                "\n\tname='" + name + '\'' +
                ",\n\tuuid='" + Arrays.toString(uuid) + '\'' +
                ",\n\ttotalGlobalMemory=" + humanReadableByteCount(totalGlobalMemory, false) +
                ",\n\tsharedMemoryPerBlock=" + humanReadableByteCount(sharedMemoryPerBlock, false) +
                ",\n\tregisterPerBlock=" + registerPerBlock +
                ",\n\twarpSize=" + warpSize +
                ",\n\tmemoryPitch=" + humanReadableByteCount(memoryPitch, false) +
                ",\n\tmaxThreadsPerBlock=" + maxThreadsPerBlock +
                ",\n\tmaxThreadsDim=" + Arrays.toString(maxThreadsDim) +
                ",\n\tmaxGridSize=" + Arrays.toString(maxGridSize) +
                ",\n\ttotalConstantMemory=" + humanReadableByteCount(totalConstantMemory, false) +
                ",\n\tmajor=" + major +
                ",\n\tminor=" + minor +
                ",\n\tclockRate=" + clockRate + "KHz" +
                ",\n\ttextureAlignment=" + textureAlignment +
                ",\n\tdeviceOverlap=" + deviceOverlap +
                ",\n\tmultiProcessorCount=" + multiProcessorCount +
                ",\n\tkernelExecTimeoutEnabled=" + kernelExecTimeoutEnabled +
                ",\n\tintegrated=" + integrated +
                ",\n\tcanMapHostMemory=" + canMapHostMemory +
                ",\n\tcomputeMode=" + computeMode +
                "\n}";
    }

    public int getRegisterPerBlock() {
        return registerPerBlock;
    }

    public int getWarpSize() {
        return warpSize;
    }

    public long getMemoryPitch() {
        return memoryPitch;
    }

    public int getMaxThreadsPerBlock() {
        return maxThreadsPerBlock;
    }

    public int[] getMaxThreadsDim() {
        return maxThreadsDim;
    }

    public int[] getMaxGridSize() {
        return maxGridSize;
    }

    public long getTotalConstantMemory() {
        return totalConstantMemory;
    }

    public int getMajor() {
        return major;
    }

    public int getMinor() {
        return minor;
    }

    public int getClockRate() {
        return clockRate;
    }

    public long getTextureAlignment() {
        return textureAlignment;
    }

    public boolean isDeviceOverlap() {
        return deviceOverlap;
    }

    public int getMultiProcessorCount() {
        return multiProcessorCount;
    }

    public boolean isKernelExecTimeoutEnabled() {
        return kernelExecTimeoutEnabled;
    }

    public boolean isIntegrated() {
        return integrated;
    }

    public boolean isCanMapHostMemory() {
        return canMapHostMemory;
    }

    @NotNull
    public ComputeMode getComputeMode() {
        return computeMode;
    }
}
