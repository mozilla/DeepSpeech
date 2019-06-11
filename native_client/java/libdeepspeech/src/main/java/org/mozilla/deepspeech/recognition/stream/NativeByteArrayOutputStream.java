package org.mozilla.deepspeech.recognition.stream;

import org.mozilla.deepspeech.utils.NativeAccess;

import java.io.OutputStream;

public class NativeByteArrayOutputStream extends OutputStream {

    private static final long INITIAL_BUFFER_SIZE = 512;
    private long buffer = NativeAccess.allocateMemory(INITIAL_BUFFER_SIZE);
    private long index;
    private long bufferSize = INITIAL_BUFFER_SIZE;
    private long streamSize = 0;
    private final float growingFactor;

    public NativeByteArrayOutputStream(float growingFactor) {
        this.growingFactor = growingFactor;
    }

    public NativeByteArrayOutputStream() {
        this(1.125f);
    }

    @Override
    public void write(int b) {
        if (index >= bufferSize)
            growBuffer();
        NativeAccess.writeByte((byte) b, buffer + index++);
        streamSize++;
    }

    public void clear() {
        this.index = 0;
        this.streamSize = 0;
    }

    private void growBuffer() {
        long oldSize = bufferSize;
        bufferSize *= growingFactor;
        long newBuffer = NativeAccess.allocateMemory(bufferSize);
        NativeAccess.copyMemory(newBuffer, buffer, oldSize);
        NativeAccess.freeMemory(buffer);
        this.buffer = newBuffer;
    }

    public long getStreamSize() {
        return streamSize;
    }

    /**
     * <p><font color="red">WARNING DO NOT STORE THIS AS A VARIABLE!!!! THIS ADDRESS IS VERY LIKELY TO BE FREED AND REPLACED BY A NEW ONE. ALWAYS REQUEST A NEW ADDRESS FROM THE STREAM AND DON'T TOUCH THE STREAM IN ANY WAY WHILE YOU WORK WITH THE ADDRESS!!!</font>
     *
     * @return the address of the current buffer.
     */
    public long address() {
        return this.buffer;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        NativeAccess.freeMemory(buffer); // Freeing buffer
    }
}
