package org.mozilla.deepspeech.nativewrapper;

import org.mozilla.deepspeech.doc.MustCall;
import org.mozilla.deepspeech.doc.NativeType;
import org.mozilla.deepspeech.exception.struct.StructBoundsExceededException;

import static org.mozilla.deepspeech.utils.NativeAccess.*;

/**
 * Represents a wrapper object object that manages a dynamically allocated native structure.
 * Extending the object provides you with functions to access your native structure.
 * The functions use the interpretation of data types that is defined by the compiler that compiled the JNI library. eg integer may even be 2 bytes (ancient compiler required).
 * Thus the integer is not altered by assuming being equal to the java integer in byte size.
 * A pointer is represented by a long to be capable of representing an either 32 bit or 64 bit pointer depending on the compiler of the JNI library.
 */
public abstract class DynamicStruct {

    /**
     * Size for structures where the size does not matter or is not known.
     * Variables of this struct cannot be accessed via getStruct...() methods as it will throw a {@link StructBoundsExceededException}
     */
    protected static final long UNDEFINED_STRUCT_SIZE = 0;

    /**
     * The size of the struct in bytes
     */
    protected final long structSize;

    /**
     * The pointer to the dynamically allocated struct
     */
    @NativeType("void *")
    private final long pointer;

    DynamicStruct(long pointer, long structSize) {
        this.pointer = pointer;
        this.structSize = structSize;
    }

    /**
     * Called to deallocate the struct once the wrapper object has been collected by the garbage collector
     */
    protected abstract void deallocateStruct(long pointer);

    /**
     * @param byteOffset the byte offset from the struct's address.
     * @return the pointer + #byteOffset interpreted as a null terminated c-string and converted into a java string. Keep in mind that pointer is unit bytes.
     * @throws StructBoundsExceededException if the specified #byteOffset is does undershoot or overshoot the memory region dedicated to the native structure
     * @see org.mozilla.deepspeech.utils.NativeAccess#getNativeCString(long)
     */
    protected String getStructString(@NativeType("void *") long byteOffset) throws StructBoundsExceededException {
        if (byteOffset < 0 || byteOffset >= structSize)
            throw new StructBoundsExceededException(byteOffset, structSize);
        return getNativeCString(this.pointer + byteOffset);
    }

    /**
     * @param byteOffset the byte offset from the struct's address.
     * @return the {@link #pointer} + #byteOffset de-referenced as a char pointer. Keep in mind that {@link #pointer} is unit bytes.
     * @throws StructBoundsExceededException if the specified #byteOffset is does undershoot or overshoot the memory region dedicated to the native structure
     * @see org.mozilla.deepspeech.utils.NativeAccess#getNativeChar(long)
     */
    protected char getStructChar(@NativeType("void *") long byteOffset) {
        if (byteOffset < 0 || byteOffset >= structSize)
            throw new StructBoundsExceededException(byteOffset, structSize);
        return getNativeChar(this.pointer + byteOffset);
    }

    /**
     * @param byteOffset the byte offset from the struct's address.
     * @return the {@link #pointer} + #byteOffset de-referenced as a short pointer. Keep in mind that {@link #pointer} is unit bytes.
     * @throws StructBoundsExceededException if the specified #byteOffset is does undershoot or overshoot the memory region dedicated to the native structure
     * @see org.mozilla.deepspeech.utils.NativeAccess#getNativeShort(long)
     */
    protected short getStructShort(@NativeType("void *") long byteOffset) {
        if (byteOffset < 0 || byteOffset >= structSize)
            throw new StructBoundsExceededException(byteOffset, structSize);
        return getNativeShort(this.pointer + byteOffset);
    }

    /**
     * @param byteOffset the byte offset from the struct's address.
     * @return the {@link #pointer} + #byteOffset de-referenced as an integer pointer. Keep in mind that {@link #pointer} is unit bytes.
     * @throws StructBoundsExceededException if the specified #byteOffset is does undershoot or overshoot the memory region dedicated to the native structure
     * @see org.mozilla.deepspeech.utils.NativeAccess#getNativeInt(long)
     */
    protected int getStructInt(@NativeType("void *") long byteOffset) {
        if (byteOffset < 0 || byteOffset >= structSize)
            throw new StructBoundsExceededException(byteOffset, structSize);
        return getNativeInt(this.pointer + byteOffset);
    }

    /**
     * @param bytesOffset the byte offset from the struct's address.
     * @return the {@link #pointer} + bytesOffset de-referenced as a long pointer. Keep in mind that {@link #pointer} is unit bytes.
     * @throws StructBoundsExceededException if the specified #byteOffset is does undershoot or overshoot the memory region dedicated to the native structure
     * @see org.mozilla.deepspeech.utils.NativeAccess#getNativeLong(long)
     */
    protected long getStructLong(@NativeType("void *") long bytesOffset) {
        if (bytesOffset < 0 || bytesOffset >= structSize)
            throw new StructBoundsExceededException(bytesOffset, structSize);
        return getNativeLong(this.pointer + bytesOffset);
    }

    /**
     * @param bytesOffset the byte offset from the struct's address.
     * @return the {@link #pointer} + bytesOffset de-referenced as a void pointer. Keep in mind that {@link #pointer} is unit bytes.
     * @throws StructBoundsExceededException if the specified #byteOffset is does undershoot or overshoot the memory region dedicated to the native structure
     * @see org.mozilla.deepspeech.utils.NativeAccess#getNativePointer(long)
     */
    protected long getStructPointer(@NativeType("void *") long bytesOffset) {
        if (bytesOffset < 0 || bytesOffset >= structSize)
            throw new StructBoundsExceededException(bytesOffset, structSize);
        return getNativePointer(this.pointer + bytesOffset);
    }

    /**
     * @param byteOffset the byte offset from the struct's address.
     * @return the {@link #pointer} + #byteOffset de-referenced as a double pointer. Keep in mind that pointer is unit bytes.
     * @throws StructBoundsExceededException if the specified #byteOffset is does undershoot or overshoot the memory region dedicated to the native structure
     * @see org.mozilla.deepspeech.utils.NativeAccess#getNativeDouble(long)
     */
    protected double getStructDouble(@NativeType("void *") long byteOffset) {
        if (byteOffset < 0 || byteOffset >= structSize)
            throw new StructBoundsExceededException(byteOffset, structSize);
        return getNativeDouble(this.pointer + byteOffset);
    }

    /**
     * @param byteOffset the byte offset from the struct's address.
     * @return the {@link #pointer} + #byteOffset de-referenced as a float pointer. Keep in mind that pointer is unit bytes.
     * @throws StructBoundsExceededException if the specified #byteOffset is does undershoot or overshoot the memory region dedicated to the native structure
     * @see org.mozilla.deepspeech.utils.NativeAccess#getNativeFloat(long)
     */
    protected double getStructFloat(@NativeType("void *") long byteOffset) {
        if (byteOffset < 0 || byteOffset >= structSize)
            throw new StructBoundsExceededException(byteOffset, structSize);
        return getNativeFloat(this.pointer + byteOffset);
    }

    /**
     * @param byteOffset the byte offset from the struct's address.
     * @return the {@link #pointer} + byteOffset de-referenced as a c++ bool pointer. Keep in mind that pointer is unit bytes.
     * @throws StructBoundsExceededException if the specified #byteOffset is does undershoot or overshoot the memory region dedicated to the native structure
     * @see org.mozilla.deepspeech.utils.NativeAccess#getNativeBoolean(long) for more information about the interpretation of logical values
     */
    protected boolean getStructBoolean(@NativeType("void *") long byteOffset) {
        if (byteOffset < 0 || byteOffset >= structSize)
            throw new StructBoundsExceededException(byteOffset, structSize);
        return getNativeBoolean(this.pointer + byteOffset);
    }

    /**
     * Represents a wrapper object for a dynamically allocated native structure that disposes it's native resources instantly on creation of the wrapper object, after storing a JVM inner copy of the native struct properties in java field properties of the wrapper object.
     */
    public static abstract class InstantlyDisposed extends DynamicStruct {

        /**
         * This constructor must call {@link #deallocateStruct(long)} because {@link InstantlyDisposed} dynamic structure wrappers dispose their native resources after being converted into java properties.
         *
         * @param pointer    the struct pointer
         * @param structSize the size of the struct in bytes
         */
        @MustCall("deallocateStruct(long)")
        protected InstantlyDisposed(@NativeType("void *") long pointer, long structSize) {
            super(pointer, structSize);
        }

    }

    /**
     * Represents a wrapper object for a dynamically allocated native structure that disposes it's native resources on garbage collection of the wrapper object
     */
    public static abstract class LifecycleDisposed extends DynamicStruct {

        /**
         * Exposed version of {@link DynamicStruct#pointer} as it is private.
         */
        protected long pointer;

        /**
         * @param pointer    the pointer to the dynamically allocated struct
         * @param structSize the size of the structure in bytes
         */
        protected LifecycleDisposed(@NativeType("void *") long pointer, long structSize) {
            super(pointer, structSize);
            this.pointer = pointer;
        }

        /*
         * Yes, "finalize" is deprecated in Java 11. Yet there is no acceptable replacement.
         */
        @Override
        protected void finalize() throws Throwable {
            deallocateStruct(pointer);
            super.finalize();
        }
    }

    /**
     * Represents a wrapper object that is not disposed
     */
    public static class NotDisposed extends DynamicStruct {

        /**
         * @param pointer    the pointer to the dynamically allocated struct
         * @param structSize the size of the structure in bytes
         */
        protected NotDisposed(long pointer, long structSize) {
            super(pointer, structSize);
        }

        @Override
        protected void deallocateStruct(long pointer) {
        }
    }

    /**
     * Represents a wrapper object that is implicitly disposed by a superordinate structure.
     */
    public static class ImplicitlyDisposed extends NotDisposed {

        /**
         * @param pointer    the pointer to the dynamically allocated struct
         * @param structSize the size of the structure in bytes
         */
        protected ImplicitlyDisposed(long pointer, long structSize) {
            super(pointer, structSize);
        }
    }
}
