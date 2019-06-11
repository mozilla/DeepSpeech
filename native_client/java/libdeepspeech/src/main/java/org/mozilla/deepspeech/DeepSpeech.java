package org.mozilla.deepspeech;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.mozilla.deepspeech.doc.CallByReference;
import org.mozilla.deepspeech.doc.Calls;
import org.mozilla.deepspeech.doc.DynamicPointer;
import org.mozilla.deepspeech.doc.NativeType;
import org.mozilla.deepspeech.exception.buffer.BufferReadonlyException;
import org.mozilla.deepspeech.exception.buffer.IncorrectBufferByteOrderException;
import org.mozilla.deepspeech.exception.buffer.IncorrectBufferTypeException;
import org.mozilla.deepspeech.exception.buffer.UnexpectedBufferCapacityException;
import org.mozilla.deepspeech.exception.library.InvalidJNILibraryException;
import org.mozilla.deepspeech.recognition.stream.SpeechRecognitionAudioStream;
import org.mozilla.deepspeech.utils.BufferUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.mozilla.deepspeech.utils.NativeAccess.NATIVE_POINTER_SIZE;

/**
 * Unsafe. Use wrapper objects instead!
 *
 * @see org.mozilla.deepspeech.recognition.DeepSpeechModel
 * @see SpeechRecognitionAudioStream
 * @see org.mozilla.deepspeech.recognition.SpeechRecognitionResult
 */
public class DeepSpeech {

    /**
     * Represents the null-pointer
     */
    @NativeType("void *")
    public static long NULL = 0x0;

    /**
     * Represents the possible configurations of the jni library
     */
    public enum JNIConfiguration {

        /**
         * Represents an invalid configuration. This means that the jni library was not properly built!
         */
        NAN(-1),

        /**
         * Represents the cpu configuration
         */
        CPU(0),

        /**
         * Represents the cuda configuration
         */
        CUDA(1);

        private final int nativeEnumValue;

        JNIConfiguration(int nativeEnumValue) {
            this.nativeEnumValue = nativeEnumValue;
        }

        public int getNativeEnumValue() {
            return nativeEnumValue;
        }

        /**
         * @param nativeEnumValue the native enum value
         * @return the according enum constant or null if none is found
         */
        @Nullable
        public static JNIConfiguration fromNativeEnum(int nativeEnumValue) {
            JNIConfiguration[] configurations = values();
            for (JNIConfiguration configuration : configurations) {
                if (configuration.nativeEnumValue == nativeEnumValue)
                    return configuration;
            }
            return null;
        }
    }

    /**
     * A class containing constants with all error codes that the native functions can return
     */
    public static class ErrorCodes {

        /**
         * OK error code
         */
        public static int OK = 0x0000;

        /**
         * Missing information
         */
        public static int DS_ERR_NO_MODEL = 0x1000;

        /**
         * Invalid parameters
         */
        public static int DS_ERR_INVALID_ALPHABET = 0x2000,
                DS_ERR_INVALID_SHAPE = 0x2001,
                DS_ERR_INVALID_LM = 0x2002,
                DS_ERR_MODEL_INCOMPATIBLE = 0x2003;

        /**
         * Runtime failures
         */
        public static int DS_ERR_FAIL_INIT_MMAP = 0x3000,
                DS_ERR_FAIL_INIT_SESS = 0x3001,
                DS_ERR_FAIL_INTERPRETER = 0x3002,
                DS_ERR_FAIL_RUN_SESS = 0x3003,
                DS_ERR_FAIL_CREATE_STREAM = 0x3004,
                DS_ERR_FAIL_READ_PROTOBUF = 0x3005,
                DS_ERR_FAIL_CREATE_SESS = 0x3006;

    }

    /**
     * An object providing an interface to a trained DeepSpeech recognition.
     *
     * @param modelPath          The path to the frozen recognition graph.
     * @param nCep               The number of cepstrum the recognition was trained with.
     * @param nContext           The context window the recognition was trained with.
     * @param alphabetConfigPath The path to the configuration file specifying
     *                           the alphabet used by the network. See alphabet.h.
     * @param beamWidth          The beam width used by the decoder. A larger beam
     *                           width generates better results at the cost of decoding
     *                           time.
     * @param modelStatePointer  the long buffer of at least one long of capacity where the the ModelState pointer is written to. The recognition must be destroyed with {@link #destroyModel(long)} when no longer used.
     * @return Zero on success, non-zero on failure.
     * @throws UnexpectedBufferCapacityException if the buffer has a capacity smaller than {@link Long#BYTES} bytes.
     * @throws IncorrectBufferByteOrderException if the buffer has a byte order different to {@link ByteOrder#nativeOrder()}.
     * @throws IncorrectBufferTypeException      if the buffer is not directly allocated.
     * @throws BufferReadonlyException           if the buffer is read only
     */
    @NativeType("jint")
    public static int createModel(@NotNull String modelPath,
                                  @NativeType("jlong") long nCep,
                                  @NativeType("jlong") long nContext,
                                  @NotNull String alphabetConfigPath,
                                  @NativeType("jlong") long beamWidth,
                                  @CallByReference
                                  @NativeType("struct ModelState *")
                                  @NotNull
                                  @DynamicPointer("destroyModel") ByteBuffer modelStatePointer) throws UnexpectedBufferCapacityException, IncorrectBufferByteOrderException, IncorrectBufferTypeException, BufferReadonlyException {
        BufferUtils.checkByteBuffer(modelStatePointer, ByteOrder.nativeOrder(), 8); // 8 -> Long.BYTES
        return nCreateModel(modelPath, nCep, nContext, alphabetConfigPath, beamWidth, modelStatePointer);
    }

    /**
     * Unexposed unsafe method that should not be used. Use instead: {@link #createModel(String, long, long, String, long, ByteBuffer)}
     */
    @NativeType("jint")
    private static native int nCreateModel(@NotNull String modelPath,
                                           @NativeType("jlong") long nCep,
                                           @NativeType("jlong") long nContext,
                                           @NotNull String alphabetConfigPath,
                                           @NativeType("jlong") long beamWidth,
                                           @CallByReference
                                           @NativeType("struct ModelState *")
                                           @NotNull
                                           @DynamicPointer("destroyModel") ByteBuffer modelStatePointer) throws UnexpectedBufferCapacityException;

    /**
     * Frees associated resources and destroys recognition object.
     *
     * @param modelStatePointer the pointer pointing to the memory of the recognition state that should be freed
     */
    public static native void destroyModel(@NativeType("ModelState *") long modelStatePointer);

    /**
     * Enable decoding using beam scoring with a KenLM language recognition.
     *
     * @param modelStatePtr      The ModelState pointer for the recognition being changed.
     * @param alphaBetConfigPath The path to the configuration file specifying the alphabet used by the network. See alphabet.h.
     * @param lmPath             The path to the language recognition binary file.
     * @param triePath           The path to the trie file build from the same vocabulary as the language recognition binary.
     * @param alpha              The alpha hyperparameter of the CTC decoder. Language Model weight.
     * @param beta               The beta hyperparameter of the CTC decoder. Word insertion weight.
     * @return Zero on success, non-zero on failure (invalid arguments).
     */
    @NativeType("jint")
    public static native int enableDecoderWithLM(@NativeType("struct ModelState *") long modelStatePtr,
                                                 @NativeType("jstring") @NotNull String alphaBetConfigPath,
                                                 @NativeType("jstring") @NotNull String lmPath,
                                                 @NativeType("jstring") @NotNull String triePath,
                                                 @NativeType("jfloat") float alpha,
                                                 @NativeType("jfloat") float beta);

    /**
     * Use the DeepSpeech recognition to perform Speech-To-Text.
     *
     * @param modelStatePointer The ModelState pointer for the recognition to use.
     * @param audioBuffer       A 16-bit, mono raw audio signal at the appropriate sample rate.
     * @param numSamples        The number of samples in the audio signal.
     * @param sampleRate        The sample-rate of the audio signal.
     * @return The STT result. Returns null on error.
     * @throws UnexpectedBufferCapacityException if #numSamples does not match the allocated buffer capacity. Condition: {@code numSamples * Short.BYTES > audioBuffer.capacity()}
     * @throws IncorrectBufferByteOrderException if the audioBuffer has a byte order different to {@link ByteOrder#LITTLE_ENDIAN}.
     * @throws IncorrectBufferTypeException      if the audioBuffer is not directly allocated.
     * @throws BufferReadonlyException           if the buffer is read only
     */
    @Nullable
    @Calls("DS_SpeechToText")
    public static String speechToText(@NativeType("struct ModelState *") long modelStatePointer,
                                      @NativeType("const short *")
                                      @NotNull ByteBuffer audioBuffer,
                                      @NativeType("jlong") long numSamples,
                                      @NativeType("jlong") long sampleRate) throws UnexpectedBufferCapacityException, IncorrectBufferByteOrderException, IncorrectBufferTypeException, BufferReadonlyException {
        BufferUtils.checkByteBuffer(audioBuffer, ByteOrder.LITTLE_ENDIAN, numSamples * 2 /* sizeof(short) */);
        return nSpeechToText(modelStatePointer, audioBuffer, numSamples, sampleRate);
    }


    /**
     * Unexposed unsafe method that should not be used. Use instead: {@link #speechToText(long, ByteBuffer, long, long)}
     */
    @Nullable
    @Calls("DS_SpeechToText")
    private static native String nSpeechToText(@NativeType("struct ModelState *") long modelStatePointer,
                                               @NativeType("const short *") @NotNull ByteBuffer audioBuffer,
                                               @NativeType("jlong") long numSamples,
                                               @NativeType("jlong") long sampleRate);

    /**
     * WARNING: Unsafe function. Consider using {@link #speechToText(long, ByteBuffer, long, long)}
     * Use the DeepSpeech recognition to perform Speech-To-Text.
     *
     * @param modelStatePointer The ModelState pointer for the recognition to use.
     * @param audioBufferPtr    A 16-bit, mono raw audio signal at the appropriate sample rate.
     * @param numSamples        The number of samples in the audio signal.
     * @param sampleRate        The sample-rate of the audio signal.
     * @return The STT result. Returns null on error.
     */
    @Nullable
    @Calls("DS_SpeechToText")
    public static native String speechToTextUnsafe(@NativeType("struct ModelState *") long modelStatePointer,
                                                    @NativeType("const short *") long audioBufferPtr,
                                                    @NativeType("jlong") long numSamples,
                                                    @NativeType("jlong") long sampleRate);

    /**
     * Use the DeepSpeech recognition to perform Speech-To-Text and output metadata
     * about the results.
     *
     * @param modelStatePointer The ModelState pointer for the recognition to use.
     * @param audioBufferPtr    A 16-bit, mono raw audio signal at the appropriate sample rate.
     * @param numSamples        The number of samples in the audio signal.
     * @param sampleRate        The sample-rate of the audio signal.
     * @return Outputs a struct of individual letters along with their timing information.
     * The user is responsible for freeing Metadata by calling {@link #freeMetadata(long)}. Returns {@link #NULL} on error.
     */
    @Calls("DS_SpeechToTextWithMetadata")
    @NativeType("struct MetaData *")
    @DynamicPointer("freeMetadata")
    public static native long speechToTextWithMetadataUnsafe(@NativeType("struct ModelState *") long modelStatePointer,
                                                             @NativeType("const short *") long audioBufferPtr,
                                                             @NativeType("jlong") long numSamples,
                                                             @NativeType("jlong") long sampleRate);

    /**
     * Use the DeepSpeech recognition to perform Speech-To-Text and output metadata
     * about the results.
     *
     * @param modelStatePointer The ModelState pointer for the recognition to use.
     * @param audioBuffer       A 16-bit, mono raw audio signal at the appropriate sample rate.
     * @param numSamples        The number of samples in the audio signal.
     * @param sampleRate        The sample-rate of the audio signal.
     * @return Outputs a struct of individual letters along with their timing information.
     * The user is responsible for freeing Metadata by calling {@link #freeMetadata(long)}. Returns {@link #NULL} on error.
     * @throws UnexpectedBufferCapacityException if #numSamples does not match the allocated buffer capacity. Condition: {@code numSamples * Short.BYTES > audioBuffer.capacity()}
     * @throws IncorrectBufferByteOrderException if the audioBuffer has a byte order different to {@link ByteOrder#nativeOrder()}.
     * @throws IncorrectBufferTypeException      if the audioBuffer is not directly allocated.
     * @throws BufferReadonlyException           if the buffer is read only
     */
    @Calls("DS_SpeechToTextWithMetadata")
    @NativeType("struct MetaData *")
    @DynamicPointer("freeMetadata")
    public static long speechToTextWithMetadata(@NativeType("struct ModelState *") long modelStatePointer,
                                                @NativeType("const short *")
                                                @NotNull ByteBuffer audioBuffer,
                                                @NativeType("jlong") long numSamples,
                                                @NativeType("jlong") long sampleRate) throws UnexpectedBufferCapacityException, IncorrectBufferByteOrderException, IncorrectBufferTypeException, BufferReadonlyException {
        BufferUtils.checkByteBuffer(audioBuffer, ByteOrder.nativeOrder(), numSamples * 2 /* sizeof(short) */);
        return nSpeechToTextWithMetadata(modelStatePointer, audioBuffer, numSamples, sampleRate);
    }

    /**
     * Unexposed unsafe method that should not be used. Use instead: {@link #speechToTextWithMetadata(long, ByteBuffer, long, long)}
     */
    @Calls("DS_SpeechToTextWithMetadata")
    @NativeType("struct MetaData *")
    @DynamicPointer("freeMetadata")
    private static native long nSpeechToTextWithMetadata(@NativeType("struct ModelState *") long modelStatePointer,
                                                         @NativeType("const short *")
                                                         @NotNull ByteBuffer audioBuffer,
                                                         @NativeType("jlong") long numSamples,
                                                         @NativeType("jlong") long sampleRate);

    /**
     * Create a new streaming inference state. The streaming state returned
     * by this function can then be passed to {@link #feedAudioContent(long, ByteBuffer, long)}
     * and {@link #finishStream(long)}.
     *
     * @param modelStatePointer The ModelState pointer for the recognition to use.
     * @param preAllocFrames    Number of timestep frames to reserve. One timestep
     *                          is equivalent to two window lengths (20ms). If set to
     *                          0 we reserve enough frames for 3 seconds of audio (150).
     * @param sampleRate        The sample-rate of the audio signal.
     * @param streamPointerOut  an opaque pointer that represents the streaming state. Can
     *                          be {@link #NULL} if an error occurs.
     *                          Note for JavaBindings: The long buffer must have a capacity of one long otherwise the function will return -1. No native memory will be allocated, so this does not result in a memory leak.
     *                          The function will throw an {@link UnexpectedBufferCapacityException} stating the buffer does not have enough capacity.
     * @return Zero for success, non-zero on failure.
     * @throws UnexpectedBufferCapacityException if the buffer has a capacity smaller than {@link Long#BYTES} bytes.
     * @throws IncorrectBufferByteOrderException if the buffer has a byte order different to {@link ByteOrder#nativeOrder()}.
     * @throws IncorrectBufferTypeException      if the buffer is not directly allocated.
     * @throws BufferReadonlyException           if the buffer is read only
     */
    @Calls("DS_SetupStream")
    @NativeType("jint")
    public static int setupStream(@NativeType("struct ModelState *") long modelStatePointer,
                                  @NativeType("jlong") long preAllocFrames,
                                  @NativeType("jlong") long sampleRate,
                                  @DynamicPointer("finishStream")
                                  @NativeType("struct StreamingState **")
                                  @NotNull
                                  @CallByReference ByteBuffer streamPointerOut) throws UnexpectedBufferCapacityException, IncorrectBufferByteOrderException, IncorrectBufferTypeException, BufferReadonlyException {
        BufferUtils.checkByteBuffer(streamPointerOut, ByteOrder.nativeOrder(), NATIVE_POINTER_SIZE);
        return nSetupStream(modelStatePointer, preAllocFrames, sampleRate, streamPointerOut);
    }

    /**
     * Unexposed unsafe method that should not be used. Use instead: {@link #setupStream(long, long, long, ByteBuffer)}
     */
    @Calls("DS_SetupStream")
    @NativeType("jint")
    private static native int nSetupStream(@NativeType("struct ModelState *") long modelStatePointer,
                                           @NativeType("jlong") long preAllocFrames,
                                           @NativeType("jlong") long sampleRate,
                                           @DynamicPointer("finishStream")
                                           @NativeType("struct StreamingState **")
                                           @NotNull
                                           @CallByReference ByteBuffer streamPointerOut);

    /**
     * Feed audio samples to an ongoing streaming inference.
     *
     * @param streamPointer A streaming state pointer created by {@link #setupStream(long, long, long, ByteBuffer)}.
     * @param audioBuffer   An array of 16-bit, mono raw audio samples at the appropriate sample rate.
     * @param numSamples    The number of samples in the audio content.
     * @throws UnexpectedBufferCapacityException if #numSamples does not match the allocated buffer capacity. Condition: {@code numSamples * Short.BYTES < audioBuffer.capacity()}
     * @throws IncorrectBufferByteOrderException if the audioBuffer has a byte order different to {@link ByteOrder#nativeOrder()}.
     * @throws IncorrectBufferTypeException      if the audioBuffer is not directly allocated.
     * @throws BufferReadonlyException           if the buffer is read only
     */
    @Calls("DS_FeedAudioContent")
    public static void feedAudioContent(@NativeType("struct StreamingState *") long streamPointer,
                                        @NativeType("const short *")
                                        @NotNull ByteBuffer audioBuffer,
                                        @NativeType("jlong") long numSamples) throws UnexpectedBufferCapacityException, IncorrectBufferByteOrderException, IncorrectBufferTypeException, BufferReadonlyException {
        BufferUtils.checkByteBuffer(audioBuffer, ByteOrder.nativeOrder(), numSamples * 2); // 2 -> Short.BYTES
        nFeedAudioContent(streamPointer, audioBuffer, numSamples);
    }


    /**
     * Unexposed unsafe method that should not be used. Use instead: {@link #feedAudioContent(long, ByteBuffer, long)}
     */
    @Calls("DS_FeedAudioContent")
    private static native void nFeedAudioContent(@NativeType("struct StreamingState *") long streamPointer,
                                                 @NativeType("const short *")
                                                 @NotNull ByteBuffer audioBuffer,
                                                 @NativeType("jlong") long numSamples);

    /**
     * Compute the intermediate decoding of an ongoing streaming inference.
     * This is an expensive process as the decoder implementation isn't
     * currently capable of streaming, so it always starts from the beginning
     * of the audio.
     *
     * @param streamPointer A streaming state pointer created by {@link #setupStream(long, long, long, ByteBuffer)}.
     * @return The STT intermediate result.
     */
    @Calls("DS_IntermediateDecode")
    @NativeType("jstring")
    public static native String intermediateDecode(@NativeType("struct StreamingState *") long streamPointer);

    /**
     * This method will free the state pointer (#streamPointer)
     * Signal the end of an audio signal to an ongoing streaming
     * inference, returns the STT result over the whole audio signal.
     *
     * @param streamPointer A streaming state pointer created by {@link #setupStream(long, long, long, ByteBuffer)}.
     * @return The STT result.
     */
    @Calls("DS_FinishStream")
    @NativeType("jstring")
    public static native String finishStream(@NativeType("struct StreamingState *") long streamPointer);

    /**
     * This method will free the state pointer #streamPointer.
     * Signal the end of an audio signal to an ongoing streaming
     * inference, returns per-letter metadata.
     *
     * @param streamPointer A streaming state pointer created by {@link #setupStream(long, long, long, ByteBuffer)}.
     * @return Outputs a struct of individual letters along with their timing information.
     * The user is responsible for freeing Metadata by calling {@link #freeMetadata(long)}. Returns {@link #NULL} on error.
     */
    @Calls("DS_FinishStreamWithMetadata")
    @NativeType("struct Metadata *")
    @DynamicPointer("freeMetadata")
    public static native long finishStreamWithMetadata(@NativeType("struct StreamingState *") long streamPointer);

    /**
     * This method will free the state pointer #streamPointer.
     * Destroys a streaming state without decoding the computed logits. This
     * can be used if you no longer need the result of an ongoing streaming
     * inference and don't want to perform a costly decode operation.
     *
     * @param streamPointer A streaming state pointer created by {@link #setupStream(long, long, long, ByteBuffer)}.
     */
    @Calls("DS_DiscardStream")
    public static native void discardStream(@NativeType("struct StreamingState *") long streamPointer);

    /**
     * Frees memory allocated for metadata information.
     *
     * @param metaDataPointer the pointer pointing to the memory to be freed
     */
    @Calls("DS_Free_Metadata")
    public static native void freeMetadata(@NativeType("struct Metadata *") long metaDataPointer);

    /**
     * Prints version of this library and of the linked TensorFlow library.
     */
    @Calls("DS_PrintVersions")
    public static native void printVersions();

    /**
     * @return the configuration the jni library has been built for
     */
    @NotNull
    public static JNIConfiguration getConfiguration() {
        int configuration = nGetConfiguration();
        JNIConfiguration jniConfiguration = JNIConfiguration.fromNativeEnum(configuration);
        if (jniConfiguration == null)
            throw new InvalidJNILibraryException(configuration);
        return jniConfiguration;
    }

    /**
     * Unexposed native function. Use instead: {@link #getConfiguration()}
     *
     * @return the configuration enum value
     */
    private static native int nGetConfiguration();
}
