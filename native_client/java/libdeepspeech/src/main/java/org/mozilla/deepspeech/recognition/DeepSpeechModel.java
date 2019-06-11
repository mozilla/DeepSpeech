package org.mozilla.deepspeech.recognition;

import org.jetbrains.annotations.NotNull;
import org.mozilla.deepspeech.doc.NativeType;
import org.mozilla.deepspeech.doc.WrappsStruct;
import org.mozilla.deepspeech.exception.buffer.BufferReadonlyException;
import org.mozilla.deepspeech.exception.buffer.IncorrectBufferByteOrderException;
import org.mozilla.deepspeech.exception.buffer.IncorrectBufferTypeException;
import org.mozilla.deepspeech.exception.buffer.UnexpectedBufferCapacityException;
import org.mozilla.deepspeech.nativewrapper.DynamicStruct;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.mozilla.deepspeech.DeepSpeech.*;
import static org.mozilla.deepspeech.utils.BufferUtils.getBufferAddress;
import static org.mozilla.deepspeech.utils.NativeAccess.NATIVE_POINTER_SIZE;
import static org.mozilla.deepspeech.utils.NativeAccess.getNativePointer;
import static org.mozilla.deepspeech.utils.UtilFunctions.checkExists;

/**
 * Represents a trained deep speech recognition recognition
 */
@WrappsStruct("ModelState")
public class DeepSpeechModel extends DynamicStruct.LifecycleDisposed {

    /**
     * @param modelFile          The file pointing to the frozen recognition graph.
     * @param numCep             The number of cepstrum the recognition was trained with.
     * @param context            The context window the recognition was trained with.
     * @param alphabetConfigFile The path to the configuration file specifying
     *                           the alphabet used by the network. See alphabet.h.
     * @param beamWidth          The beam width used by the decoder. A larger beam
     *                           width generates better results at the cost of decoding
     *                           time.
     * @throws FileNotFoundException if either modelFile or alphabetConfigFile is not found
     */
    public DeepSpeechModel(@NotNull File modelFile, long numCep, long context, @NotNull File alphabetConfigFile, long beamWidth) throws FileNotFoundException {
        super(newModel(modelFile, numCep, context, alphabetConfigFile, beamWidth), UNDEFINED_STRUCT_SIZE);
    }

    /**
     * Enables decoding using beam scoring with a KenLM language model.
     *
     * @param alphabetFile The path to the configuration file specifying the alphabet used by the network.
     * @param lmBinaryFile The path to the language model binary file.
     * @param trieFile     The path to the trie file build from the same vocabulary as the language model binary
     * @param lmAlpha      The alpha hyper-parameter of the CTC decoder. Language Model weight.
     * @param lmBeta       The beta hyper-parameter of the CTC decoder. Word insertion weight.
     * @throws FileNotFoundException if one of the files is not found.
     */
    public void enableLMLanguageModel(@NotNull File alphabetFile, @NotNull File lmBinaryFile, @NotNull File trieFile, float lmAlpha, float lmBeta) throws FileNotFoundException {
        enableDecoderWithLM(this.pointer, checkExists(alphabetFile).getPath(), checkExists(lmBinaryFile).getPath(), checkExists(trieFile).getPath(), lmAlpha, lmBeta);
    }

    /**
     * Performs a text to speech call on the recognition
     *
     * @param audioBuffer the audio buffer storing the audio data in samples / frames to perform the recognition on
     * @param numSamples  the number of samples / frames in the buffer
     * @param sampleRate  the amount of samples representing a given duration of audio. sampleRate = Δ samples / Δ time
     * @return the transcription string
     * @throws UnexpectedBufferCapacityException if #numSamples does not match the allocated buffer capacity. Condition: {@code numSamples * Short.BYTES > audioBuffer.capacity()}
     * @throws IncorrectBufferByteOrderException if the audioBuffer has a byte order different to {@link ByteOrder#nativeOrder()}.
     * @throws IncorrectBufferTypeException      if the audioBuffer is not directly allocated.
     * @throws BufferReadonlyException           if the buffer is read only
     */
    @NotNull
    public String doSpeechToText(@NativeType("const short *") @NotNull ByteBuffer audioBuffer, long numSamples, long sampleRate) throws UnexpectedBufferCapacityException, IncorrectBufferByteOrderException, IncorrectBufferTypeException, BufferReadonlyException {
        String ret = speechToText(this.pointer, audioBuffer, numSamples, sampleRate);
        if (ret == null) throw new NullPointerException();
        return ret;
    }

    /**
     * Performs a text to speech call on the recognition and returns a more detailed STT result with additional meta data - not just a sting
     *
     * @param audioBuffer the audio buffer storing the audio data in samples / frames to perform the recognition on
     * @param numSamples  the number of samples / frames in the buffer
     * @param sampleRate  the amount of samples representing a given duration of audio. sampleRate = Δ samples / Δ time
     * @return the meta data of transcription
     * @see SpeechRecognitionResult
     */
    @NotNull
    public SpeechRecognitionResult doSpeechRecognitionWithMeta(@NativeType("const short *") @NotNull ByteBuffer audioBuffer, long numSamples, long sampleRate) {
        long metaPointer = speechToTextWithMetadata(this.pointer, audioBuffer, numSamples, sampleRate);
        if (metaPointer == NULL) throw new NullPointerException();
        return new SpeechRecognitionResult(metaPointer); // Meta pointer is freed as Recognition Result instantly disposes it after copying the values.
    }

    /**
     * Allocates a new native recognition structure and returns the pointer pointing to the dynamically allocated memory
     *
     * @see DeepSpeechModel#DeepSpeechModel(File, long, long, File, long)
     */
    private static long newModel(@NotNull File modelFile, long numCep, long context, @NotNull File alphabetConfigFile, long beamWidth) throws FileNotFoundException {
        ByteBuffer ptr = ByteBuffer.allocateDirect(NATIVE_POINTER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
        if (createModel(checkExists(modelFile).getPath(), numCep, context, checkExists(alphabetConfigFile).getPath(), beamWidth, ptr) != 0)
            throw new RuntimeException("Failed to create recognition!");
        return getNativePointer(getBufferAddress(ptr));
    }

    /**
     * De-allocates the struct memory when this object is garbage collected
     */
    @Override
    protected void deallocateStruct(long pointer) {
        destroyModel(pointer);
    }

    /**
     * @return the pointer to the native struct below
     */
    public long getPointer() {
        return this.pointer;
    }

    /**
     * Unsafe function! Consider using {@link #doSpeechToText(ByteBuffer, long, long)} instead.
     * Performs a text to speech call on the recognition.
     *
     * @param audioBufferPointer the audio buffer storing the audio data in samples / frames to perform the recognition on
     * @param numSamples         the number of samples / frames in the buffer
     * @param sampleRate         the amount of samples representing a given duration of audio. sampleRate = Δ samples / Δ time
     * @return the transcription string
     */
    @NotNull
    public String doSpeechToTextUnsafe(@NativeType("const short *") long audioBufferPointer, long numSamples, long sampleRate) {
        String ret = speechToTextUnsafe(this.pointer, audioBufferPointer, numSamples, sampleRate);
        if (ret == null) throw new NullPointerException();
        return ret;
    }

    /**
     * Unsafe function! Consider using {@link #doSpeechRecognitionWithMeta(ByteBuffer, long, long)} instead.
     *
     * @param audioBufferPointer the audio buffer storing the audio data in samples / frames to perform the recognition on
     * @param numSamples         the number of samples / frames in the buffer
     * @param sampleRate         the amount of samples representing a given duration of audio. sampleRate = Δ samples / Δ time
     * @return the meta data of transcription
     * @see SpeechRecognitionResult
     */
    @NotNull
    public SpeechRecognitionResult doSpeechRecognitionWithMetaUnsafe(@NativeType("const short *") long audioBufferPointer, long numSamples, long sampleRate) {
        long metaPtr = speechToTextWithMetadataUnsafe(this.pointer, audioBufferPointer, numSamples, sampleRate);
        if (metaPtr == NULL) throw new NullPointerException();
        return new SpeechRecognitionResult(metaPtr); // MetaPtr is freed after this action
    }
}
