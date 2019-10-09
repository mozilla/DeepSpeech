package org.mozilla.deepspeech.libdeepspeech;

/**
 * @brief Exposes a DeepSpeech model in Java
 **/
public class DeepSpeechModel {

    static {
        System.loadLibrary("deepspeech-jni");
        System.loadLibrary("deepspeech");
    }

    // FIXME: We should have something better than those SWIGTYPE_*
    SWIGTYPE_p_p_ModelState _mspp;
    SWIGTYPE_p_ModelState   _msp;

   /**
    * @brief An object providing an interface to a trained DeepSpeech model.
    *
    * @constructor
    *
    * @param modelPath The path to the frozen model graph.
    * @param alphabetPath The path to the configuration file specifying
    *                     the alphabet used by the network. See alphabet.h.
    * @param beam_width The beam width used by the decoder. A larger beam
    *                   width generates better results at the cost of decoding
    *                   time.
    */
    public DeepSpeechModel(String modelPath, String alphabetPath, int beam_width) {
        this._mspp = impl.new_modelstatep();
        impl.CreateModel(modelPath, alphabetPath, beam_width, this._mspp);
        this._msp  = impl.modelstatep_value(this._mspp);
    }

   /**
    * @brief Frees associated resources and destroys model object.
    */
    public void freeModel() {
        impl.FreeModel(this._msp);
    }

   /**
    * @brief Enable decoding using beam scoring with a KenLM language model.
    *
    * @param lm The path to the language model binary file.
    * @param trie The path to the trie file build from the same vocabulary as the language model binary.
    * @param lm_alpha The alpha hyperparameter of the CTC decoder. Language Model weight.
    * @param lm_beta The beta hyperparameter of the CTC decoder. Word insertion weight.
    *
    * @return Zero on success, non-zero on failure (invalid arguments).
    */
    public void enableDecoderWihLM(String lm, String trie, float lm_alpha, float lm_beta) {
        impl.EnableDecoderWithLM(this._msp, lm, trie, lm_alpha, lm_beta);
    }

   /*
    * @brief Use the DeepSpeech model to perform Speech-To-Text.
    *
    * @param buffer A 16-bit, mono raw audio signal at the appropriate
    *                sample rate.
    * @param buffer_size The number of samples in the audio signal.
    *
    * @return The STT result.
    */
    public String stt(short[] buffer, int buffer_size) {
        return impl.SpeechToText(this._msp, buffer, buffer_size);
    }

   /**
    * @brief Use the DeepSpeech model to perform Speech-To-Text and output metadata
    * about the results.
    *
    * @param buffer A 16-bit, mono raw audio signal at the appropriate
    *                sample rate.
    * @param buffer_size The number of samples in the audio signal.
    *
    * @return Outputs a Metadata object of individual letters along with their timing information.
    */
    public Metadata sttWithMetadata(short[] buffer, int buffer_size) {
        return impl.SpeechToTextWithMetadata(this._msp, buffer, buffer_size);
    }

   /**
    * @brief Create a new streaming inference state. The streaming state returned
    *        by this function can then be passed to feedAudioContent()
    *        and finishStream().
    *
    * @return An opaque object that represents the streaming state.
    */
    public DeepSpeechStreamingState createStream() {
        SWIGTYPE_p_p_StreamingState ssp = impl.new_streamingstatep();
        impl.CreateStream(this._msp, ssp);
        return new DeepSpeechStreamingState(impl.streamingstatep_value(ssp));
    }

   /**
    * @brief Feed audio samples to an ongoing streaming inference.
    *
    * @param cctx A streaming state pointer returned by createStream().
    * @param buffer An array of 16-bit, mono raw audio samples at the
    *                appropriate sample rate.
    * @param buffer_size The number of samples in @p buffer.
    */
    public void feedAudioContent(DeepSpeechStreamingState ctx, short[] buffer, int buffer_size) {
        impl.FeedAudioContent(ctx.get(), buffer, buffer_size);
    }

   /**
    * @brief Compute the intermediate decoding of an ongoing streaming inference.
    *        This is an expensive process as the decoder implementation isn't
    *        currently capable of streaming, so it always starts from the beginning
    *        of the audio.
    *
    * @param ctx A streaming state pointer returned by createStream().
    *
    * @return The STT intermediate result.
    */
    public String intermediateDecode(DeepSpeechStreamingState ctx) {
        return impl.IntermediateDecode(ctx.get());
    }

   /**
    * @brief Signal the end of an audio signal to an ongoing streaming
    *        inference, returns the STT result over the whole audio signal.
    *
    * @param ctx A streaming state pointer returned by createStream().
    *
    * @return The STT result.
    *
    * @note This method will free the state pointer (@p ctx).
    */
    public String finishStream(DeepSpeechStreamingState ctx) {
        return impl.FinishStream(ctx.get());
    }

   /**
    * @brief Signal the end of an audio signal to an ongoing streaming
    *        inference, returns per-letter metadata.
    *
    * @param ctx A streaming state pointer returned by createStream().
    *
    * @return Outputs a Metadata object of individual letters along with their timing information.
    *
    * @note This method will free the state pointer (@p ctx).
    */
    public Metadata finishStreamWithMetadata(DeepSpeechStreamingState ctx) {
        return impl.FinishStreamWithMetadata(ctx.get());
    }
}
