package org.mozilla.voice.stt;

/**
 * @brief Exposes a Mozilla Voice STT model in Java
 **/
public class MozillaVoiceSttModel {

    static {
        System.loadLibrary("deepspeech-jni");
        System.loadLibrary("mozilla_voice_stt");
    }

    // FIXME: We should have something better than those SWIGTYPE_*
    private SWIGTYPE_p_p_ModelState _mspp;
    private SWIGTYPE_p_ModelState   _msp;

    private void evaluateErrorCode(int errorCode) {
        Error_Codes code = Error_Codes.swigToEnum(errorCode);
        if (code != Error_Codes.ERR_OK) {
            throw new RuntimeException("Error: " + impl.ErrorCodeToErrorMessage(errorCode) + " (0x" + Integer.toHexString(errorCode) + ").");
        }
    }

   /**
    * @brief An object providing an interface to a trained Mozilla Voice STT model.
    *
    * @constructor
    *
    * @param modelPath The path to the frozen model graph.
    *
    * @throws RuntimeException on failure.
    */
    public MozillaVoiceSttModel(String modelPath) {
        this._mspp = impl.new_modelstatep();
        evaluateErrorCode(impl.CreateModel(modelPath, this._mspp));
        this._msp  = impl.modelstatep_value(this._mspp);
    }

   /**
    * @brief Get beam width value used by the model. If setModelBeamWidth was not
    *        called before, will return the default value loaded from the model file.
    *
    * @return Beam width value used by the model.
    */
    public long beamWidth() {
        return impl.GetModelBeamWidth(this._msp);
    }

    /**
     * @brief Set beam width value used by the model.
     *
     * @param aBeamWidth The beam width used by the model. A larger beam width value
     *                   generates better results at the cost of decoding time.
     *
     * @throws RuntimeException on failure.
     */
    public void setBeamWidth(long beamWidth) {
        evaluateErrorCode(impl.SetModelBeamWidth(this._msp, beamWidth));
    }

   /**
    * @brief Return the sample rate expected by the model.
    *
    * @return Sample rate.
    */
    public int sampleRate() {
        return impl.GetModelSampleRate(this._msp);
    }

   /**
    * @brief Frees associated resources and destroys model object.
    */
    public void freeModel() {
        impl.FreeModel(this._msp);
    }

   /**
    * @brief Enable decoding using an external scorer.
    *
    * @param scorer The path to the external scorer file.
    *
    * @throws RuntimeException on failure.
    */
    public void enableExternalScorer(String scorer) {
        evaluateErrorCode(impl.EnableExternalScorer(this._msp, scorer));
    }

    /**
    * @brief Disable decoding using an external scorer.
    *
    * @throws RuntimeException on failure.
    */
    public void disableExternalScorer() {
        evaluateErrorCode(impl.DisableExternalScorer(this._msp));
    }

    /**
    * @brief Enable decoding using beam scoring with a KenLM language model.
    *
    * @param alpha The alpha hyperparameter of the decoder. Language model weight.
    * @param beta The beta hyperparameter of the decoder. Word insertion weight.
    *
    * @throws RuntimeException on failure.
    */
    public void setScorerAlphaBeta(float alpha, float beta) {
        evaluateErrorCode(impl.SetScorerAlphaBeta(this._msp, alpha, beta));
    }

   /*
    * @brief Use the Mozilla Voice STT model to perform Speech-To-Text.
    *
    * @param buffer A 16-bit, mono raw audio signal at the appropriate
    *                sample rate (matching what the model was trained on).
    * @param buffer_size The number of samples in the audio signal.
    *
    * @return The STT result.
    */
    public String stt(short[] buffer, int buffer_size) {
        return impl.SpeechToText(this._msp, buffer, buffer_size);
    }

   /**
    * @brief Use the Mozilla Voice STT model to perform Speech-To-Text and output metadata
    * about the results.
    *
    * @param buffer A 16-bit, mono raw audio signal at the appropriate
    *                sample rate (matching what the model was trained on).
    * @param buffer_size The number of samples in the audio signal.
    * @param num_results Maximum number of candidate transcripts to return. Returned list might be smaller than this.
    *
    * @return Metadata struct containing multiple candidate transcripts. Each transcript
    *         has per-token metadata including timing information.
    */
    public Metadata sttWithMetadata(short[] buffer, int buffer_size, int num_results) {
        return impl.SpeechToTextWithMetadata(this._msp, buffer, buffer_size, num_results);
    }

   /**
    * @brief Create a new streaming inference state. The streaming state returned
    *        by this function can then be passed to feedAudioContent()
    *        and finishStream().
    *
    * @return An opaque object that represents the streaming state.
    *
    * @throws RuntimeException on failure.
    */
    public MozillaVoiceSttStreamingState createStream() {
        SWIGTYPE_p_p_StreamingState ssp = impl.new_streamingstatep();
        evaluateErrorCode(impl.CreateStream(this._msp, ssp));
        return new MozillaVoiceSttStreamingState(impl.streamingstatep_value(ssp));
    }

   /**
    * @brief Feed audio samples to an ongoing streaming inference.
    *
    * @param cctx A streaming state pointer returned by createStream().
    * @param buffer An array of 16-bit, mono raw audio samples at the
    *                appropriate sample rate (matching what the model was trained on).
    * @param buffer_size The number of samples in @p buffer.
    */
    public void feedAudioContent(MozillaVoiceSttStreamingState ctx, short[] buffer, int buffer_size) {
        impl.FeedAudioContent(ctx.get(), buffer, buffer_size);
    }

   /**
    * @brief Compute the intermediate decoding of an ongoing streaming inference.
    *
    * @param ctx A streaming state pointer returned by createStream().
    *
    * @return The STT intermediate result.
    */
    public String intermediateDecode(MozillaVoiceSttStreamingState ctx) {
        return impl.IntermediateDecode(ctx.get());
    }

   /**
    * @brief Compute the intermediate decoding of an ongoing streaming inference.
    *
    * @param ctx A streaming state pointer returned by createStream().
    * @param num_results Maximum number of candidate transcripts to return. Returned list might be smaller than this.
    *
    * @return The STT intermediate result.
    */
    public Metadata intermediateDecodeWithMetadata(MozillaVoiceSttStreamingState ctx, int num_results) {
        return impl.IntermediateDecodeWithMetadata(ctx.get(), num_results);
    }

   /**
    * @brief Compute the final decoding of an ongoing streaming inference and return
    *        the result. Signals the end of an ongoing streaming inference.
    *
    * @param ctx A streaming state pointer returned by createStream().
    *
    * @return The STT result.
    *
    * @note This method will free the state pointer (@p ctx).
    */
    public String finishStream(MozillaVoiceSttStreamingState ctx) {
        return impl.FinishStream(ctx.get());
    }

   /**
    * @brief Compute the final decoding of an ongoing streaming inference and return
    *        the results including metadata. Signals the end of an ongoing streaming
    *        inference.
    *
    * @param ctx A streaming state pointer returned by createStream().
    * @param num_results Maximum number of candidate transcripts to return. Returned list might be smaller than this.
    *
    * @return Metadata struct containing multiple candidate transcripts. Each transcript
    *         has per-token metadata including timing information.
    *
    * @note This method will free the state pointer (@p ctx).
    */
    public Metadata finishStreamWithMetadata(MozillaVoiceSttStreamingState ctx, int num_results) {
        return impl.FinishStreamWithMetadata(ctx.get(), num_results);
    }
}
