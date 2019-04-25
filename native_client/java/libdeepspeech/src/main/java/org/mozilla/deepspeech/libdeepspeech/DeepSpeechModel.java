package org.mozilla.deepspeech.libdeepspeech;

public class DeepSpeechModel {

    static {
        System.loadLibrary("deepspeech-jni");
        System.loadLibrary("deepspeech");
    }

    // FIXME: We should have something better than those SWIGTYPE_*
    SWIGTYPE_p_p_ModelState _mspp;
    SWIGTYPE_p_ModelState   _msp;

    public DeepSpeechModel(String modelPath, int n_cep, int n_context, String alphabetPath, int beam_width) {
        this._mspp = impl.new_modelstatep();
        impl.CreateModel(modelPath, n_cep, n_context, alphabetPath, beam_width, this._mspp);
        this._msp  = impl.modelstatep_value(this._mspp);
    }

    public void destroyModel() {
        impl.DestroyModel(this._msp);
    }

    public void enableDecoderWihLM(String alphabet, String lm, String trie, float lm_alpha, float lm_beta) {
        impl.EnableDecoderWithLM(this._msp, alphabet, lm, trie, lm_alpha, lm_beta);
    }

    public String stt(short[] buffer, int buffer_size, int sample_rate) {
        return impl.SpeechToText(this._msp, buffer, buffer_size, sample_rate);
    }

    public Metadata sttWithMetadata(short[] buffer, int buffer_size, int sample_rate) {
        return impl.SpeechToTextWithMetadata(this._msp, buffer, buffer_size, sample_rate);
    }

    public DeepSpeechStreamingState setupStream(int prealloc_frames, int sample_rate) {
        SWIGTYPE_p_p_StreamingState ssp = impl.new_streamingstatep();
        impl.SetupStream(this._msp, prealloc_frames, sample_rate, ssp);
        return new DeepSpeechStreamingState(impl.streamingstatep_value(ssp));
    }

    public void feedAudioContent(DeepSpeechStreamingState ctx, short[] buffer, int buffer_size) {
        impl.FeedAudioContent(ctx.get(), buffer, buffer_size);
    }

    public String intermediateDecode(DeepSpeechStreamingState ctx) {
        return impl.IntermediateDecode(ctx.get());
    }

    public String finishStream(DeepSpeechStreamingState ctx) {
        return impl.FinishStream(ctx.get());
    }

    public Metadata finishStreamWithMetadata(DeepSpeechStreamingState ctx) {
        return impl.FinishStreamWithMetadata(ctx.get());
    } 
}
