package deepspeech.mozilla.org.deepspeech;

public class Model {

    // FIXME: We should have something better than those SWIGTYPE_*
    SWIGTYPE_p_p_ModelState _mspp;
    SWIGTYPE_p_ModelState   _msp;

    public Model(String modelPath, int n_cep, int n_context, String alphabetPath, int beam_width) {
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

    public SWIGTYPE_p_StreamingState setupStream(int prealloc_frames, int sample_rate) {
        SWIGTYPE_p_p_StreamingState ssp = impl.new_streamingstatep();
        impl.SetupStream(this._msp, prealloc_frames, sample_rate, ssp);
        return impl.streamingstatep_value(ssp);
    }

    public void feedAudioContent(SWIGTYPE_p_StreamingState ctx, short[] buffer, int buffer_size) {
        impl.FeedAudioContent(ctx, buffer, buffer_size);
    }

    public String intermediateDecode(SWIGTYPE_p_StreamingState ctx) {
        return impl.IntermediateDecode(ctx);
    }

    public String finishStream(SWIGTYPE_p_StreamingState ctx) {
        return impl.FinishStream(ctx);
    }
}
