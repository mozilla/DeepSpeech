declare module 'deepspeech' {
/**
 * Stores text of an individual token, along with its timing information
 */
export interface TokenMetadata {
    text: string;
    timestep: number;
    start_time: number;
}

/**
 * A single transcript computed by the model, including a confidence value and
 * the metadata for its constituent tokens.
 */
export interface CandidateTranscript {
    tokens: TokenMetadata[];
    confidence: number;
}

/**
 * An array of CandidateTranscript objects computed by the model.
 */
export interface Metadata {
    transcripts: CandidateTranscript[];
}

/**
 * @class
 * Provides an interface to a DeepSpeech stream. The constructor cannot be called
 * directly, use :js:func:`Model.createStream`.
 */
export class Stream {}

/**
 * An object providing an interface to a trained DeepSpeech model.
 *
 * @param aModelPath The path to the frozen model graph.
 *
 * @throws on error
 */
export class Model {
constructor(aModelPath: string)

/**
 * Get beam width value used by the model. If :js:func:Model.setBeamWidth was
 * not called before, will return the default value loaded from the model file.
 * 
 * @return Beam width value used by the model.
 */
beamWidth(): number;

/**
 * Set beam width value used by the model.
 * 
 * @param The beam width used by the model. A larger beam width value generates better results at the cost of decoding time.
 *
 * @return Zero on success, non-zero on failure.
 */
setBeamWidth(aBeamWidth: number): number;

/**
 * Return the sample rate expected by the model.
 *
 * @return Sample rate.
 */
sampleRate(): number;

/**
 * Enable decoding using an external scorer.
 *
 * @param aScorerPath The path to the external scorer file.
 *
 * @return Zero on success, non-zero on failure (invalid arguments).
 */
enableExternalScorer(aScorerPath: string): number;

/**
 * Disable decoding using an external scorer.
 *
 * @return Zero on success, non-zero on failure (invalid arguments).
 */
disableExternalScorer(): number;

/**
 * Set hyperparameters alpha and beta of the external scorer.
 *
 * @param aLMAlpha The alpha hyperparameter of the CTC decoder. Language Model weight.
 * @param aLMBeta The beta hyperparameter of the CTC decoder. Word insertion weight.
 *
 * @return Zero on success, non-zero on failure (invalid arguments).
 */
setScorerAlphaBeta(aLMAlpha: number, aLMBeta: number): number;

/**
 * Use the DeepSpeech model to perform Speech-To-Text.
 *
 * @param aBuffer A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).
 *
 * @return The STT result. Returns undefined on error.
 */
stt(aBuffer: object): string;

/**
 * Use the DeepSpeech model to perform Speech-To-Text and output metadata
 * about the results.
 *
 * @param aBuffer A 16-bit, mono raw audio signal at the appropriate sample rate (matching what the model was trained on).
 * @param aNumResults Maximum number of candidate transcripts to return. Returned list might be smaller than this.
 * Default value is 1 if not specified.
 *
 * @return :js:func:`Metadata` object containing multiple candidate transcripts. Each transcript has per-token metadata including timing information.
 * The user is responsible for freeing Metadata by calling :js:func:`FreeMetadata`. Returns undefined on error.
 */
sttWithMetadata(aBuffer: object, aNumResults: number): Metadata;

/**
 * Create a new streaming inference state. One can then call :js:func:`Stream.feedAudioContent` and :js:func:`Stream.finishStream` on the returned stream object.
 *
 * @return a :js:func:`Stream` object that represents the streaming state.
 *
 * @throws on error
 */
createStream(): object;

/**
 * Feed audio samples to an ongoing streaming inference.
 *
 * @param aBuffer An array of 16-bit, mono raw audio samples at the
 *                 appropriate sample rate (matching what the model was trained on).
 */
feedAudioContent(aBuffer: object): void;

/**
 * Compute the intermediate decoding of an ongoing streaming inference.
 *
 * @return The STT intermediate result.
 */
intermediateDecode(aSctx: object): string;

/**
 * Compute the intermediate decoding of an ongoing streaming inference, return results including metadata.
 *
 * @param aNumResults Maximum number of candidate transcripts to return. Returned list might be smaller than this. Default value is 1 if not specified.
 *
 * @return :js:func:`Metadata` object containing multiple candidate transcripts. Each transcript has per-token metadata including timing information. The user is responsible for freeing Metadata by calling :js:func:`FreeMetadata`. Returns undefined on error.
 */
intermediateDecodeWithMetadata (aNumResults: number): Metadata;

/**
 * Compute the final decoding of an ongoing streaming inference and return the result. Signals the end of an ongoing streaming inference.
 *
 * @return The STT result.
 *
 * This method will free the stream, it must not be used after this method is called.
 */
finishStream(): string;

/**
 * Compute the final decoding of an ongoing streaming inference and return the results including metadata. Signals the end of an ongoing streaming inference.
 *
 * @param aNumResults Maximum number of candidate transcripts to return. Returned list might be smaller than this. Default value is 1 if not specified.
 *
 * @return Outputs a :js:func:`Metadata` struct of individual letters along with their timing information. The user is responsible for freeing Metadata by calling :js:func:`FreeMetadata`.
 *
 * This method will free the stream, it must not be used after this method is called.
 */
finishStreamWithMetadata(aNumResults: number): Metadata;
}

/**
 * Frees associated resources and destroys model object.
 *
 * @param model A model pointer returned by :js:func:`Model`
 *
 */
export function FreeModel(model: Model): void;

/**
 * Free memory allocated for metadata information.
 *
 * @param metadata Object containing metadata as returned by :js:func:`Model.sttWithMetadata` or :js:func:`Model.finishStreamWithMetadata`
 */
export function FreeMetadata(metadata: Metadata): void;

/**
 * Destroy a streaming state without decoding the computed logits. This
 * can be used if you no longer need the result of an ongoing streaming
 * inference and don't want to perform a costly decode operation.
 *
 * @param stream A streaming state pointer returned by :js:func:`Model.createStream`.
 */
export function FreeStream(stream: object): void;

/**
 * Print version of this library and of the linked TensorFlow library on standard output.
 */
export function Version(): void;
}
