//
//  MozillaVoiceStt.swift
//  mozilla_voice_stt
//
//  Created by Reuben Morais on 14.06.20.
//  Copyright © 2020 Mozilla. All rights reserved.
//

import mozilla_voice_stt.libmozilla_voice_stt_Private

public enum MozillaVoiceSttError: Error {
    // Should be kept in sync with mozilla_voice_stt.h
    case noModel(errorCode: Int32)
    case invalidAlphabet(errorCode: Int32)
    case invalidShape(errorCode: Int32)
    case invalidScorer(errorCode: Int32)
    case modelIncompatible(errorCode: Int32)
    case scorerNotEnabled(errorCode: Int32)
    case scorerUnreadable(errorCode: Int32)
    case scorerInvalidLm(errorCode: Int32)
    case scorerNoTrie(errorCode: Int32)
    case scorerInvalidTrie(errorCode: Int32)
    case scorerVersionMismatch(errorCode: Int32)
    case failInitMmap(errorCode: Int32)
    case failInitSess(errorCode: Int32)
    case failInterpreter(errorCode: Int32)
    case failRunSess(errorCode: Int32)
    case failCreateStream(errorCode: Int32)
    case failReadProtobuf(errorCode: Int32)
    case failCreateSess(errorCode: Int32)
    case failCreateModel(errorCode: Int32)

    // Additional case for invalid error codes, should never happen unless the
    // user has mixed header and binary versions.
    case invalidErrorCode(errorCode: Int32)
}

extension MozillaVoiceSttError : LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .noModel(let errorCode),
             .invalidAlphabet(let errorCode),
             .invalidShape(let errorCode),
             .invalidScorer(let errorCode),
             .modelIncompatible(let errorCode),
             .scorerNotEnabled(let errorCode),
             .scorerUnreadable(let errorCode),
             .scorerInvalidLm(let errorCode),
             .scorerNoTrie(let errorCode),
             .scorerInvalidTrie(let errorCode),
             .scorerVersionMismatch(let errorCode),
             .failInitMmap(let errorCode),
             .failInitSess(let errorCode),
             .failInterpreter(let errorCode),
             .failRunSess(let errorCode),
             .failCreateStream(let errorCode),
             .failReadProtobuf(let errorCode),
             .failCreateSess(let errorCode),
             .failCreateModel(let errorCode),
             .invalidErrorCode(let errorCode):
            let result = STT_ErrorCodeToErrorMessage(errorCode)
            defer { STT_FreeString(result) }
            return String(cString: result!)
        }
    }
}

private func errorCodeToEnum(errorCode: Int32) -> MozillaVoiceSttError {
    switch Int(errorCode) {
    case Int(STT_ERR_NO_MODEL.rawValue):
        return MozillaVoiceSttError.noModel(errorCode: errorCode)
    case Int(STT_ERR_INVALID_ALPHABET.rawValue):
        return MozillaVoiceSttError.invalidAlphabet(errorCode: errorCode)
    case Int(STT_ERR_INVALID_SHAPE.rawValue):
        return MozillaVoiceSttError.invalidShape(errorCode: errorCode)
    case Int(STT_ERR_INVALID_SCORER.rawValue):
        return MozillaVoiceSttError.invalidScorer(errorCode: errorCode)
    case Int(STT_ERR_MODEL_INCOMPATIBLE.rawValue):
        return MozillaVoiceSttError.modelIncompatible(errorCode: errorCode)
    case Int(STT_ERR_SCORER_NOT_ENABLED.rawValue):
        return MozillaVoiceSttError.scorerNotEnabled(errorCode: errorCode)
    case Int(STT_ERR_SCORER_UNREADABLE.rawValue):
        return MozillaVoiceSttError.scorerUnreadable(errorCode: errorCode)
    case Int(STT_ERR_SCORER_INVALID_LM.rawValue):
        return MozillaVoiceSttError.scorerInvalidLm(errorCode: errorCode)
    case Int(STT_ERR_SCORER_NO_TRIE.rawValue):
        return MozillaVoiceSttError.scorerNoTrie(errorCode: errorCode)
    case Int(STT_ERR_SCORER_INVALID_TRIE.rawValue):
        return MozillaVoiceSttError.scorerInvalidTrie(errorCode: errorCode)
    case Int(STT_ERR_SCORER_VERSION_MISMATCH.rawValue):
        return MozillaVoiceSttError.scorerVersionMismatch(errorCode: errorCode)
    case Int(STT_ERR_FAIL_INIT_MMAP.rawValue):
        return MozillaVoiceSttError.failInitMmap(errorCode: errorCode)
    case Int(STT_ERR_FAIL_INIT_SESS.rawValue):
        return MozillaVoiceSttError.failInitSess(errorCode: errorCode)
    case Int(STT_ERR_FAIL_INTERPRETER.rawValue):
        return MozillaVoiceSttError.failInterpreter(errorCode: errorCode)
    case Int(STT_ERR_FAIL_RUN_SESS.rawValue):
        return MozillaVoiceSttError.failRunSess(errorCode: errorCode)
    case Int(STT_ERR_FAIL_CREATE_STREAM.rawValue):
        return MozillaVoiceSttError.failCreateStream(errorCode: errorCode)
    case Int(STT_ERR_FAIL_READ_PROTOBUF.rawValue):
        return MozillaVoiceSttError.failReadProtobuf(errorCode: errorCode)
    case Int(STT_ERR_FAIL_CREATE_SESS.rawValue):
        return MozillaVoiceSttError.failCreateSess(errorCode: errorCode)
    case Int(STT_ERR_FAIL_CREATE_MODEL.rawValue):
        return MozillaVoiceSttError.failCreateModel(errorCode: errorCode)
    default:
        return MozillaVoiceSttError.invalidErrorCode(errorCode: errorCode)
    }
}

private func evaluateErrorCode(errorCode: Int32) throws {
    if errorCode != Int32(STT_ERR_OK.rawValue) {
        throw errorCodeToEnum(errorCode: errorCode)
    }
}

/// Stores text of an individual token, along with its timing information
public struct MozillaVoiceSttTokenMetadata {
    /// The text corresponding to this token
    let text: String

    /// Position of the token in units of 20ms
    let timestep: Int

    /// Position of the token in seconds
    let startTime: Float

    internal init(fromInternal: TokenMetadata) {
        text = String(cString: fromInternal.text)
        timestep = Int(fromInternal.timestep)
        startTime = fromInternal.start_time
    }
}

/** A single transcript computed by the model, including a confidence value and
    the metadata for its constituent tokens
*/
public struct MozillaVoiceSttCandidateTranscript {
    /// Array of MozillaVoiceSttTokenMetadata objects
    private(set) var tokens: [MozillaVoiceSttTokenMetadata] = []

    /** Approximated confidence value for this transcript. This corresponds to
        both acoustic model and language model scores that contributed to the
        creation of this transcript.
    */
    let confidence: Double

    internal init(fromInternal: CandidateTranscript) {
        let tokensBuffer = UnsafeBufferPointer<TokenMetadata>(start: fromInternal.tokens, count: Int(fromInternal.num_tokens))
        for tok in tokensBuffer {
            tokens.append(MozillaVoiceSttTokenMetadata(fromInternal: tok))
        }
        confidence = fromInternal.confidence
    }
}

/// An array of MozillaVoiceSttCandidateTranscript objects computed by the model
public struct MozillaVoiceSttMetadata {
    /// Array of MozillaVoiceSttCandidateTranscript objects
    private(set) var transcripts: [MozillaVoiceSttCandidateTranscript] = []

    internal init(fromInternal: UnsafeMutablePointer<Metadata>) {
        let md = fromInternal.pointee
        let transcriptsBuffer = UnsafeBufferPointer<CandidateTranscript>(
            start: md.transcripts,
            count: Int(md.num_transcripts))

        for tr in transcriptsBuffer {
            transcripts.append(MozillaVoiceSttCandidateTranscript(fromInternal: tr))
        }
    }
}

public class MozillaVoiceSttStream {
    private var streamCtx: OpaquePointer!

    internal init(streamContext: OpaquePointer) {
        streamCtx = streamContext
    }

    deinit {
        if streamCtx != nil {
            STT_FreeStream(streamCtx)
            streamCtx = nil
        }
    }

    /** Feed audio samples to an ongoing streaming inference.

        - Parameter buffer: A 16-bit, mono raw audio signal at the appropriate
                            sample rate (matching what the model was trained on).

        - Precondition: `finishStream()` has not been called on this stream.
    */
    public func feedAudioContent(buffer: Array<Int16>) {
        precondition(streamCtx != nil, "calling method on invalidated Stream")

        buffer.withUnsafeBufferPointer { unsafeBufferPointer in
            feedAudioContent(buffer: unsafeBufferPointer)
        }
    }

    /** Feed audio samples to an ongoing streaming inference.

        - Parameter buffer: A 16-bit, mono raw audio signal at the appropriate
                            sample rate (matching what the model was trained on).

        - Precondition: `finishStream()` has not been called on this stream.
    */
    public func feedAudioContent(buffer: UnsafeBufferPointer<Int16>) {
        precondition(streamCtx != nil, "calling method on invalidated Stream")

        STT_FeedAudioContent(streamCtx, buffer.baseAddress, UInt32(buffer.count))
    }

    /** Compute the intermediate decoding of an ongoing streaming inference.

        - Precondition: `finishStream()` has not been called on this stream.

        - Returns: The STT intermediate result.
    */
    public func intermediateDecode() -> String {
        precondition(streamCtx != nil, "calling method on invalidated Stream")

        let result = STT_IntermediateDecode(streamCtx)
        defer { STT_FreeString(result) }
        return String(cString: result!)
    }

    /** Compute the intermediate decoding of an ongoing streaming inference,
        return results including metadata.

        - Parameter numResults: The number of candidate transcripts to return.

        - Precondition: `finishStream()` has not been called on this stream.

        - Returns: Metadata struct containing multiple CandidateTranscript structs.
                   Each transcript has per-token metadata including timing information.
    */
    public func intermediateDecodeWithMetadata(numResults: Int) -> MozillaVoiceSttMetadata {
        precondition(streamCtx != nil, "calling method on invalidated Stream")
        let result = STT_IntermediateDecodeWithMetadata(streamCtx, UInt32(numResults))!
        defer { STT_FreeMetadata(result) }
        return MozillaVoiceSttMetadata(fromInternal: result)
    }

    /** Compute the final decoding of an ongoing streaming inference and return
        the result. Signals the end of an ongoing streaming inference.

        - Precondition: `finishStream()` has not been called on this stream.

        - Returns: The STT result.

        - Postcondition: This method will invalidate this streaming context.
    */
    public func finishStream() -> String {
        precondition(streamCtx != nil, "calling method on invalidated Stream")

        let result = STT_FinishStream(streamCtx)
        defer {
            STT_FreeString(result)
            streamCtx = nil
        }
        return String(cString: result!)
    }

    /** Compute the final decoding of an ongoing streaming inference and return
        results including metadata. Signals the end of an ongoing streaming
        inference.

        - Parameter numResults: The number of candidate transcripts to return.

        - Precondition: `finishStream()` has not been called on this stream.

        - Returns: Metadata struct containing multiple CandidateTranscript structs.
                   Each transcript has per-token metadata including timing information.

        - Postcondition: This method will invalidate this streaming context.
    */
    public func finishStreamWithMetadata(numResults: Int) -> MozillaVoiceSttMetadata {
        precondition(streamCtx != nil, "calling method on invalidated Stream")

        let result = STT_FinishStreamWithMetadata(streamCtx, UInt32(numResults))!
        defer { STT_FreeMetadata(result) }
        return MozillaVoiceSttMetadata(fromInternal: result)
    }
}

/// An object providing an interface to a trained Mozilla Voice STT model.
public class MozillaVoiceSttModel {
    private var modelCtx: OpaquePointer!

    /**
        - Parameter modelPath: The path to the model file.

        - Throws: `MozillaVoiceSttError` on failure.
    */
    public init(modelPath: String) throws {
        let err = STT_CreateModel(modelPath, &modelCtx)
        try evaluateErrorCode(errorCode: err)
    }

    deinit {
        STT_FreeModel(modelCtx)
        modelCtx = nil
    }

    /** Get beam width value used by the model. If {@link STT_SetModelBeamWidth}
        was not called before, will return the default value loaded from the
        model file.

        - Returns: Beam width value used by the model.
    */
    public func getBeamWidth() -> Int {
        return Int(STT_GetModelBeamWidth(modelCtx))
    }

    /** Set beam width value used by the model.

        - Parameter beamWidth: The beam width used by the model. A larger beam
                               width value generates better results at the cost
                               of decoding time.

        - Throws: `MozillaVoiceSttError` on failure.
    */
    public func setBeamWidth(beamWidth: Int) throws {
        let err = STT_SetModelBeamWidth(modelCtx, UInt32(beamWidth))
        try evaluateErrorCode(errorCode: err)
    }

    // The sample rate expected by the model.
    public var sampleRate: Int {
        get {
            return Int(STT_GetModelSampleRate(modelCtx))
        }
    }

    /** Enable decoding using an external scorer.

        - Parameter scorerPath: The path to the external scorer file.

        - Throws: `MozillaVoiceSttError` on failure.
    */
    public func enableExternalScorer(scorerPath: String) throws {
        let err = STT_EnableExternalScorer(modelCtx, scorerPath)
        try evaluateErrorCode(errorCode: err)
    }

    /** Disable decoding using an external scorer.

        - Throws: `MozillaVoiceSttError` on failure.
    */
    public func disableExternalScorer() throws {
        let err = STT_DisableExternalScorer(modelCtx)
        try evaluateErrorCode(errorCode: err)
    }

    /** Set hyperparameters alpha and beta of the external scorer.

        - Parameter alpha: The alpha hyperparameter of the decoder. Language model weight.
        - Parameter beta: The beta hyperparameter of the decoder. Word insertion weight.

        - Throws: `MozillaVoiceSttError` on failure.
    */
    public func setScorerAlphaBeta(alpha: Float, beta: Float) throws {
        let err = STT_SetScorerAlphaBeta(modelCtx, alpha, beta)
        try evaluateErrorCode(errorCode: err)
    }

    /** Use the Mozilla Voice STT model to convert speech to text.

        - Parameter buffer: A 16-bit, mono raw audio signal at the appropriate
                            sample rate (matching what the model was trained on).

        - Returns: The STT result.
    */
    public func speechToText(buffer: Array<Int16>) -> String {
        return buffer.withUnsafeBufferPointer { unsafeBufferPointer -> String in
            return speechToText(buffer: unsafeBufferPointer)
        }
    }

    /** Use the Mozilla Voice STT model to convert speech to text.

        - Parameter buffer: A 16-bit, mono raw audio signal at the appropriate
                            sample rate (matching what the model was trained on).

        - Returns: The STT result.
    */
    public func speechToText(buffer: UnsafeBufferPointer<Int16>) -> String {
        let result = STT_SpeechToText(modelCtx, buffer.baseAddress, UInt32(buffer.count))
        defer { STT_FreeString(result) }
        return String(cString: result!)
    }

    /** Use the Mozilla Voice STT model to convert speech to text and output results
        including metadata.

        - Parameter buffer: A 16-bit, mono raw audio signal at the appropriate
                            sample rate (matching what the model was trained on).
        - Parameter numResults: The maximum number of MozillaVoiceSttCandidateTranscript
                                structs to return. Returned value might be smaller than this.

        - Returns: Metadata struct containing multiple CandidateTranscript structs.
                   Each transcript has per-token metadata including timing information.
   */
    public func speechToTextWithMetadata(buffer: Array<Int16>, numResults: Int) -> MozillaVoiceSttMetadata {
        return buffer.withUnsafeBufferPointer { unsafeBufferPointer -> MozillaVoiceSttMetadata in
            return speechToTextWithMetadata(buffer: unsafeBufferPointer, numResults: numResults)
        }
    }

    /** Use the Mozilla Voice STT model to convert speech to text and output results
        including metadata.

        - Parameter buffer: A 16-bit, mono raw audio signal at the appropriate
                            sample rate (matching what the model was trained on).
        - Parameter numResults: The maximum number of MozillaVoiceSttCandidateTranscript
                                structs to return. Returned value might be smaller than this.

        - Returns: Metadata struct containing multiple CandidateTranscript structs.
                   Each transcript has per-token metadata including timing information.
   */
    public func speechToTextWithMetadata(buffer: UnsafeBufferPointer<Int16>, numResults: Int) -> MozillaVoiceSttMetadata {
        let result = STT_SpeechToTextWithMetadata(
            modelCtx,
            buffer.baseAddress,
            UInt32(buffer.count),
            UInt32(numResults))!
        defer { STT_FreeMetadata(result) }
        return MozillaVoiceSttMetadata(fromInternal: result)
    }

    /** Create a new streaming inference state.

        - Returns: MozillaVoiceSttStream object representing the streaming state.

        - Throws: `MozillaVoiceSttError` on failure.
    */
    public func createStream() throws -> MozillaVoiceSttStream {
        var streamContext: OpaquePointer!
        let err = STT_CreateStream(modelCtx, &streamContext)
        try evaluateErrorCode(errorCode: err)
        return MozillaVoiceSttStream(streamContext: streamContext)
    }
}

public func MozillaVoiceSttVersion() -> String {
    let result = STT_Version()
    defer { STT_FreeString(result) }
    return String(cString: result!)
}
