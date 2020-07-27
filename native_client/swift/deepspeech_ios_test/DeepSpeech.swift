//
//  DeepSpeech.swift
//  deepspeech_ios_test
//
//  Created by Erik Ziegler on 27.07.20.
//  Copyright © 2020 Mozilla. All rights reserved.
//

import Foundation
import AVFoundation
import AudioToolbox
import Accelerate

import deepspeech_ios

struct FillComplexInputParm {
    var source: UnsafeMutablePointer<Int8>
    var sourceSize: UInt32
};

class DeepSpeech : NSObject, AVCaptureAudioDataOutputSampleBufferDelegate {
    private var model: DeepSpeechModel
    private var stream: DeepSpeechStream?
    
    private var captureSession = AVCaptureSession()
    private var audioData = Data()
    
    override init() {
        let modelPath = Bundle.main.path(forResource: "deepspeech-0.7.4-models", ofType: "tflite")!
        let scorerPath = Bundle.main.path(forResource: "deepspeech-0.7.4-models", ofType: "scorer")!

        model = try! DeepSpeechModel(modelPath: modelPath)
        try! model.enableExternalScorer(scorerPath: scorerPath)
        
        super.init()
        
        // prepare audio capture
        self.configureCaptureSession()
    }
    
    // MARK: Microphone recognition
    
    private func configureCaptureSession() {
        captureSession.beginConfiguration()
        
        let audioDevice = AVCaptureDevice.default(.builtInMicrophone, for: .audio, position: .unspecified)
        
        let audioDeviceInput = try! AVCaptureDeviceInput(device: audioDevice!)
        guard captureSession.canAddInput(audioDeviceInput) else { return }
        captureSession.addInput(audioDeviceInput)
        
        let serialQueue = DispatchQueue(label: "serialQueue")
        let audioOutput = AVCaptureAudioDataOutput()
        audioOutput.setSampleBufferDelegate(self, queue: serialQueue)
        
        guard captureSession.canAddOutput(audioOutput) else { return }
        captureSession.sessionPreset = .inputPriority
        captureSession.addOutput(audioOutput)
        captureSession.commitConfiguration()
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        var sourceFormat = (sampleBuffer.formatDescription?.audioFormatList[0].mASBD)!
        var destinationFormat = sourceFormat
        destinationFormat.mSampleRate = 16000.0
        
        var audioConverterRef: AudioConverterRef?
        let createConverterStatus = AudioConverterNew(&sourceFormat, &destinationFormat, &audioConverterRef)
        
        if (createConverterStatus != noErr) {
            print("Error creating converter")
        }
        
        var quality = kAudioConverterQuality_Max
        
        AudioConverterSetProperty(audioConverterRef!, kAudioConverterSampleRateConverterQuality, UInt32(MemoryLayout<UInt32>.size), &quality)

        let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer)

        var pcmLength: Int = 0
        var pcmData: UnsafeMutablePointer<Int8>?
        let status: OSStatus = CMBlockBufferGetDataPointer(blockBuffer!, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &pcmLength, dataPointerOut: &pcmData)
        
        if status != noErr {
            print("Error getting something")
        } else {
            var input = FillComplexInputParm(source: pcmData!, sourceSize: UInt32(pcmLength))
            
            let outputBuffer = malloc(pcmLength)
            memset(outputBuffer, 0, pcmLength);
            
            var outputBufferList = AudioBufferList()
            outputBufferList.mNumberBuffers = 1
            outputBufferList.mBuffers.mData = outputBuffer
            outputBufferList.mBuffers.mDataByteSize = UInt32(Double(pcmLength) * destinationFormat.mSampleRate / sourceFormat.mSampleRate)
            outputBufferList.mBuffers.mNumberChannels = 1

            func inputDataProc(
                inAudioConverter: AudioConverterRef,
                ioNumberDataPacket: UnsafeMutablePointer<UInt32>,
                ioData: UnsafeMutablePointer<AudioBufferList>,
                outDataPacketDescription: UnsafeMutablePointer<UnsafeMutablePointer<AudioStreamPacketDescription>?>?,
                inUserData: UnsafeMutableRawPointer?
            ) -> OSStatus {
                var inputPtr = inUserData!.load(as: FillComplexInputParm.self)
                
                if (inputPtr.sourceSize <= 0) {
                    ioNumberDataPacket.pointee = 1
                    return -1
                }
                
                let rawPtr = UnsafeMutableRawPointer(inputPtr.source)
                
                ioData.pointee.mNumberBuffers = 1
                ioData.pointee.mBuffers.mData = rawPtr
                ioData.pointee.mBuffers.mDataByteSize = inputPtr.sourceSize
                ioData.pointee.mBuffers.mNumberChannels = 1
                
                ioNumberDataPacket.pointee = (inputPtr.sourceSize / 2)
                inputPtr.sourceSize = 0
                
                return noErr
            };
            
            var packetSize: UInt32 = UInt32(pcmLength / 2)
            
            let status: OSStatus = AudioConverterFillComplexBuffer(audioConverterRef!, inputDataProc, &input, &packetSize, &outputBufferList, nil)
        
            if (status != noErr) {
                print("Error: " + status.description)
            } else {
                let data = outputBufferList.mBuffers.mData!
                let byteSize = outputBufferList.mBuffers.mDataByteSize
                
                let shorts = UnsafeBufferPointer(start: data.assumingMemoryBound(to: Int16.self), count: Int(byteSize / 2))
                stream!.feedAudioContent(buffer: shorts)
                let intermediateResult = stream!.intermediateDecode()
                print("Intermediate result: " + intermediateResult)
                
                // save bytes to audio data for creating a pcm file later for the captured audio
                let ptr = UnsafePointer(data.assumingMemoryBound(to: UInt8.self))
                audioData.append(ptr, count: Int(byteSize))
            }
            
            free(outputBuffer)
            AudioConverterDispose(audioConverterRef!)
        }
    }
    
    
    public func startMicrophoneRecognition() {
        audioData = Data()
        stream = try! model.createStream()
        captureSession.startRunning()
        print("Started listening...")
    }
    
    private func writeAudioDataToPCMFile() {
        let documents = NSSearchPathForDirectoriesInDomains(FileManager.SearchPathDirectory.documentDirectory, FileManager.SearchPathDomainMask.userDomainMask, true)[0]
        let filePath = documents + "/recording.pcm"
        let url = URL(fileURLWithPath: filePath)
        try! audioData.write(to: url)
        print("Saved audio to " + filePath)
    }
    
    public func stopMicrophoneRecognition() {
        captureSession.stopRunning()
        
        let result = stream?.finishStream()
        print("Result: " + result!)
        
        // optional, useful for checking the recorded audio
        writeAudioDataToPCMFile()
    }
    
    // MARK: Audio file recognition
    
    private func render(audioContext: AudioContext?, stream: DeepSpeechStream) {
        guard let audioContext = audioContext else {
            fatalError("Couldn't create the audioContext")
        }

        let sampleRange: CountableRange<Int> = 0..<audioContext.totalSamples

        guard let reader = try? AVAssetReader(asset: audioContext.asset)
            else {
                fatalError("Couldn't initialize the AVAssetReader")
        }

        reader.timeRange = CMTimeRange(start: CMTime(value: Int64(sampleRange.lowerBound), timescale: audioContext.asset.duration.timescale),
                                       duration: CMTime(value: Int64(sampleRange.count), timescale: audioContext.asset.duration.timescale))

        let outputSettingsDict: [String : Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]

        let readerOutput = AVAssetReaderTrackOutput(track: audioContext.assetTrack,
                                                    outputSettings: outputSettingsDict)
        readerOutput.alwaysCopiesSampleData = false
        reader.add(readerOutput)

        var sampleBuffer = Data()

        // 16-bit samples
        reader.startReading()
        defer { reader.cancelReading() }

        while reader.status == .reading {
            guard let readSampleBuffer = readerOutput.copyNextSampleBuffer(),
                let readBuffer = CMSampleBufferGetDataBuffer(readSampleBuffer) else {
                    break
            }
            // Append audio sample buffer into our current sample buffer
            var readBufferLength = 0
            var readBufferPointer: UnsafeMutablePointer<Int8>?
            CMBlockBufferGetDataPointer(readBuffer,
                                        atOffset: 0,
                                        lengthAtOffsetOut: &readBufferLength,
                                        totalLengthOut: nil,
                                        dataPointerOut: &readBufferPointer)
            sampleBuffer.append(UnsafeBufferPointer(start: readBufferPointer, count: readBufferLength))
            CMSampleBufferInvalidate(readSampleBuffer)

            let totalSamples = sampleBuffer.count / MemoryLayout<Int16>.size
            print("read \(totalSamples) samples")

            sampleBuffer.withUnsafeBytes { (samples: UnsafeRawBufferPointer) in
                let unsafeBufferPointer = samples.bindMemory(to: Int16.self)
                stream.feedAudioContent(buffer: unsafeBufferPointer)
            }

            sampleBuffer.removeAll()
        }

        // if (reader.status == AVAssetReaderStatusFailed || reader.status == AVAssetReaderStatusUnknown)
        guard reader.status == .completed else {
            fatalError("Couldn't read the audio file")
        }
    }
    
    private func recognizeFile(audioPath: String, completion: @escaping () -> ()) {
        let url = URL(fileURLWithPath: audioPath)

        let stream = try! model.createStream()
        print("\(audioPath)")
        let start = CFAbsoluteTimeGetCurrent()
        AudioContext.load(fromAudioURL: url, completionHandler: { audioContext in
            guard let audioContext = audioContext else {
                fatalError("Couldn't create the audioContext")
            }
            self.render(audioContext: audioContext, stream: stream)
            let result = stream.finishStream()
            let end = CFAbsoluteTimeGetCurrent()
            print("\"\(audioPath)\": \(end - start) - \(result)")
            completion()
        })
    }
    
    public func recognizeFiles() {
        let files = [
            "5639-40744-0008",
            "1089-134686-0019",
            "2094-142345-0053",
            "8463-294825-0010",
            "121-123852-0001",
            "7021-79740-0008",
            "6930-76324-0010",
            "5105-28240-0001",
            "1089-134691-0012",
            "5142-33396-0027",
            "260-123288-0004",
            "6930-75918-0008",
            "8463-294828-0005",
            "61-70970-0002"
        ]

        let serialQueue = DispatchQueue(label: "serialQueue")
        let group = DispatchGroup()
        group.enter()
        
        serialQueue.async {
            self.recognizeFile(audioPath: Bundle.main.path(forResource: "1284-134647-0003", ofType: "wav")!) {
                group.leave()
            }
        }
        
        for path in files {
            group.wait()
            group.enter()
            self.recognizeFile(audioPath: Bundle.main.path(forResource: path, ofType: "wav")!) {
                group.leave()
            }
        }
    }
}
