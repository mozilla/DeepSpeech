//
//  AppDelegate.swift
//  deepspeech_ios_test
//
//  Created by Reuben Morais on 15.06.20.
//  Copyright © 2020 Mozilla. All rights reserved.
//

import UIKit
import Foundation
import AVFoundation
import AudioToolbox
import Accelerate

import deepspeech_ios

/// Holds audio information used for building waveforms
final class AudioContext {

    /// The audio asset URL used to load the context
    public let audioURL: URL

    /// Total number of samples in loaded asset
    public let totalSamples: Int

    /// Loaded asset
    public let asset: AVAsset

    // Loaded assetTrack
    public let assetTrack: AVAssetTrack

    private init(audioURL: URL, totalSamples: Int, asset: AVAsset, assetTrack: AVAssetTrack) {
        self.audioURL = audioURL
        self.totalSamples = totalSamples
        self.asset = asset
        self.assetTrack = assetTrack
    }

    public static func load(fromAudioURL audioURL: URL, completionHandler: @escaping (_ audioContext: AudioContext?) -> ()) {
        let asset = AVURLAsset(url: audioURL, options: [AVURLAssetPreferPreciseDurationAndTimingKey: NSNumber(value: true as Bool)])

        guard let assetTrack = asset.tracks(withMediaType: AVMediaType.audio).first else {
            fatalError("Couldn't load AVAssetTrack")
        }

        asset.loadValuesAsynchronously(forKeys: ["duration"]) {
            var error: NSError?
            let status = asset.statusOfValue(forKey: "duration", error: &error)
            switch status {
            case .loaded:
                guard
                    let formatDescriptions = assetTrack.formatDescriptions as? [CMAudioFormatDescription],
                    let audioFormatDesc = formatDescriptions.first,
                    let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(audioFormatDesc)
                    else { break }

                let totalSamples = Int((asbd.pointee.mSampleRate) * Float64(asset.duration.value) / Float64(asset.duration.timescale))
                let audioContext = AudioContext(audioURL: audioURL, totalSamples: totalSamples, asset: asset, assetTrack: assetTrack)
                completionHandler(audioContext)
                return

            case .failed, .cancelled, .loading, .unknown:
                print("Couldn't load asset: \(error?.localizedDescription ?? "Unknown error")")
            }

            completionHandler(nil)
        }
    }
}

func render(audioContext: AudioContext?, stream: DeepSpeechStream) {
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

func test(model: DeepSpeechModel, audioPath: String, completion: @escaping () -> ()) {
    let url = URL(fileURLWithPath: audioPath)

    //var format = AudioStreamBasicDescription.init()
    //format.mSampleRate = 16000;
    //format.mFormatID = kAudioFormatLinearPCM;
    //format.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagsNativeEndian | kAudioFormatFlagIsPacked;
    //format.mBitsPerChannel = 16;
    //format.mChannelsPerFrame = 1;
    //format.mBytesPerFrame = format.mChannelsPerFrame * format.mBitsPerChannel / 8;
    //format.mFramesPerPacket = 1;
    //format.mBytesPerPacket = format.mFramesPerPacket * format.mBytesPerFrame;
    //
    //var file = Optional<ExtAudioFileRef>.init(nilLiteral: ());
    //let status = ExtAudioFileCreateWithURL(url as CFURL,
    //                                       kAudioFileWAVEType,
    //                                       &format,
    //                                       nil,
    //                                       0,
    //                                       &file)
    //print("status: \(status)")
    //let status2 = ExtAudioFileSetProperty(file!,
    //                                     kExtAudioFileProperty_ClientDataFormat,
    //                                     UInt32(MemoryLayout<AudioStreamBasicDescription>.size),
    //                                     &format)
    //print("status: \(status2)")
    //
    //ExtAudioFileRead(file, <#T##ioNumberFrames: UnsafeMutablePointer<UInt32>##UnsafeMutablePointer<UInt32>#>, <#T##ioData: UnsafeMutablePointer<AudioBufferList>##UnsafeMutablePointer<AudioBufferList>#>)

    let stream = try! model.createStream()
    print("\(audioPath)")
    let start = CFAbsoluteTimeGetCurrent()
    AudioContext.load(fromAudioURL: url, completionHandler: { audioContext in
        guard let audioContext = audioContext else {
            fatalError("Couldn't create the audioContext")
        }
        render(audioContext: audioContext, stream: stream)
        let result = stream.finishStream()
        let end = CFAbsoluteTimeGetCurrent()
        print("\"\(audioPath)\": \(end - start) - \(result)")
        completion()
    })

    //let file = try! AVAudioFile(forReading: url)
    //print("file length \(file.length)")
    //let format = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: false)!
    //let stream = createStream(modelState: modelState)
    //while file.framePosition < file.length {
    //    let pcmBuf = AVAudioPCMBuffer.init(pcmFormat: format, frameCapacity: 8 * 1024)! // arbitrary frameCapacity
    //    try! file.read(into: pcmBuf)
    //    if pcmBuf.frameLength == 0 {
    //        break
    //    }
    //    print("read \(pcmBuf.frameLength) frames into buffer")
    //    let rawPtr = pcmBuf.audioBufferList.pointee.mBuffers.mData!
    //    let ptr = rawPtr.bindMemory(to: Int16.self, capacity: Int(pcmBuf.frameLength))
    //    print("first few samples: \(ptr[0]) \(ptr[1]) \(ptr[2]) \(ptr[3]) ")
    //    DS_FeedAudioContent(stream, ptr, UInt32(pcmBuf.frameLength))
    //}
    //let result = DS_FinishStream(stream)
    //return String.init(cString: result!)
}

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        let model = try! DeepSpeechModel(modelPath: Bundle.main.path(forResource: "output_graph", ofType: "tflite")!)
        try! model.enableExternalScorer(scorerPath: Bundle.main.path(forResource: "librispeech_en_utf8_nonpruned_o6", ofType: "scorer")!)

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
            test(model: model, audioPath: Bundle.main.path(forResource: "1284-134647-0003", ofType: "wav")!) {
                group.leave()
            }
        }
        for path in files {
            group.wait()
            group.enter()
            test(model: model, audioPath: Bundle.main.path(forResource: path, ofType: "wav")!) {
                group.leave()
            }
        }
        return true
    }

    // MARK: UISceneSession Lifecycle

    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        // Called when a new scene session is being created.
        // Use this method to select a configuration to create the new scene with.
        return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }

    func application(_ application: UIApplication, didDiscardSceneSessions sceneSessions: Set<UISceneSession>) {
        // Called when the user discards a scene session.
        // If any sessions were discarded while the application was not running, this will be called shortly after application:didFinishLaunchingWithOptions.
        // Use this method to release any resources that were specific to the discarded scenes, as they will not return.
    }
}
