//
//  AudioContext.swift
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
