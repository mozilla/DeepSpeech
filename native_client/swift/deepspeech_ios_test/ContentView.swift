//
//  ContentView.swift
//  deepspeech_ios_test
//
//  Created by Reuben Morais on 15.06.20.
//  Copyright © 2020 Mozilla. All rights reserved.
//

import SwiftUI

struct ContentView: View {
    private var deepspeech = DeepSpeech()
    @State var isRecognizingMicrophone = false
    
    var body: some View {
        VStack {
            Text("DeepSpeech iOS Demo")
                .font(.system(size: 30))
            Button("Recognize files", action: recognizeFiles)
                .padding(30)
            Button(
                isRecognizingMicrophone
                    ? "Stop Microphone Recognition"
                    : "Start Microphone Recognition",
                action: isRecognizingMicrophone
                    ? stopMicRecognition
                    : startMicRecognition)
                .padding(30)
        }
    }
    
    func recognizeFiles() {
        self.deepspeech.recognizeFiles()
    }
    
    func startMicRecognition() {
        isRecognizingMicrophone = true
        self.deepspeech.startMicrophoneRecognition()
    }
    
    func stopMicRecognition() {
        isRecognizingMicrophone = false
        self.deepspeech.stopMicrophoneRecognition()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
