# Pull in version from outside
version = File.read(File.join(__dir__, "../../training/deepspeech_training/VERSION")).split("\n")[0]

Pod::Spec.new do |s|
  s.name         = "deepspeech-ios"
  s.version      = version
  s.summary      = "DeepSpeech"
  s.homepage     = "https://github.com/mozilla/DeepSpeech"
  s.license      = "Mozilla Public License 2.0"
  s.authors      = "DeepSpeech authors"

  s.platforms    = { :ios => "9.0" }
  s.source       = { :git => "https://github.com/mozilla/DeepSpeech.git", :tag => "v#{s.version}" }

  # Assuming CI build location. Depending on your Xcode setup, this might be in
  # build/Release-iphoneos/deepspeech_ios.framework instead.
  s.vendored_frameworks = "native_client/swift/DerivedData/Build/Products/Release-iphoneos/deepspeech_ios.framework"
  s.source_files = "native_client/swift/deepspeech_ios/**/*.{h,m,mm,swift}"
end
