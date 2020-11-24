# Pull in version from outside
version = File.read(File.join(__dir__, "../../training/deepspeech_training/VERSION")).split("\n")[0]

Pod::Spec.new do |s|
  s.name         = "deepspeech-ios"
  s.version      = version
  s.summary      = "DeepSpeech"
  s.homepage     = "https://github.com/mozilla/DeepSpeech"
  s.license      = "Mozilla Public License 2.0"
  s.authors      = "Mozilla et al."

  s.platforms    = { :ios => "13.5" }

  # TODO: Consider what we want to do here
  s.source       = { :git => "", :tag => "#{s.version}" }
  s.vendored_frameworks = "build/Release-iphoneos/deepspeech_ios.framework"
  s.source_files = "deepspeech_ios/**/*.{h,m,mm,swift}"
end
