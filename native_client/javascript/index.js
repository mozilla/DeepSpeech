const binary = require('node-pre-gyp');
const path = require('path')
// 'lib', 'binding', 'v0.0.2', ['node', 'v' + process.versions.modules, process.platform, process.arch].join('-'), 'deepspeech-bingings.node')
const binding_path = binary.find(path.resolve(path.join(__dirname, 'package.json')));
const binding = require(binding_path);

module.exports = binding;
