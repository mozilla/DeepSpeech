using AsyncAwaitBestPractices.MVVM;
using CSCore;
using CSCore.CoreAudioAPI;
using CSCore.SoundIn;
using CSCore.Streams;
using DeepSpeechClient.Interfaces;
using DeepSpeechClient.Models;
using GalaSoft.MvvmLight.CommandWpf;
using Microsoft.Win32;
using System;
using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace DeepSpeech.WPF.ViewModels
{
    /// <summary>
    /// View model of the MainWindow View.
    /// </summary>
    public class MainWindowViewModel : BindableBase
    {
        #region Constants
        private const int SampleRate = 16000;
        private const string ScorerPath = "kenlm.scorer";
        #endregion

        private readonly IDeepSpeech _sttClient;

        #region Commands
        /// <summary>
        /// Gets or sets the command that enables the external scorer.
        /// </summary>
        public IAsyncCommand EnableExternalScorerCommand { get; private set; }

        /// <summary>
        /// Gets or sets the command that runs inference using an audio file.
        /// </summary>
        public IAsyncCommand InferenceFromFileCommand { get; private set; }

        /// <summary>
        /// Gets or sets the command that opens a dialog to select an audio file.
        /// </summary>
        public RelayCommand SelectFileCommand { get; private set; }

        /// <summary>
        /// Gets or sets the command that starts to record.
        /// </summary>
        public RelayCommand StartRecordingCommand { get; private set; }

        /// <summary>
        /// Gets or sets the command that stops the recording and gets the result.
        /// </summary>
        public IAsyncCommand StopRecordingCommand { get; private set; }

        #endregion

        #region Streaming

        /// <summary>
        /// Stream used to feed data into the acoustic model.
        /// </summary>
        private DeepSpeechStream _sttStream;

        /// <summary>
        /// Records the audio of the selected device.
        /// </summary>
        private WasapiCapture _audioCapture;

        /// <summary>
        /// Converts the device source into a wavesource.
        /// </summary>
        private SoundInSource _soundInSource;

        /// <summary>
        /// Target wave source.(16KHz Mono 16bit for DeepSpeech)
        /// </summary>
        private IWaveSource _convertedSource;

        /// <summary>
        /// Queue that prevents feeding data to the inference engine if it is busy.
        /// </summary>
        private ConcurrentQueue<short[]> _bufferQueue = new ConcurrentQueue<short[]>();

        private int _threadSafeBoolBackValue = 0;

        /// <summary>
        /// Lock to process items in the queue one at time.
        /// </summary>
        public bool StreamingIsBusy
        {
            get => (Interlocked.CompareExchange(ref _threadSafeBoolBackValue, 1, 1) == 1);
            set
            {
                if (value) Interlocked.CompareExchange(ref _threadSafeBoolBackValue, 1, 0);
                else Interlocked.CompareExchange(ref _threadSafeBoolBackValue, 0, 1);
            }
        }

        #endregion

        #region ViewProperties

        private bool _enableStartRecord;
        /// <summary>
        /// Gets or sets record status to control the record command.
        /// </summary>
        public bool EnableStartRecord
        {
            get => _enableStartRecord;
            set => SetProperty(ref _enableStartRecord, value);
        }

        private bool _stopRecordStopRecord;
        /// <summary>
        /// Gets or sets record status to control stop command.
        /// </summary>
        public bool EnableStopRecord
        {
            get => _stopRecordStopRecord;
            set => SetProperty(ref _stopRecordStopRecord, value,
                onChanged: ()=> ((AsyncCommand)StopRecordingCommand).RaiseCanExecuteChanged());
        }

        private MMDevice _selectedDevice;
        /// <summary>
        /// Gets or sets the selected recording device.
        /// </summary>
        public MMDevice SelectedDevice
        {
            get => _selectedDevice;
            set => SetProperty(ref _selectedDevice, value, 
                onChanged: UpdateSelectedDevice);
        }

        private string _statusMessage;
        /// <summary>
        /// Gets or sets status message.
        /// </summary>
        public string StatusMessage
        {
            get => _statusMessage;
            set => SetProperty(ref _statusMessage, value);
        }

        private bool _externalScorerEnabled;
        /// <summary>
        /// Gets or sets the external scorer status.
        /// </summary>
        private bool ExternalScorerEnabled
        {
            get => _externalScorerEnabled;
            set => SetProperty(ref _externalScorerEnabled, value,
                    onChanged: () => ((AsyncCommand)EnableExternalScorerCommand).RaiseCanExecuteChanged());
        }

        private bool _isRunningInference;
        /// <summary>
        /// Gets or sets whenever the model is running inference.
        /// </summary>
        private bool IsRunningInference
        {
            get => _isRunningInference;
            set => SetProperty(ref _isRunningInference, value,
                   onChanged: () => ((AsyncCommand)InferenceFromFileCommand).RaiseCanExecuteChanged());
        }

        private string _transcription;
        /// <summary>
        /// Gets or sets the current transcription.
        /// </summary>
        public string Transcription
        {
            get => _transcription;
            set => SetProperty(ref _transcription, value);
        }

        private string _audioFilePaht;
        /// <summary>
        /// Gets or sets the selected audio file path.
        /// </summary>
        public string AudioFilePath
        {
            get => _audioFilePaht;
            set => SetProperty(ref _audioFilePaht, value);
        }

        private ObservableCollection<MMDevice> _deviceNames;
        /// <summary>
        /// Gets or sets the available recording devices.
        /// </summary>
        public ObservableCollection<MMDevice> AvailableRecordDevices
        {
            get => _deviceNames;
            set => SetProperty(ref _deviceNames, value);
        }

        #endregion

        #region Ctors
        public MainWindowViewModel(IDeepSpeech sttClient)
        {
            _sttClient = sttClient;

            EnableExternalScorerCommand = new AsyncCommand(()=>EnableExternalScorerAsync(ScorerPath),
                _ => !ExternalScorerEnabled);

            InferenceFromFileCommand = new AsyncCommand(ExecuteInferenceFromFileAsync,
                _ => !IsRunningInference);

            SelectFileCommand = new RelayCommand(SelectAudioFile);

            StartRecordingCommand = new RelayCommand(StartRecording,
                canExecute: CanExecuteStartRecording);

            StopRecordingCommand = new AsyncCommand(StopRecordingAsync,
                _ => EnableStopRecord);

            LoadAvailableCaptureDevices();
        }
        #endregion

        /// <summary>
        /// Releases the current capture device and initializes the selected one.
        /// </summary>
        private void UpdateSelectedDevice()
        {
            ReleaseCapture();
            InitializeAudioCapture();
        }

        /// <summary>
        /// Releases the capture device.
        /// </summary>
        private void ReleaseCapture()
        {
            if (_audioCapture != null)
            {
                _audioCapture.DataAvailable -= Capture_DataAvailable;
                _audioCapture.Dispose();
            }
        }
        /// <summary>
        /// Command usage to know when the recording can start.
        /// </summary>
        /// <returns>If the device is not null.</returns>
        private bool CanExecuteStartRecording() =>
            SelectedDevice != null;

        /// <summary>
        /// Loads all the available audio capture devices.
        /// </summary>
        private void LoadAvailableCaptureDevices()
        {
            AvailableRecordDevices = new ObservableCollection<MMDevice>(
                MMDeviceEnumerator.EnumerateDevices(DataFlow.All, DeviceState.Active)); //we get only enabled devices    
            EnableStartRecord = true;
            if (AvailableRecordDevices?.Count != 0)
                SelectedDevice = AvailableRecordDevices[0];
        }

        /// <summary>
        /// Initializes the capture source.
        /// </summary>
        private void InitializeAudioCapture()
        {
            if (SelectedDevice != null)
            {
                _audioCapture = SelectedDevice.DataFlow == DataFlow.Capture ?
                    new WasapiCapture() : new WasapiLoopbackCapture();
                _audioCapture.Device = SelectedDevice;
                _audioCapture.Initialize();
                _audioCapture.DataAvailable += Capture_DataAvailable;
                _soundInSource = new SoundInSource(_audioCapture) { FillWithZeros = false };
                //create a source, that converts the data provided by the
                //soundInSource to required format
                _convertedSource = _soundInSource
                   .ChangeSampleRate(SampleRate) // sample rate
                   .ToSampleSource()
                   .ToWaveSource(16); //bits per sample

                _convertedSource = _convertedSource.ToMono();
            } 
        }

        private void Capture_DataAvailable(object sender, DataAvailableEventArgs e)
        {
            //read data from the converedSource
            //important: don't use the e.Data here
            //the e.Data contains the raw data provided by the 
            //soundInSource which won't have the deepspeech required audio format
            byte[] buffer = new byte[_convertedSource.WaveFormat.BytesPerSecond / 2];

            int read;
            //keep reading as long as we still get some data
            while ((read = _convertedSource.Read(buffer, 0, buffer.Length)) > 0)
            {
                short[] sdata = new short[(int)Math.Ceiling(Convert.ToDecimal(read / 2))];
                Buffer.BlockCopy(buffer, 0, sdata, 0, read);
                _bufferQueue.Enqueue(sdata);
                Task.Run(() => OnNewData());
            }
        }

        /// <summary>
        /// Starts processing data from the queue.
        /// </summary>
        private void OnNewData()
        {
            while (!StreamingIsBusy && !_bufferQueue.IsEmpty)
            {
                if (_bufferQueue.TryDequeue(out short[] buffer))
                {
                    StreamingIsBusy = true;
                    _sttClient.FeedAudioContent(_sttStream, buffer, Convert.ToUInt32(buffer.Length));
                    StreamingIsBusy = false;
                }
            }
        }
       
        /// <summary>
        /// Enables the external scorer.
        /// </summary>
        /// <param name="scorerPath">External scorer path.</param>
        /// <returns>A Task to await.</returns>
        public async Task EnableExternalScorerAsync(string scorerPath)
        {
            try
            {
                StatusMessage = "Loading external scorer...";
                await Task.Run(() => _sttClient.EnableExternalScorer(ScorerPath));
                ExternalScorerEnabled = true;
                StatusMessage = "External scorer loaded.";
            }
            catch (Exception ex)
            {
                StatusMessage = ex.Message;
            }
        }
        /// <summary>
        /// Runs inference and sets the transcription of an audio file.
        /// </summary>
        /// <returns>A Task to await.</returns>
        public async Task ExecuteInferenceFromFileAsync()
        {
            try
            {
                IsRunningInference = true;
                Transcription = string.Empty;
                StatusMessage = "Running inference...";
                Stopwatch watch = new Stopwatch();
                var waveBuffer = new NAudio.Wave.WaveBuffer(File.ReadAllBytes(AudioFilePath));
                using (var waveInfo = new NAudio.Wave.WaveFileReader(AudioFilePath))
                {
                    watch.Start();
                    string speechResult = await Task.Run(() => _sttClient.SpeechToText(
                        waveBuffer.ShortBuffer,
                        Convert.ToUInt32(waveBuffer.MaxSize / 2)));

                    watch.Stop();
                    Transcription = $"Audio duration: {waveInfo.TotalTime.ToString()} {Environment.NewLine}" +
                        $"Inference took: {watch.Elapsed.ToString()} {Environment.NewLine}" +
                        $"Recognized text: {speechResult}";
                }
                waveBuffer.Clear();
                StatusMessage = string.Empty;
            }
            catch (Exception ex)
            {
                StatusMessage = ex.Message;
            }
            finally
            {
                IsRunningInference = false;
            }
        }

        /// <summary>
        /// Stops the recording and sets the transcription of the closed stream.
        /// </summary>
        /// <returns>A Task to await.</returns>
        private async Task StopRecordingAsync()
        {
            EnableStopRecord = false;
            _audioCapture.Stop();
            while (!_bufferQueue.IsEmpty && StreamingIsBusy) //we wait for all the queued buffers to be processed
            {
                await Task.Delay(90);
            }
            Transcription = _sttClient.FinishStream(_sttStream);
            EnableStartRecord = true;
        }

        /// <summary>
        /// Creates a new stream and starts the recording.
        /// </summary>
        private void StartRecording()
        {
            _sttStream =_sttClient.CreateStream();
            _audioCapture.Start();
            EnableStartRecord = false;
            EnableStopRecord = true;
        }

        /// <summary>
        /// Opens a dialog to select an audio file.
        /// </summary>
        private void SelectAudioFile()
        {
            OpenFileDialog dialog = new OpenFileDialog
            {
                Filter = "wav Files |*.wav",
                Multiselect = false,
                Title = "Please select a wav file."
            };

            if ((bool)dialog.ShowDialog())
            {
                AudioFilePath = dialog.FileName;
            }
        }
    }
}