using CSCore;
using CSCore.CoreAudioAPI;
using CSCore.SoundIn;
using CSCore.Streams;
using DeepSpeechClient.Interfaces;
using Microsoft.Win32; 
using System;
using System.Collections.Concurrent;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

namespace DeepSpeechWPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly IDeepSpeech _sttClient;
         
        private const uint BEAM_WIDTH = 500;
        private const float LM_ALPHA = 0.75f;
        private const float LM_BETA = 1.85f;



        #region Streaming

        private readonly WasapiCapture _audioCapture;

        private MMDeviceCollection _audioCaptureDevices;
        private SoundInSource _soundInSource;
        private IWaveSource _convertedSource;

        /// <summary>
        /// Queue that prevents feeding data to the inference engine if it is busy.
        /// </summary>
        private ConcurrentQueue<short[]> _bufferQueue = new ConcurrentQueue<short[]>();

        private int _threadSafeBoolBackValue = 0;

        /// <summary>
        /// Lock to process items in the queue one at time.
        /// </summary>
        public bool IsBusy
        {
            get => (Interlocked.CompareExchange(ref _threadSafeBoolBackValue, 1, 1) == 1);
            set
            {
                if (value) Interlocked.CompareExchange(ref _threadSafeBoolBackValue, 1, 0);
                else Interlocked.CompareExchange(ref _threadSafeBoolBackValue, 0, 1);
            }
        }
        #endregion

        public MainWindow()
        {
            InitializeComponent();
            _sttClient = new DeepSpeechClient.DeepSpeech();

            //if you want to record from the mic change to this
            //_audioCapture = new WasapiCapture();

            //we capture the windows audio output
            _audioCapture = new WasapiLoopbackCapture();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            LoadAvailableCaptureDevices();

            Task.Run(()=>
            {
                try
                {
                    _sttClient.CreateModel("output_graph.pbmm", "alphabet.txt", BEAM_WIDTH);
                    Dispatcher.Invoke(() => { EnableControls(); });
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                    Dispatcher.Invoke(() => { Close(); }); 
                }
            });
        }

        /// <summary>
        /// Loads all the available audio capture devices.
        /// </summary>
        private void LoadAvailableCaptureDevices()
        {
            DataFlow dataFlow = DataFlow.Render;  //Use render to get output devices

            // Use capture to get input devices such a microphone
            // DataFlow dataFlow = DataFlow.Capture;

            _audioCaptureDevices = MMDeviceEnumerator.EnumerateDevices(dataFlow, DeviceState.Active); //we get only enabled devices
            foreach (var device in _audioCaptureDevices)
            {
                cbxAudioInputs.Items.Add(device.FriendlyName);
            }
            if (_audioCaptureDevices.Count > 0)
            {
                cbxAudioInputs.SelectedIndex = 0;
            }
        }

        private void EnableControls()
        {
            btnEnableLM.IsEnabled = true;
            btnOpenFile.IsEnabled = true;
            btnStartRecording.IsEnabled = true;
        }

        private async void BtnTranscript_Click(object sender, RoutedEventArgs e)
        {
            txtResult.Text = string.Empty;
            btnTranscript.IsEnabled = false;
            lblStatus.Content = "Running inference...";
            Stopwatch watch = new Stopwatch();
            var waveBuffer = new NAudio.Wave.WaveBuffer(File.ReadAllBytes(txtFileName.Text));
            using (var waveInfo = new NAudio.Wave.WaveFileReader(txtFileName.Text))
            {
                Console.WriteLine("Running inference....");

                watch.Start();
                await Task.Run(() =>
                {
                    string speechResult = _sttClient.SpeechToText(waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2));
                    watch.Stop();
                    Dispatcher.Invoke(() =>
                    {
                        txtResult.Text = $"Audio duration: {waveInfo.TotalTime.ToString()} {Environment.NewLine}" +
                            $"Inference took: {watch.Elapsed.ToString()} {Environment.NewLine}" +
                            $"Recognized text: {speechResult}";
                    });
                });
            }
            waveBuffer.Clear();
            lblStatus.Content = string.Empty;
            btnTranscript.IsEnabled = true;
        }

        private async void BtnEnableLM_Click(object sender, RoutedEventArgs e)
        {
            lblStatus.Content = "Loading LM.....";
            btnEnableLM.IsEnabled = false;
            await Task.Run(() =>
            {
                try
                {
                    _sttClient.EnableDecoderWithLM("lm.binary", "trie", LM_ALPHA, LM_BETA);
                    Dispatcher.Invoke(() => lblStatus.Content = "LM loaded.");
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() => btnEnableLM.IsEnabled = true);
                    MessageBox.Show(ex.Message);
                }
            });
        }

        private void BtnOpenFile_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog
            {
                Filter = "wav Files |*.wav",
                Multiselect = false,
                Title = "Please select a wav file."
            };

            if ((bool)dialog.ShowDialog())
            {
                txtFileName.Text = dialog.FileName;
                btnTranscript.IsEnabled = true;
            }
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            _sttClient.Dispose();
            base.OnClosing(e);
        }

        private void CbxAudioInputs_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            btnStartRecording.IsEnabled = false;
            btnStopRecording.IsEnabled = false;
            if (_audioCapture.RecordingState == RecordingState.Recording)
            {
                _audioCapture.Stop();
                _soundInSource.Dispose();
                _convertedSource.Dispose();
                _audioCapture.DataAvailable -= _capture_DataAvailable;
                _sttClient.FreeStream(); //this a good example of FreeStream, the user changed the audio input, so we no longer need the current stream
            }
            if (_audioCaptureDevices!=null)
            {
                _audioCapture.Device = _audioCaptureDevices[cbxAudioInputs.SelectedIndex]; 
            }
            InitializeAudioCapture(_sttClient.GetModelSampleRate());
        }

        /// <summary>
        /// Initializes the recorder and setup the native stream.
        /// </summary>
        private void InitializeAudioCapture(int desiredSampleRate)
        {
            _audioCapture.Initialize();
            _audioCapture.DataAvailable += _capture_DataAvailable;
            _soundInSource = new SoundInSource(_audioCapture) { FillWithZeros = false };  
            //create a source, that converts the data provided by the
            //soundInSource to required by the deepspeech model
            _convertedSource = _soundInSource
               .ChangeSampleRate(desiredSampleRate) // sample rate
               .ToSampleSource()
               .ToWaveSource(16); //bits per sample
             
            _convertedSource = _convertedSource.ToMono();
            btnStartRecording.IsEnabled = true;
            
        }

        private void _capture_DataAvailable(object sender, DataAvailableEventArgs e)
        {
            //read data from the converedSource
            //important: don't use the e.Data here
            //the e.Data contains the raw data provided by the 
            //soundInSource which won't have the deepspeech required audio format
            byte[] buffer = new byte[_convertedSource.WaveFormat.BytesPerSecond / 2];

            int read;
            //int bytesReadIndex = 0;
            //keep reading as long as we still get some data
            while ((read = _convertedSource.Read(buffer, 0, buffer.Length)) > 0)
            { 
                short[] sdata = new short[(int)Math.Ceiling(Convert.ToDecimal(read / 2))];
                Buffer.BlockCopy(buffer, 0, sdata, 0, read);
                _bufferQueue.Enqueue(sdata);
                Task.Run(() => OnNewData());
            } 
        }


        private void BtnStartRecording_Click(object sender, RoutedEventArgs e)
        {
            _sttClient.CreateStream();
            _audioCapture.Start();
            btnStartRecording.IsEnabled = false;
            btnStopRecording.IsEnabled = true;
        }

        private async void BtnStopRecording_Click(object sender, RoutedEventArgs e)
        {
            btnStartRecording.IsEnabled = false;
            btnStopRecording.IsEnabled = false;
            _audioCapture.Stop();
            await Task.Run(async () =>
            {
                while (!_bufferQueue.IsEmpty && IsBusy) //we wait for all the queued buffers to be processed
                {
                    await Task.Delay(90);
                }
                string sttResult = _sttClient.FinishStream();
                Dispatcher.Invoke(() => txtResult.Text = sttResult);
            });
            btnStartRecording.IsEnabled = true;
        }

        /// <summary>
        /// Starts processing data from the queue.
        /// </summary>
        private void OnNewData()
        {
            while (!IsBusy && !_bufferQueue.IsEmpty)
            {
                if (_bufferQueue.TryDequeue(out short[] buffer))
                {
                    IsBusy = true;
                    _sttClient.FeedAudioContent(buffer, Convert.ToUInt32(buffer.Length));
                    IsBusy = false;
                }
            }
        }
    }
}
