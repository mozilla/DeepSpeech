using CommonServiceLocator;
using DeepSpeech.WPF.ViewModels;
using DeepSpeechClient.Interfaces;
using GalaSoft.MvvmLight.Ioc;
using System.Windows;

namespace DeepSpeechWPF
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            ServiceLocator.SetLocatorProvider(() => SimpleIoc.Default);

            const int BEAM_WIDTH = 500;

            try
            {
                //Register instance of DeepSpeech
                DeepSpeechClient.DeepSpeech deepSpeechClient =
                    new DeepSpeechClient.DeepSpeech("output_graph.pbmm", BEAM_WIDTH);

                SimpleIoc.Default.Register<IDeepSpeech>(() => deepSpeechClient);
                SimpleIoc.Default.Register<MainWindowViewModel>();
            }
            catch (System.Exception ex)
            {
                MessageBox.Show(ex.Message);
                Current.Shutdown();
            }
        }

        protected override void OnExit(ExitEventArgs e)
        {
            base.OnExit(e);
            //Dispose instance of DeepSpeech
            ServiceLocator.Current.GetInstance<IDeepSpeech>()?.Dispose();
        }
    }
}
