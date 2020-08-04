using CommonServiceLocator;
using MozillaVoiceStt.WPF.ViewModels;
using MozillaVoiceStt.Interfaces;
using GalaSoft.MvvmLight.Ioc;
using System.Windows;

namespace MozillaVoiceSttWPF
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

            try
            {
                //Register instance of Mozilla Voice STT
                MozillaVoiceSttClient.Model client =
                    new MozillaVoiceSttClient.Model("deepspeech-0.8.0-models.pbmm");

                SimpleIoc.Default.Register<IMozillaVoiceSttModel>(() => client);
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
            //Dispose instance of Mozilla Voice STT
            ServiceLocator.Current.GetInstance<IMozillaVoiceSttModel>()?.Dispose();
        }
    }
}
