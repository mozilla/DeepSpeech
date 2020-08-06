using CommonServiceLocator;
using MozillaVoiceStt.WPF.ViewModels;
using System.Windows;

namespace MozillaVoiceSttWPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow() => InitializeComponent();

        private void Window_Loaded(object sender, RoutedEventArgs e) =>
            DataContext = ServiceLocator.Current.GetInstance<MainWindowViewModel>();
    }
}
