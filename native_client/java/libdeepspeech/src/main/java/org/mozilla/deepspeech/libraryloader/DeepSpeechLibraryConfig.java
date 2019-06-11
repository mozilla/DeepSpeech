package org.mozilla.deepspeech.libraryloader;

import org.jetbrains.annotations.NotNull;
import org.mozilla.deepspeech.DeepSpeech;
import org.mozilla.deepspeech.cuda.device.Device;
import org.mozilla.deepspeech.exception.cuda.CudaException;
import org.mozilla.deepspeech.exception.library.InvalidJNILibraryException;
import org.mozilla.deepspeech.exception.library.WrongJNILibraryException;
import org.mozilla.deepspeech.libraryloader.utils.NativeLibraryLoader;

import java.io.IOException;
import java.net.URL;

/**
 * A library loading configuration
 */
public abstract class DeepSpeechLibraryConfig {

    /**
     * The expected configuration that the jni library must be built for to use this library loading configuration
     */
    @NotNull
    private final DeepSpeech.JNIConfiguration parentJNIConfiguration;

    protected DeepSpeechLibraryConfig(@NotNull DeepSpeech.JNIConfiguration parentJNIConfiguration) {
        this.parentJNIConfiguration = parentJNIConfiguration;
    }

    /**
     * Called to pre-initialize the library configuration
     */
    protected void preInit() throws IOException {
    }

    /**
     * Called to load the deep-speech native library
     */
    protected abstract void loadDeepSpeechLibrary() throws IOException;

    /**
     * Called to load the deep-speech jni function wrapper library
     */
    protected abstract void loadDeepSpeechJNI() throws IOException;

    /**
     * Loads the DeepSpeech native library accordingly to this library configuration
     */
    public void loadDeepSpeech() throws IOException {
        preInit(); // Pre-initializing the configuration
        loadDeepSpeechLibrary(); // The deep-speech library must be loaded before the jni library!
        loadDeepSpeechJNI();
        DeepSpeech.JNIConfiguration configuration = DeepSpeech.getConfiguration();
        if (configuration == DeepSpeech.JNIConfiguration.NAN) {
            throw new InvalidJNILibraryException(configuration);
        }
        if (configuration != this.parentJNIConfiguration) {
            throw new WrongJNILibraryException(parentJNIConfiguration, configuration);
        }
    }

    /**
     * The library loading configuration to used for on desktop
     */
    public abstract static class Desktop extends DeepSpeechLibraryConfig {

        Desktop(@NotNull DeepSpeech.JNIConfiguration parentJNIConfiguration) {
            super(parentJNIConfiguration);
        }

        public static class CPU extends Desktop {

            /**
             * The library loader for loading the deep speech native library
             */
            @NotNull
            private final NativeLibraryLoader deepSpeechLibraryLoader;

            /**
             * The library loader for loading the deep speech jni library
             */
            @NotNull
            private final NativeLibraryLoader deepSpeechJNILibraryLoader;

            /**
             * @param deepSpeechLibrary    the URL pointing to the deep speech library resource
             * @param deepSpeechJNILibrary the URL pointing to the deep speech jni library resource
             */
            public CPU(@NotNull URL deepSpeechLibrary, @NotNull URL deepSpeechJNILibrary) {
                super(DeepSpeech.JNIConfiguration.CPU);
                this.deepSpeechLibraryLoader = new NativeLibraryLoader(deepSpeechLibrary);
                this.deepSpeechJNILibraryLoader = new NativeLibraryLoader(deepSpeechJNILibrary);
            }

            @Override
            protected void loadDeepSpeechLibrary() throws IOException {
                deepSpeechLibraryLoader.load();
            }

            @Override
            protected void loadDeepSpeechJNI() throws IOException {
                deepSpeechJNILibraryLoader.load();
            }
        }

        /**
         * The library configuration providing cuda-support
         */
        public static class Cuda extends Desktop {


            /**
             * The library loader for loading the deep speech native library
             */
            @NotNull
            private final NativeLibraryLoader deepSpeechLibraryLoader;

            /**
             * The library loader for loading the deep speech jni library
             */
            @NotNull
            private final NativeLibraryLoader deepSpeechJNILibraryLoader;

            /**
             * The library loader for loading the cuda runtime library
             */
            @NotNull
            private final NativeLibraryLoader cudaRuntimeLibraryLoader;

            /**
             * @param deepSpeechLibrary    the URL pointing to the deep speech library resource
             * @param deepSpeechJNILibrary the URL pointing to the deep speech jni library resource
             * @param cudaRuntimeLibrary   the URL pointing to the cuda runtime library of the cuda installation
             */
            public Cuda(@NotNull URL deepSpeechLibrary, @NotNull URL deepSpeechJNILibrary, @NotNull URL cudaRuntimeLibrary) {
                super(DeepSpeech.JNIConfiguration.CUDA);
                this.deepSpeechLibraryLoader = new NativeLibraryLoader(deepSpeechLibrary);
                this.deepSpeechJNILibraryLoader = new NativeLibraryLoader(deepSpeechJNILibrary);
                this.cudaRuntimeLibraryLoader = new NativeLibraryLoader(cudaRuntimeLibrary);
            }

            @Override
            protected void preInit() throws IOException {
                this.cudaRuntimeLibraryLoader.load(); // Loading the cuda runtime library
            }

            @Override
            public void loadDeepSpeech() throws IOException {
                super.loadDeepSpeech();
                checkCudaRuntime();
            }

            @Override
            protected void loadDeepSpeechLibrary() throws IOException {
                deepSpeechLibraryLoader.load();
            }

            @Override
            protected void loadDeepSpeechJNI() throws IOException {
                deepSpeechJNILibraryLoader.load();
            }

            /**
             * Checks whether the cuda runtime is functional
             *
             * @throws CudaException if the cuda runtime is invalid
             */
            private void checkCudaRuntime() throws CudaException {
                Device device = org.mozilla.deepspeech.cuda.Cuda.getDevice();
                System.out.println("Device used for cuda runtime: \n" + device);
            }
        }
    }

    /**
     * The android library loading configuration
     */
    public static class Android extends DeepSpeechLibraryConfig {

        public Android() {
            super(DeepSpeech.JNIConfiguration.CPU);
        }

        @Override
        protected void loadDeepSpeechJNI() {
            System.loadLibrary("deepspeech-jni");
        }

        @Override
        protected void loadDeepSpeechLibrary() {
            System.loadLibrary("deepspeech");
        }
    }
}
