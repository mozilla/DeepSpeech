package org.mozilla.deepspeech.test;

import org.junit.Test;
import org.mozilla.deepspeech.libraryloader.DeepSpeechLibraryConfig;

import java.io.File;
import java.io.IOException;

import static org.mozilla.deepspeech.test.WorkingDirectory.WORKING_DIRECTORY;

/**
 * Test for the cuda configuration
 */
public class CudaTest {

    /**
     * Tests if the cuda configuration can be loaded without any exceptions.
     */
    @Test
    public void testCudaRuntime() throws IOException {
        System.out.println("When running this test, make sure the working directory of the test application is setup like this:\n\nworkingDirectory\n\tmodel\n\t\toutput_graph.pb\n\t\talphabet.txt\n\t\tlm.binary\n\tnatives\n\t\tlibdeepspeech.so\n\t\tlibdeepspeech-jni.so\n\ttestInput.wav\n");
        DeepSpeechLibraryConfig config = new DeepSpeechLibraryConfig.Desktop.Cuda(
                new File(WORKING_DIRECTORY, "natives/libdeepspeech_cuda.so").toURI().toURL(),
                new File(WORKING_DIRECTORY, "natives/libdeepspeech-jni_cuda.so").toURI().toURL(),
                new File(System.getenv("CUDA_PATH"), "lib64/libcudart.so").toURI().toURL() // Make sure CUDA_PATH is defined
        );
        config.loadDeepSpeech();
    }

}
