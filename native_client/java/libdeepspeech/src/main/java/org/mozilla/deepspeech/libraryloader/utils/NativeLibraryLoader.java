package org.mozilla.deepspeech.libraryloader.utils;

import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;

/**
 * Loads a given native libary
 */
public class NativeLibraryLoader {

    /**
     * Buffer size for IO copying
     */
    private static final int BUFFER_SIZE = 4096;

    /**
     * The URL pointing to the native library to be loaded
     */
    @NotNull
    private final URL libraryResource;

    /**
     * @param libraryResource sets {@link #libraryResource}
     */
    public NativeLibraryLoader(@NotNull URL libraryResource) {
        this.libraryResource = libraryResource;
    }

    /**
     * Reads all bytes from the input stream and writes them to the output stream.
     *
     * @param in  where the data comes from
     * @param out where the data is written to
     */
    private static void copy(@NotNull InputStream in, @NotNull OutputStream out) throws IOException {
        byte[] buf = new byte[BUFFER_SIZE];
        int n;
        while ((n = in.read(buf)) > 0) {
            out.write(buf, 0, n);
        }
    }

    /**
     * Loads the library
     */
    public void load() throws IOException {
        InputStream inputStream = libraryResource.openStream();
        File file = null;
        try {
            file = new File(libraryResource.toURI());
        } catch (URISyntaxException ignored) {
        }
        if (file == null || !file.exists()) { // If the URI points to a resource eg in an archive (which might even be the jar itself) -> extract to temp file
            file = File.createTempFile("lib", "");
            file.deleteOnExit();
            OutputStream out = new BufferedOutputStream(new FileOutputStream(file));
            copy(inputStream, out);
            out.close();
        }
        inputStream.close();
        System.load(file.getAbsolutePath());
    }

    @NotNull
    public URL getLibraryResource() {
        return libraryResource;
    }
}
