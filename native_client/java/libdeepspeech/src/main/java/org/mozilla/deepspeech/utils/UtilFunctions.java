package org.mozilla.deepspeech.utils;

import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.FileNotFoundException;

/**
 * Generic utility functions
 */
public class UtilFunctions {

    /**
     * @param bytes the byte count to represent in a readable form
     * @param si    true when to use si prefixes with powers of ten - false when to use powers of two. This means. 1000 bytes = 1 KB; 1024 bytes = 1 KiB;
     * @return the human readable string representation
     */
    @NotNull
    public static String humanReadableByteCount(long bytes, boolean si) {
        int unit = si ? 1000 : 1024;
        if (bytes < unit) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(unit));
        String pre = (si ? "kMGTPE" : "KMGTPE").charAt(exp - 1) + (si ? "" : "i");
        return String.format("%.1f %sB", bytes / Math.pow(unit, exp), pre);
    }

    /**
     * Returns the same file object passed to the function, if it exists. It throws a {@link FileNotFoundException} otherwise.
     * @param file a file object
     * @return the same file object
     * @throws FileNotFoundException if the file does not exist
     */
    @NotNull
    public static File checkExists(@NotNull File file) throws FileNotFoundException {
        if (!file.exists())
            throw new FileNotFoundException("File " + file + " does not exist!");
        return file;
    }

}
