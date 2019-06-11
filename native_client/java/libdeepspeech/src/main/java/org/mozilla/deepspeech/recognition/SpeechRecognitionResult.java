package org.mozilla.deepspeech.recognition;

import org.jetbrains.annotations.NotNull;
import org.mozilla.deepspeech.doc.NativeType;
import org.mozilla.deepspeech.doc.WrappsStruct;
import org.mozilla.deepspeech.nativewrapper.DynamicStruct;
import org.mozilla.deepspeech.utils.NativeAccess;

import java.util.ArrayList;
import java.util.Collection;

import static org.mozilla.deepspeech.DeepSpeech.freeMetadata;

/**
 * Represents the entire STT output as an array of character metadata objects.
 * Stores properties like a confidence value and time stamps for each spoken character.
 *
 * @see #spokenCharacters
 */
@WrappsStruct("Metadata")
public class SpeechRecognitionResult extends DynamicStruct.InstantlyDisposed {

    /**
     * Size of the native Metadata structure
     */
    public static final long SIZE = NativeAccess.NATIVE_POINTER_SIZE + NativeAccess.NATIVE_INT_SIZE + 8; // 8 --> sizeof(double)

    /**
     * Approximated probability (confidence value) for this transcription.
     */
    private final double confidence;

    /**
     * An immutable collection of meta data about the individual characters spoken.
     *
     * @see SpokenCharacterData
     */
    @NotNull
    private final Collection<SpokenCharacterData> spokenCharacters;

    /**
     * The transcription text
     */
    @NotNull
    private final String text;

    /**
     * @param pointer the pointer to the dynamically allocated Metadata structure. Note that the memory of the pointer is freed after the copy action completes.
     */
    SpeechRecognitionResult(long pointer) {
        super(pointer, SIZE);
        long offset = 0;

        // Converting array
        {
            @NativeType("MetadataItem *")
            long metaDataItemsArray = getStructPointer(offset);
            offset += NativeAccess.NATIVE_POINTER_SIZE;
            int numItems = getStructInt(offset);
            ArrayList<SpokenCharacterData> data = new ArrayList<SpokenCharacterData>(numItems);
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < numItems; i++) {
                SpokenCharacterData charData = new SpokenCharacterData(metaDataItemsArray + (i * SpokenCharacterData.SIZE));
                data.add(charData);
                builder.append(charData.character);
            }
            this.spokenCharacters = data;
            this.text = builder.toString();
        }

        offset += 4;
        this.confidence = getStructDouble(offset);
        deallocateStruct(pointer); // Instantly de-allocated
    }

    @NotNull
    public Collection<SpokenCharacterData> getSpokenCharacters() {
        return spokenCharacters;
    }

    public double getConfidence() {
        return confidence;
    }

    public String getText() {
        return text;
    }

    @Override
    protected void deallocateStruct(long pointer) {
        freeMetadata(pointer);
    }

    @Override
    public String toString() {
        return "SpeechRecognitionResult {" +
                " confidence = " + confidence +
                ", spokenCharacters = " + spokenCharacters +
                ", text = \"" + text + "\"" +
                " }";
    }

    /**
     * Stores each individual character, along with its timing information
     */
    @WrappsStruct("MetadataItem")
    public static class SpokenCharacterData extends DynamicStruct.ImplicitlyDisposed {

        /**
         * Size of the native MetadataItem structure
         */
        public static final long SIZE = NativeAccess.NATIVE_POINTER_SIZE + NativeAccess.NATIVE_INT_SIZE + 4; // 4 --> sizeof(float)

        /**
         * The spoken character
         */
        private final char character;

        /**
         * Position of the character in units milliseconds
         */
        private final long startTime;

        /**
         * @param pointer the pointer to the dynamically allocated struct
         */
        protected SpokenCharacterData(long pointer) {
            super(pointer, SIZE);
            long offset = 0;
            this.character = NativeAccess.getNativeChar(getStructPointer(offset));
            offset += NativeAccess.NATIVE_POINTER_SIZE + NativeAccess.NATIVE_INT_SIZE;
            this.startTime = (long) (getStructFloat(offset) * 1000); // units of 20 ms * 20 = ms
        }

        public char getCharacter() {
            return character;
        }

        public long getStartTime() {
            return startTime;
        }

        @Override
        public String toString() {
            return "SpokenCharacterData {" +
                    " character = " + character +
                    ", startTime = " + startTime +
                    " }";
        }
    }
}
