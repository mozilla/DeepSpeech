package org.mozilla.deepspeech.doc;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * An annotation indicating what native structure type a wrapper object manages or represents
 *
 * @see org.mozilla.deepspeech.nativewrapper.DynamicStruct
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.SOURCE)
public @interface WrappsStruct {
    /**
     * The name of the native structure that the wrapper object manages or represents
     */
    String value();
}
