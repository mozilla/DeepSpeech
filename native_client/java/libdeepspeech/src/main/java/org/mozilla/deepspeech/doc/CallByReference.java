package org.mozilla.deepspeech.doc;

import java.lang.annotation.*;

/**
 * Documents a call by reference usage of a parameter eg. storage of an integer into an IntBuffer that is passed to a method.
 */
@Documented
@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.SOURCE)
public @interface CallByReference {
}
