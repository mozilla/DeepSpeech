--signature_name=serving_default
                        Signature of the SavedModel Graph or TF-Hub module to
                        load. Applicable only if input format is "tf_hub" or
                        "tf_saved_model".
--saved_model_tags=serve
                        Tags of the MetaGraphDef to load, in comma separated
                        string format. Defaults to "serve". Applicable only if
                        input format is "tf_saved_model".
