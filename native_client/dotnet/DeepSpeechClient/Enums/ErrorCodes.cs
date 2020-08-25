namespace DeepSpeechClient.Enums
{
    /// <summary>
    /// Error codes from the native DeepSpeech binary.
    /// </summary>
    internal enum ErrorCodes
    {
        // OK
        DS_ERR_OK = 0x0000,

        // Missing invormations
        DS_ERR_NO_MODEL = 0x1000,

        // Invalid parameters
        DS_ERR_INVALID_ALPHABET = 0x2000,
        DS_ERR_INVALID_SHAPE = 0x2001,
        DS_ERR_INVALID_SCORER = 0x2002,
        DS_ERR_MODEL_INCOMPATIBLE = 0x2003,
        DS_ERR_SCORER_NOT_ENABLED = 0x2004,

        // Runtime failures
        DS_ERR_FAIL_INIT_MMAP = 0x3000,
        DS_ERR_FAIL_INIT_SESS = 0x3001,
        DS_ERR_FAIL_INTERPRETER = 0x3002,
        DS_ERR_FAIL_RUN_SESS = 0x3003,
        DS_ERR_FAIL_CREATE_STREAM = 0x3004,
        DS_ERR_FAIL_READ_PROTOBUF = 0x3005,
        DS_ERR_FAIL_CREATE_SESS = 0x3006,
    }
}
