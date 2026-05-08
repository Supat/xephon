import os

public enum AppLog {
    public static let subsystem = "com.supatsaetia.xephon"

    public static let app           = Logger(subsystem: subsystem, category: "Xephon")
    public static let audio         = Logger(subsystem: subsystem, category: "Audio")
    public static let asr           = Logger(subsystem: subsystem, category: "ASR")
    public static let diarization   = Logger(subsystem: subsystem, category: "Diarization")
    public static let serAcoustic   = Logger(subsystem: subsystem, category: "SER.Acoustic")
    public static let serText       = Logger(subsystem: subsystem, category: "SER.Text")
    public static let fusion        = Logger(subsystem: subsystem, category: "Fusion")
    public static let export        = Logger(subsystem: subsystem, category: "Export")
}
