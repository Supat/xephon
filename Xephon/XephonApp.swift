import SwiftUI
import XephonLogging

@main
struct XephonApp: App {
    init() {
        AppLog.app.info("Xephon launching")
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
