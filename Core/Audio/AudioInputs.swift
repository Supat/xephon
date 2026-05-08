import Foundation

public struct AudioInputDescription: Sendable, Identifiable, Hashable {
    public let uid: String
    public let displayName: String
    public let kind: Kind

    public var id: String { uid }

    public enum Kind: String, Sendable, Hashable, CaseIterable {
        case builtInMic
        case wiredHeadset
        case bluetooth
        case usb
        case airPlay
        case carPlay
        case other
    }

    public init(uid: String, displayName: String, kind: Kind) {
        self.uid = uid
        self.displayName = displayName
        self.kind = kind
    }
}
