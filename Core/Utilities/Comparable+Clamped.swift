public extension Comparable {
    /// Reads as intent at the call site — `value.clamped(to: 0...1)`
    /// beats `min(max(value, 0), 1)` for grep, review, and locale-
    /// independence (no confusion about which arg is the lower
    /// bound when the literals are non-zero).
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
