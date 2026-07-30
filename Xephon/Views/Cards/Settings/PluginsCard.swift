import SwiftUI

/// Installed-plugins list on the Settings page: one row per plugin
/// with an enable/disable toggle. Only rendered when at least one
/// plugin is installed (the registry ships empty until the first
/// real plugin), so plugin-free builds show no trace of the
/// mechanism.
///
/// Toggling takes effect immediately — the registry activates or
/// drops the plugin's handle, which adds/removes its control-pane
/// pages and menu items live. A disabled plugin's saved data is
/// untouched (payloads still round-trip through `.xph` opaquely).
struct PluginsCard: View {
    let registry: PluginRegistry

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(String(localized: "plugins.header"))
                .font(.caption.bold())
                .foregroundStyle(.secondary)
            Text(String(localized: "plugins.subtitle"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
            ForEach(registry.entries) { entry in
                HStack(spacing: 8) {
                    Text(entry.properName)
                        .font(.caption)
                        .lineLimit(1)
                    Spacer(minLength: 8)
                    Toggle(
                        String(localized: "plugins.enabled"),
                        isOn: Binding(
                            get: { registry.isEnabled(entry.pluginType.id) },
                            set: { registry.setEnabled($0, id: entry.pluginType.id) }
                        )
                    )
                    .labelsHidden()
                }
                .padding(.vertical, 2)
                if entry.id != registry.entries.last?.id {
                    Divider()
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}
