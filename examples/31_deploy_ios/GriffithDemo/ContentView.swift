import SwiftUI

struct ContentView: View {
    private let models = ModelInfo.all()
    @State private var selected = 0
    @State private var running = false
    @State private var result: RunResult?
    @State private var errorText: String?
    @State private var showSamples = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Picker("モデル", selection: $selected) {
                        ForEach(models) { m in Text(m.name).tag(m.id) }
                    }
                    .pickerStyle(.menu)
                    .disabled(running)

                    if let m = models.indices.contains(selected) ? models[selected] : nil {
                        Text("入力 \(m.isImage ? "u8 画像 \(m.inputBytes) B" : "\(m.inputWidth) bit") / 出力 \(m.outputWidth) / ゲート \(m.numGates) / ctx \(m.ctxBytes) B")
                            .font(.footnote).foregroundStyle(.secondary)
                    }

                    Button(action: run) {
                        HStack {
                            if running { ProgressView().tint(.white) }
                            Text(running ? "実行中..." : "実行").bold()
                        }
                        .frame(maxWidth: .infinity, minHeight: 44)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(running || models.isEmpty)
                    .accessibilityIdentifier("runButton")

                    if let e = errorText {
                        Text(e).foregroundStyle(.red).font(.callout)
                    }
                    if let r = result {
                        ResultCard(result: r)
                        DisclosureGroup("サンプルごとの結果 (\(r.samples.count))", isExpanded: $showSamples) {
                            LazyVStack(alignment: .leading, spacing: 4) {
                                ForEach(r.samples.prefix(300)) { s in SampleRow(s: s) }
                                if r.samples.count > 300 {
                                    Text("先頭 300 件のみ表示").font(.caption).foregroundStyle(.secondary)
                                }
                            }
                            .padding(.top, 6)
                        }
                        ShareLink(item: r.json) { Label("結果 JSON を共有", systemImage: "square.and.arrow.up") }
                            .font(.footnote)
                    }
                }
                .padding(16)
            }
            .navigationTitle("Griffith Demo")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onAppear(perform: autoRunIfRequested)
    }

    private func run() {
        guard models.indices.contains(selected) else { return }
        let m = models[selected]
        running = true
        errorText = nil
        Task.detached(priority: .userInitiated) {
            do {
                let r = try Runner.run(model: m)
                await MainActor.run { result = r; running = false }
            } catch {
                await MainActor.run { errorText = error.localizedDescription; running = false }
            }
        }
    }

    /// `-gd_autorun 1 [-gd_model xor]` で起動すると自動実行する(CI/スクリーンショット用)。
    private func autoRunIfRequested() {
        let d = UserDefaults.standard
        guard d.bool(forKey: "gd_autorun") else { return }
        if let name = d.string(forKey: "gd_model"), let i = models.firstIndex(where: { $0.name == name }) {
            selected = i
        }
        showSamples = d.bool(forKey: "gd_expand")
        run()
    }
}

private struct ResultCard: View {
    let result: RunResult

    private func fmt(_ ms: Double) -> String {
        let ns = ms * 1e6
        if ns < 1_000 { return String(format: "%.1f ns", ns) }
        if ns < 1_000_000 { return String(format: "%.2f µs", ns / 1e3) }
        return String(format: "%.3f ms", ms)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("結果: \(result.model.name)").font(.headline)
            row("サンプル数", "\(result.samples.count)")
            if let a = result.accuracy {
                row("正解率", String(format: "%.2f%%  (%d / %d)", a * 100,
                                  result.labeled.filter { $0.correct == true }.count, result.labeled.count))
            } else {
                row("正解率", "—(ラベル無し)")
            }
            if result.exactTotal > 0 {
                row("スコア整数一致", "\(result.exactMatchCount) / \(result.exactTotal)")
            }
            Divider()
            row("平均 / 枚", fmt(result.meanMs))
            row("中央値 / 枚", fmt(result.medianMs))
            row("最小 / 最大", "\(fmt(result.minMs)) / \(fmt(result.maxMs))")
            Divider()
            row("MDL_CTX_BYTES", "\(result.model.ctxBytes) B")
            row("入力", result.model.isImage
                ? "u8 画像 \(result.model.inputBytes) B"
                : "\(result.model.inputWidth) bit")
            row("ゲート数", "\(result.model.numGates)")
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(k).foregroundStyle(.secondary)
            Spacer()
            Text(v).font(.system(.body, design: .monospaced)).fontWeight(.medium)
        }
    }
}

private struct SampleRow: View {
    let s: SampleResult

    var body: some View {
        HStack {
            Text("#\(s.id)").foregroundStyle(.secondary).frame(width: 44, alignment: .leading)
            Text(s.label != nil ? "pred \(s.pred)  exp \(s.label.map(String.init) ?? "-")"
                                : "argmax \(s.pred)  hash \(String(s.scoreHash, radix: 16))")
            Spacer()
            if let ok = s.correct {
                Text(ok ? "OK" : "NG").foregroundStyle(ok ? .green : .red).bold()
            } else if let ok = s.scoreMatch {
                Text(ok ? "一致" : "不一致").foregroundStyle(ok ? .green : .red).bold()
            }
        }
        .font(.system(.footnote, design: .monospaced))
    }
}
