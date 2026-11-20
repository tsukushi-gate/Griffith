import Foundation
import os

/// サンプル1件。bits モデルは 1 バイト 1 ビット(0/1)で MDL_INPUT_WIDTH 個、
/// 画像モデルは u8 NCHW 画像(C*H*W バイト)。どちらも ctx_infer へそのまま渡す。
struct Sample {
    let bits: [UInt8]
    let label: Int?
    let expectedScores: [Int32]?
}

/// samples.json のデコード。
/// 形式: {"input_width":N, "samples":[{"bits":"<hex>","label":int|null,"expected_scores":[...]}]}
/// bits は hex 文字列、バイト内 LSB-first(入力ビット i = バイト i/8 の bit i%8)。
/// 画像モデル(input_kind=image_u8)は bits の代わりに image(u8 CHW の hex)を持つ。
/// ESP32 側 examples/30_deploy_esp32/ と同じ規約。
struct SamplesFile: Decodable {
    struct Raw: Decodable {
        let bits: String?
        let image: String?
        let label: Int?
        let expected_scores: [Int32]?
    }
    let input_width: Int
    let samples: [Raw]

    func decode() throws -> [Sample] {
        try samples.map { raw in
            let data: [UInt8]
            if let img = raw.image {
                data = try Self.bytes(hex: img)
            } else if let bits = raw.bits {
                data = try Self.unpack(hex: bits, width: input_width)
            } else {
                throw NSError(domain: "samples", code: 6,
                              userInfo: [NSLocalizedDescriptionKey: "sample に bits/image が無い"])
            }
            return Sample(bits: data, label: raw.label, expectedScores: raw.expected_scores)
        }
    }

    /// hex 文字列 → バイト列(画像用)。
    static func bytes(hex: String) throws -> [UInt8] {
        let chars = Array(hex.utf8)
        guard chars.count % 2 == 0 else {
            throw NSError(domain: "samples", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "image の hex 長が奇数"])
        }
        func nib(_ c: UInt8) -> UInt8 {
            switch c {
            case 48...57: return c - 48
            case 97...102: return c - 87
            case 65...70: return c - 55
            default: return 0
            }
        }
        return (0..<chars.count / 2).map { nib(chars[2 * $0]) << 4 | nib(chars[2 * $0 + 1]) }
    }

    static func unpack(hex: String, width: Int) throws -> [UInt8] {
        let chars = Array(hex.utf8)
        let nbytes = (width + 7) / 8
        guard chars.count == nbytes * 2 else {
            throw NSError(domain: "samples", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "bits の長さが不正: \(hex)"])
        }
        func nib(_ c: UInt8) -> UInt8 {
            switch c {
            case 48...57: return c - 48
            case 97...102: return c - 87
            case 65...70: return c - 55
            default: return 0
            }
        }
        var bytes = [UInt8](repeating: 0, count: nbytes)
        for i in 0..<nbytes { bytes[i] = nib(chars[2 * i]) << 4 | nib(chars[2 * i + 1]) }
        return (0..<width).map { (bytes[$0 >> 3] >> UInt8($0 & 7)) & 1 }
    }
}

/// C 側のモデル記述子(gd_model)の Swift ビュー。
struct ModelInfo: Identifiable {
    let id: Int
    let name: String
    let inputWidth: Int
    let outputWidth: Int
    let ctxBytes: Int
    let numGates: Int
    let inputKind: Int      // 0=bits / 1=image_u8
    let inputBytes: Int
    var isImage: Bool { inputKind == 1 }

    static func all() -> [ModelInfo] {
        (0..<Int(gd_model_count())).compactMap { i in
            guard let m = gd_model_at(Int32(i)) else { return nil }
            return ModelInfo(id: i, name: String(cString: m.pointee.name),
                             inputWidth: Int(m.pointee.input_width),
                             outputWidth: Int(m.pointee.output_width),
                             ctxBytes: Int(m.pointee.ctx_bytes),
                             numGates: Int(m.pointee.num_gates),
                             inputKind: Int(m.pointee.input_kind),
                             inputBytes: Int(m.pointee.input_bytes))
        }
    }
}

struct SampleResult: Identifiable {
    let id: Int
    let scores: [Int32]
    let pred: Int
    let label: Int?
    let expected: [Int32]?
    let nsPerInfer: Double
    var correct: Bool? { label.map { $0 == pred } }
    var scoreHash: UInt32 {
        var h: UInt32 = 2166136261          // FNV-1a 32bit(int32 リトルエンディアン列)
        for v in scores { for k in 0..<4 { h = (h ^ UInt32((UInt32(bitPattern: v) >> (8 * UInt32(k))) & 0xff)) &* 16777619 } }
        return h
    }
    var scoreMatch: Bool? { expected.map { $0 == scores } }
}

struct RunResult {
    let model: ModelInfo
    let samples: [SampleResult]
    let wallSeconds: Double

    var labeled: [SampleResult] { samples.filter { $0.label != nil } }
    var accuracy: Double? {
        labeled.isEmpty ? nil : Double(labeled.filter { $0.correct == true }.count) / Double(labeled.count)
    }
    var exactMatchCount: Int { samples.filter { $0.scoreMatch == true }.count }
    var exactTotal: Int { samples.filter { $0.expected != nil }.count }
    var scoreMismatch: Int { exactTotal - exactMatchCount }

    private var ms: [Double] { samples.map { $0.nsPerInfer / 1e6 }.sorted() }
    var meanMs: Double { ms.isEmpty ? 0 : ms.reduce(0, +) / Double(ms.count) }
    var medianMs: Double {
        guard !ms.isEmpty else { return 0 }
        return ms.count % 2 == 1 ? ms[ms.count / 2] : (ms[ms.count / 2 - 1] + ms[ms.count / 2]) / 2
    }
    var minMs: Double { ms.first ?? 0 }
    var maxMs: Double { ms.last ?? 0 }

    static var buildConfig: String {
        #if DEBUG
        return "Debug"
        #else
        return "Release"
        #endif
    }

    /// 端末から回収しやすい 1 行 JSON(`RESULT {...}`)。
    var json: String {
        var d: [String: Any] = [
            "model": model.name,
            "n_samples": samples.count,
            "n_labeled": labeled.count,
            "n_correct": labeled.filter { $0.correct == true }.count,
            "exact_score_match": exactMatchCount,
            "exact_score_total": exactTotal,
            "score_mismatch": scoreMismatch,
            "input_kind": model.isImage ? "image_u8" : "bits",
            "input_bytes": model.inputBytes,
            "build": Self.buildConfig,
            "mean_ms": meanMs, "median_ms": medianMs, "min_ms": minMs, "max_ms": maxMs,
            "mdl_ctx_bytes": model.ctxBytes,
            "input_width": model.inputWidth,
            "output_width": model.outputWidth,
            "num_gates": model.numGates,
            "wall_seconds": wallSeconds,
            "device": ProcessInfo.processInfo.environment["SIMULATOR_DEVICE_NAME"] ?? "device",
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
        if let a = accuracy { d["accuracy"] = a }
        let data = (try? JSONSerialization.data(withJSONObject: d, options: [.sortedKeys])) ?? Data()
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}

enum Runner {
    static let log = Logger(subsystem: "com.example.griffithdemo", category: "run")

    static func loadSamples(for model: ModelInfo) throws -> [Sample] {
        guard let url = Bundle.main.url(forResource: "\(model.name)_samples", withExtension: "json") else {
            throw NSError(domain: "samples", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "\(model.name)_samples.json が Bundle に無い"])
        }
        let file = try JSONDecoder().decode(SamplesFile.self, from: Data(contentsOf: url))
        guard file.input_width == model.inputWidth else {
            throw NSError(domain: "samples", code: 3,
                          userInfo: [NSLocalizedDescriptionKey:
                            "input_width \(file.input_width) != MDL_INPUT_WIDTH \(model.inputWidth)"])
        }
        let decoded = try file.decode()
        guard decoded.allSatisfy({ $0.bits.count == model.inputBytes }) else {
            throw NSError(domain: "samples", code: 7,
                          userInfo: [NSLocalizedDescriptionKey: "入力バイト数が \(model.inputBytes) ではない"])
        }
        return decoded
    }

    /// 全サンプルを推論する(重い処理なのでメインスレッド外で呼ぶ)。
    /// 作業バッファ(MDL_CTX_BYTES)とコンテキストはここで 1 回だけ確保する。
    static func run(model: ModelInfo) throws -> RunResult {
        let samples = try loadSamples(for: model)
        guard let m = gd_model_at(Int32(model.id)) else { fatalError("model") }

        let work = UnsafeMutableRawPointer.allocate(byteCount: max(model.ctxBytes, 16), alignment: 16)
        let state = UnsafeMutableRawPointer.allocate(byteCount: max(m.pointee.state_size, 16), alignment: 16)
        defer { work.deallocate(); state.deallocate() }
        guard m.pointee.ctx_init!(state, work, max(model.ctxBytes, 16)) == 0 else {
            throw NSError(domain: "run", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "mdl_ctx_init に失敗"])
        }

        var scores = [Int32](repeating: 0, count: model.outputWidth)
        var out: [SampleResult] = []
        out.reserveCapacity(samples.count)
        let wall0 = DispatchTime.now().uptimeNanoseconds
        for (i, s) in samples.enumerated() {
            var reps: UInt64 = 0
            // 1 推論は µs 以下なので、約 200µs 回して 1 推論あたりを求める。
            let ns = s.bits.withUnsafeBufferPointer { b in
                gd_time_inference(m, state, b.baseAddress, &scores, 200_000, &reps)
            }
            guard ns >= 0 else {
                throw NSError(domain: "run", code: 5, userInfo: [NSLocalizedDescriptionKey: "推論に失敗"])
            }
            var best = 0
            for k in 1..<max(scores.count, 1) where scores[k] > scores[best] { best = k }
            out.append(SampleResult(id: i, scores: scores, pred: best, label: s.label,
                                    expected: s.expectedScores, nsPerInfer: ns))
        }
        let wall = Double(DispatchTime.now().uptimeNanoseconds - wall0) / 1e9
        let result = RunResult(model: model, samples: out, wallSeconds: wall)
        log.notice("RESULT \(result.json, privacy: .public)")
        print("RESULT \(result.json)")
        fflush(stdout)   // パイプ(devicectl --console)でも即座に出す
        return result
    }
}
