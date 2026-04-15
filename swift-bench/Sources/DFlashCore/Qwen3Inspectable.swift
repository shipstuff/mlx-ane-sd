// Vendored from mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3.swift, modified to:
//  1. Rename all classes to Qwen3Insp* (avoid conflict with mlx-swift-lm exports)
//  2. Expose `layers` as public so we can iterate from outside
//  3. Add `forwardCapturing(hiddenStates:captureAt:)` that captures hidden
//     states at specified layer indices (0-based, after the layer's residual+MLP).
//     Matches the semantic our DFlash draft needs (target_layer_ids).

import Foundation
import MLX
import MLXFast
import MLXLLM       // for LLMModel protocol
import MLXLMCommon  // for LanguageModel, KVCache, applyRotaryPosition, attentionWithCacheUpdate
import MLXNN

class Qwen3InspAttention: Module {
    let args: Qwen3InspConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: Qwen3InspConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class Qwen3InspMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

public class Qwen3InspTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen3InspAttention
    let mlp: Qwen3InspMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen3InspConfiguration) {
        _attention.wrappedValue = Qwen3InspAttention(args)
        self.mlp = Qwen3InspMLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

public class Qwen3InspModelInner: Module {
    // embedTokens public so SD loop can call it for noise_embedding construction.
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding

    // layers public so we can iterate from outside (key change vs upstream).
    public let layers: [Qwen3InspTransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen3InspConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in Qwen3InspTransformerBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }

    /// Forward with hidden state capture at specified 0-based layer indices.
    /// Captures the output of each specified layer (post residual + post MLP).
    /// DFlash needs the raw hidden (not lm_head logits), so capture BEFORE the final norm.
    public func forwardCapturing(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        captureAt: [Int]
    ) -> (finalHidden: MLXArray, captures: [MLXArray]) {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        let captureSet = Set(captureAt)
        var capturedByIndex: [Int: MLXArray] = [:]

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            if captureSet.contains(i) {
                capturedByIndex[i] = h
            }
        }
        let captures = captureAt.map { capturedByIndex[$0]! }
        return (norm(h), captures)
    }
}

public class Qwen3InspModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3InspModelInner
    let configuration: Qwen3InspConfiguration

    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    public init(_ args: Qwen3InspConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3InspModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Forward + lm_head with hidden state capture.
    public func forwardCapturing(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        captureAt: [Int]
    ) -> (logits: MLXArray, captures: [MLXArray]) {
        let (hidden, captures) = model.forwardCapturing(inputs, cache: cache, captureAt: captureAt)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(hidden)
        } else {
            logits = model.embedTokens.asLinear(hidden)
        }
        return (logits, captures)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        return weights
    }

    // LoRA conformance (we don't use LoRA but LLMModel requires it).
    public var loraLayers: [Module] {
        model.layers
    }
}

public struct Qwen3InspConfiguration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var headDim: Int
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false
    var maxPositionEmbeddings: Int = 32768

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
    }
}
