# H.265 Gaussian Splatting Decoder Guide (WebGPU + WebCodecs)

This document describes how to decode and render 4DGS (4D Gaussian Splatting) data that has been encoded as H.265 video streams. Use this to build the browser-side playback pipeline.

## File Format

A converted sequence is a directory served over HTTP:

```
fish-2-h265/
├── manifest.json          # metadata + quantization bounds
├── stream_position.mp4    # H.265: position high+low bytes (3264×2176)
├── stream_motion.mp4      # H.265: rotation + scale XY (3264×2176)
└── stream_appearance.mp4  # H.265: scale ZW + DC color (5440×1088)
```

### manifest.json

```json
{
  "version": 1,
  "format": "4dgs-h265",
  "sequenceName": "fish-2",
  "frameCount": 480,
  "targetFPS": 24,
  "duration": 20.0,
  "shDegree": 0,
  "gridWidth": 1088,
  "gridHeight": 1088,
  "gaussianCount": 1179648,
  "requiredCodec": "hev1.1.6.L150.B0",
  "coordinateSpace": "colmap",
  "quaternionOrder": "wxyz",
  "quantization": {
    "position": {
      "precision": "uint16",
      "min": [-2.31, -1.85, -3.02],
      "max": [2.45, 1.92, 2.88]
    },
    "rotation": {
      "precision": "uint8",
      "min": [-1.0, -1.0, -1.0, -1.0],
      "max": [1.0, 1.0, 1.0, 1.0]
    },
    "scaleOpacity": {
      "precision": "uint8",
      "min": [0.0001, 0.0001, 0.0001, 0.0],
      "max": [0.5, 0.5, 0.5, 1.0]
    },
    "shDC": {
      "precision": "uint8",
      "min": [-3.0, -3.0, -3.0],
      "max": [3.0, 3.0, 3.0]
    }
  },
  "streams": {
    "position":   { "file": "stream_position.mp4",   "width": 3264, "height": 2176, "channels": 6 },
    "motion":     { "file": "stream_motion.mp4",     "width": 3264, "height": 2176, "channels": 6 },
    "appearance": { "file": "stream_appearance.mp4", "width": 5440, "height": 1088, "channels": 5 }
  }
}
```

## Overall Decode Pipeline

```
1. fetch manifest.json → parse
2. fetch 3 × MP4 files
3. demux MP4 → H.265 encoded chunks (use mp4box.js)
4. feed chunks to 3 × WebCodecs VideoDecoder
5. VideoDecoder output → VideoFrame
6. copyExternalImageToTexture → WebGPU GPUTexture (grayscale tiled image)
7. compute shader: untile channels + dequantize float values
8. output: per-gaussian position, rotation, scale, opacity, color
9. feed to gaussian splat renderer
```

## Step 1: Fetch & Parse Manifest

```typescript
const baseUrl = "http://localhost:8080"; // or CDN URL
const manifest = await fetch(`${baseUrl}/manifest.json`).then(r => r.json());

const { gridWidth, gridHeight, gaussianCount, targetFPS, quantization, streams } = manifest;
```

## Step 2: Demux MP4 with mp4box.js

Install: `npm install mp4box`

MP4 files contain H.265 bitstream. WebCodecs VideoDecoder needs individual encoded chunks (EncodedVideoChunk), not raw MP4 bytes. Use mp4box.js to demux.

```typescript
import MP4Box from "mp4box";

interface DecodedFrame {
  frameIndex: number;
  texture: GPUTexture;
}

async function createStreamDecoder(
  device: GPUDevice,
  streamUrl: string,
  width: number,
  height: number,
  codec: string,
  onFrame: (frame: DecodedFrame) => void
) {
  let frameIndex = 0;

  // Create VideoDecoder
  const decoder = new VideoDecoder({
    output: (videoFrame: VideoFrame) => {
      // Create texture and copy VideoFrame into it
      const texture = device.createTexture({
        size: [width, height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.RENDER_ATTACHMENT,
      });

      device.queue.copyExternalImageToTexture(
        { source: videoFrame },
        { texture },
        [width, height]
      );

      videoFrame.close();
      onFrame({ frameIndex: frameIndex++, texture });
    },
    error: (e) => console.error("VideoDecoder error:", e),
  });

  // Check codec support
  const support = await VideoDecoder.isConfigSupported({
    codec,
    codedWidth: width,
    codedHeight: height,
  });
  if (!support.supported) {
    throw new Error(`Codec ${codec} at ${width}x${height} not supported`);
  }

  // Demux MP4 with mp4box.js
  const mp4box = MP4Box.createFile();

  mp4box.onReady = (info: any) => {
    const track = info.videoTracks[0];

    // Configure decoder with codec description from MP4
    decoder.configure({
      codec,
      codedWidth: width,
      codedHeight: height,
      description: track.codec_private?.data
        ? new Uint8Array(track.codec_private.data)
        : undefined,
    });

    // Start extracting samples
    mp4box.setExtractionOptions(track.id, null, { nbSamples: 100 });
    mp4box.start();
  };

  mp4box.onSamples = (_trackId: number, _user: any, samples: any[]) => {
    for (const sample of samples) {
      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? "key" : "delta",
        timestamp: sample.cts * 1_000_000 / sample.timescale, // microseconds
        duration: sample.duration * 1_000_000 / sample.timescale,
        data: sample.data,
      });
      decoder.decode(chunk);
    }
  };

  // Fetch and feed MP4 data
  const response = await fetch(streamUrl);
  const reader = response.body!.getReader();
  let offset = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const buffer = value.buffer as ArrayBuffer;
    (buffer as any).fileStart = offset;
    mp4box.appendBuffer(buffer);
    offset += buffer.byteLength;
  }

  mp4box.flush();
  await decoder.flush();

  return decoder;
}
```

## Step 3: Synchronize 3 Streams

All 3 decoders must produce frame N before you can render frame N.

```typescript
class FrameSynchronizer {
  private frames: Map<string, Map<number, GPUTexture>> = new Map();
  private streamNames: string[];
  private onFrameReady: (frameIndex: number, textures: Map<string, GPUTexture>) => void;

  constructor(
    streamNames: string[],
    onFrameReady: (frameIndex: number, textures: Map<string, GPUTexture>) => void
  ) {
    this.streamNames = streamNames;
    this.onFrameReady = onFrameReady;
    for (const name of streamNames) {
      this.frames.set(name, new Map());
    }
  }

  addFrame(streamName: string, frameIndex: number, texture: GPUTexture) {
    this.frames.get(streamName)!.set(frameIndex, texture);
    this.tryEmit(frameIndex);
  }

  private tryEmit(frameIndex: number) {
    const textures = new Map<string, GPUTexture>();
    for (const name of this.streamNames) {
      const tex = this.frames.get(name)!.get(frameIndex);
      if (!tex) return; // not all streams ready yet
      textures.set(name, tex);
    }

    // All streams have this frame — emit
    for (const name of this.streamNames) {
      this.frames.get(name)!.delete(frameIndex);
    }
    this.onFrameReady(frameIndex, textures);
  }
}
```

## Step 4: Dequantize Compute Shader (WGSL)

The decoded textures are **tiled grayscale images** where multiple attribute channels are laid out side by side. The compute shader needs to:

1. **Untile**: Read the correct pixel from the tiled layout
2. **Dequantize**: Convert uint8 (0-255) back to float using min/max bounds
3. **Reconstruct position**: Combine high + low bytes into uint16, then to float

### Tiling Layouts

**stream_position (3264×2176):**
```
Row 0 (y < gridH):      [posHi_X | posHi_Y | posHi_Z]   each gridW=1088 wide
Row 1 (y >= gridH):     [posLo_X | posLo_Y | posLo_Z]   each gridW=1088 wide
```

**stream_motion (3264×2176):**
```
Row 0 (y < gridH):      [rot_W | rot_X | rot_Z]   each gridW wide
Row 1 (y >= gridH):     [rot_W | so_X  | so_Y ]   each gridW wide
```

Note: rotation quaternion order in the tiled image is columns [0,1,2] in row 0 and column [0] in row 1. The quaternion order stored is (W, X, Y, Z) matching PLY convention where rot_0=W.

Specifically:
- Row 0, col 0: rotation component 0 (= rot[:, 0] from encoder = W)
- Row 0, col 1: rotation component 1 (= rot[:, 1] = X)
- Row 0, col 2: rotation component 2 (= rot[:, 2] = Y... wait)

Actually, let me be more precise. The encoder tiles like this:
```python
# tile_stream_motion:
row0 = [rot[:, 0], rot[:, 1], rot[:, 2]]  # first 3 rotation components
row1 = [rot[:, 3], so[:, 0], so[:, 1]]    # 4th rotation + scaleOpacity X,Y
```

The rotation in the encoder is `(W, X, Y, Z)` order (PLY convention: rot_0=W, rot_1=X, rot_2=Y, rot_3=Z). So:
- Row 0, col 0: W
- Row 0, col 1: X
- Row 0, col 2: Y
- Row 1, col 0: Z

**stream_appearance (5440×1088):**
```
Row 0: [so_Z | so_W | shDC_R | shDC_G | shDC_B]   each gridW wide
```

Where so_Z = scale component 2, so_W = opacity (4th component of scaleOpacity).

### Position Dequantize Shader

```wgsl
struct QuantBounds {
    posMin: vec3f,
    _pad0: f32,
    posMax: vec3f,
    _pad1: f32,
    rotMin: vec4f,
    rotMax: vec4f,
    soMin: vec4f,
    soMax: vec4f,
    shMin: vec3f,
    _pad2: f32,
    shMax: vec3f,
    _pad3: f32,
}

@group(0) @binding(0) var positionTex: texture_2d<f32>;   // stream_position decoded
@group(0) @binding(1) var motionTex: texture_2d<f32>;     // stream_motion decoded
@group(0) @binding(2) var appearanceTex: texture_2d<f32>; // stream_appearance decoded
@group(0) @binding(3) var<uniform> bounds: QuantBounds;
@group(0) @binding(4) var<uniform> gridSize: vec2u;       // (gridWidth, gridHeight)

// Output buffers — one vec4 per gaussian
@group(0) @binding(5) var<storage, read_write> outPosition: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> outRotation: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> outScaleOpacity: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> outColor: array<vec4f>;

@compute @workgroup_size(256)
fn dequantize(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let gw = gridSize.x;
    let gh = gridSize.y;
    let totalGaussians = gw * gh;
    if (idx >= totalGaussians) { return; }

    // Grid position for this gaussian
    let gy = idx / gw;
    let gx = idx % gw;

    // --- Position: reconstruct uint16 from high + low bytes ---
    var pos: vec3f;
    for (var ch = 0u; ch < 3u; ch++) {
        let tiled_x = i32(gx + ch * gw);
        // Row 0 = high bytes, Row 1 (offset by gridH) = low bytes
        let high = textureLoad(positionTex, vec2i(tiled_x, i32(gy)), 0).r;
        let low  = textureLoad(positionTex, vec2i(tiled_x, i32(gy + gh)), 0).r;

        // VideoFrame delivers normalized [0,1] floats from uint8 Y channel
        let u16val = u32(high * 255.0 + 0.5) << 8u | u32(low * 255.0 + 0.5);
        let norm = f32(u16val) / 65535.0;
        pos[ch] = bounds.posMin[ch] + norm * (bounds.posMax[ch] - bounds.posMin[ch]);
    }

    // --- Rotation: 4 components from motion stream ---
    // Row 0: [W, X, Y] at columns [0, 1, 2] * gridW
    // Row 1: [Z] at column [0] * gridW
    var rot: vec4f;
    for (var ch = 0u; ch < 3u; ch++) {
        let tiled_x = i32(gx + ch * gw);
        let val = textureLoad(motionTex, vec2i(tiled_x, i32(gy)), 0).r;
        let norm = val * 255.0 + 0.5;
        let q = f32(u32(norm)) / 255.0;
        rot[ch] = bounds.rotMin[ch] + q * (bounds.rotMax[ch] - bounds.rotMin[ch]);
    }
    {
        let val = textureLoad(motionTex, vec2i(i32(gx), i32(gy + gh)), 0).r;
        let norm = val * 255.0 + 0.5;
        let q = f32(u32(norm)) / 255.0;
        rot[3] = bounds.rotMin[3] + q * (bounds.rotMax[3] - bounds.rotMin[3]);
    }
    // Renormalize quaternion (W, X, Y, Z)
    rot = normalize(rot);

    // --- ScaleOpacity: 4 components split across motion + appearance ---
    var so: vec4f;
    // so_X from motion row1, col1
    {
        let val = textureLoad(motionTex, vec2i(i32(gx + gw), i32(gy + gh)), 0).r;
        so[0] = bounds.soMin[0] + (f32(u32(val * 255.0 + 0.5)) / 255.0) * (bounds.soMax[0] - bounds.soMin[0]);
    }
    // so_Y from motion row1, col2
    {
        let val = textureLoad(motionTex, vec2i(i32(gx + 2u * gw), i32(gy + gh)), 0).r;
        so[1] = bounds.soMin[1] + (f32(u32(val * 255.0 + 0.5)) / 255.0) * (bounds.soMax[1] - bounds.soMin[1]);
    }
    // so_Z from appearance row0, col0
    {
        let val = textureLoad(appearanceTex, vec2i(i32(gx), i32(gy)), 0).r;
        so[2] = bounds.soMin[2] + (f32(u32(val * 255.0 + 0.5)) / 255.0) * (bounds.soMax[2] - bounds.soMin[2]);
    }
    // so_W (opacity) from appearance row0, col1
    {
        let val = textureLoad(appearanceTex, vec2i(i32(gx + gw), i32(gy)), 0).r;
        so[3] = bounds.soMin[3] + (f32(u32(val * 255.0 + 0.5)) / 255.0) * (bounds.soMax[3] - bounds.soMin[3]);
    }

    // --- SH DC Color: 3 components from appearance ---
    var color: vec3f;
    for (var ch = 0u; ch < 3u; ch++) {
        let tiled_x = i32(gx + (ch + 2u) * gw);  // columns 2, 3, 4
        let val = textureLoad(appearanceTex, vec2i(tiled_x, i32(gy)), 0).r;
        color[ch] = bounds.shMin[ch] + (f32(u32(val * 255.0 + 0.5)) / 255.0) * (bounds.shMax[ch] - bounds.shMin[ch]);
    }

    // --- Coordinate Transform: COLMAP → Rendering Space ---
    // The data is stored in COLMAP coordinates (x, y, z).
    // Transform to your rendering coordinate system here.
    // For WebGPU (right-handed, Y-up), a common transform is:
    //   renderPos = vec3f(pos.x, -pos.y, -pos.z)
    // Adjust based on your renderer's convention.

    // Quaternion is (W, X, Y, Z) in COLMAP space.
    // Apply the same coordinate rotation to the quaternion if needed.

    // --- Write outputs ---
    outPosition[idx] = vec4f(pos, 1.0);
    outRotation[idx] = rot;  // (W, X, Y, Z)
    outScaleOpacity[idx] = so;  // (scale_x, scale_y, scale_z, opacity)
    outColor[idx] = vec4f(color, 1.0);  // (SH_dc_r, SH_dc_g, SH_dc_b, 1)
}
```

### Important Notes on VideoFrame → Texture

When `copyExternalImageToTexture` copies a VideoFrame to an `rgba8unorm` texture:
- The H.265 stream is YUV420p with data only in Y channel (U/V are neutral 128)
- The browser's color conversion will put the Y value into R, G, B channels (all equal since it's grayscale)
- **Use `.r` channel** when reading — it contains your data as a normalized [0, 1] float
- Value 0 in texture = byte value 0, value 1.0 = byte value 255

### Dequantization Formula

For uint8 attributes:
```
float_value = min + (uint8_value / 255.0) * (max - min)
```

For uint16 position (from high + low uint8):
```
uint16_value = (high_byte << 8) | low_byte
float_value = min + (uint16_value / 65535.0) * (max - min)
```

In the shader, since we get normalized floats from texture:
```
uint8_value = round(texture_r * 255.0)
uint16_value = (round(high_r * 255.0) << 8) | round(low_r * 255.0)
```

## Step 5: Per-Frame Render Loop

```typescript
class GaussianPlayer {
  private sync: FrameSynchronizer;
  private frameQueue: Map<number, Map<string, GPUTexture>> = new Map();
  private currentFrame = 0;
  private startTime = 0;

  async load(baseUrl: string, device: GPUDevice) {
    const manifest = await fetch(`${baseUrl}/manifest.json`).then(r => r.json());

    this.sync = new FrameSynchronizer(
      ["position", "motion", "appearance"],
      (frameIndex, textures) => {
        this.frameQueue.set(frameIndex, textures);
      }
    );

    // Start 3 decoders
    for (const [name, info] of Object.entries(manifest.streams) as any) {
      createStreamDecoder(
        device,
        `${baseUrl}/${info.file}`,
        info.width,
        info.height,
        manifest.requiredCodec,
        (frame) => this.sync.addFrame(name, frame.frameIndex, frame.texture)
      );
    }

    // Create dequantize pipeline, bind groups, output buffers...
    // (set up compute pipeline with the WGSL shader above)

    this.startTime = performance.now();
  }

  renderFrame(device: GPUDevice) {
    const elapsed = (performance.now() - this.startTime) / 1000;
    const targetFrame = Math.floor(elapsed * manifest.targetFPS) % manifest.frameCount;

    const textures = this.frameQueue.get(targetFrame);
    if (!textures) return; // frame not ready yet, skip

    // 1. Bind the 3 decoded textures
    // 2. Run dequantize compute shader
    // 3. Render gaussians from output buffers

    // Clean up old frame textures
    for (const tex of textures.values()) {
      tex.destroy();
    }
    this.frameQueue.delete(targetFrame);
  }
}
```

## Coordinate System

The encoded data is in **COLMAP coordinate space**:
- Position: (x, y, z) in COLMAP world coordinates
- Rotation: quaternion (W, X, Y, Z) — W is the scalar part

The playback shader must transform to your renderer's coordinate system. The original UE5 plugin uses this transform:
- Position: `(z, x, -y) * 100` (COLMAP → UE)
- Rotation: `(qZ, qX, -qY, qW)` with normalization

For WebGPU, adjust based on your camera/scene conventions.

## Attribute Summary (Per Gaussian)

| Attribute | Components | Range (after dequant) | Notes |
|-----------|-----------|----------------------|-------|
| Position | 3 (x, y, z) | from manifest min/max | COLMAP coords |
| Rotation | 4 (W, X, Y, Z) | [-1, 1] | Unit quaternion, renormalize after dequant |
| Scale | 3 (sx, sy, sz) | from manifest min/max | Already exp-activated |
| Opacity | 1 | [0, 1] | Already sigmoid-activated |
| SH DC Color | 3 (r, g, b) | from manifest min/max | DC spherical harmonics coefficient |

## Dependencies

- **mp4box.js** — MP4 demuxing: `npm install mp4box`
- **WebCodecs API** — Chrome 107+, Edge 107+
- **WebGPU** — Chrome 113+, Edge 113+

## Testing

For local development:
```bash
# Serve the encoded directory
python -m http.server 8080 --directory D:/4dgs-data/fish-2-h265

# Then open your WebGPU app pointing to http://localhost:8080/manifest.json
```

The test dataset `fish-2-h265` has:
- 480 frames at 24 FPS (20 seconds)
- 1,179,648 gaussians per frame
- Grid: 1088×1088
- Total ~482 MB (3 MP4 files)
