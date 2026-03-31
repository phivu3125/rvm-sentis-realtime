using System;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;

/// <summary>
/// Robust Video Matting (RVM) - Simplified single-file implementation
/// Supports MobileNetV3 (faster) and ResNet50 (higher quality) backbones
/// 
/// Features:
/// - GPU compute shader for compositing and morphology
/// - Supports webcam and video input
/// - Minimal allocations with tensor/texture caching
/// 
/// Based on: https://github.com/PeterL1n/RobustVideoMatting
/// </summary>
public class RVMCore : MonoBehaviour
{
    #region Enums
    public enum ModelType { MobileNetV3, ResNet50 }
    public enum InputSourceType { Webcam, Video }
    public enum BackendMode { GPU, CPU }
    #endregion

    #region Serialized Fields
    [Header("Model")]
    [Tooltip("RVM ONNX model (MobileNetV3 or ResNet50)")]
    public ModelAsset modelAsset;

    [Tooltip("Model backbone type - must match the loaded model")]
    public ModelType modelType = ModelType.MobileNetV3;

    [Tooltip("Backend for model inference (GPU recommended, CPU for compatibility)")]
    public BackendMode backendMode = BackendMode.GPU;

    [Header("Compute Shader")]
    [Tooltip("RVM Composite compute shader")]
    public ComputeShader compositeShader;

    [Tooltip("RVM Composite shader for CPU mode (assign RVMCompositeShader)")]
    public Shader compositeShaderCPU;

    [Header("Display")]
    [Tooltip("RawImage to display result")]
    public RawImage displayImage;

    [Header("Background")]
    public Color backgroundColor = new Color(0, 1, 0, 1);
    public Texture2D backgroundImage;

    [Header("RVM Settings")]
    [Range(0.1f, 1.0f)]
    public float downsampleRatio = 0.375f;
    public bool autoDownsampleRatio = false;

    [Header("Mask Settings")]
    [Range(0f, 1f)] public float alphaThreshold = 0.1f;
    [Range(0f, 0.5f)] public float feather = 0.05f;
    [Range(0f, 1f)] public float edgeStrength = 0.5f;
    [Range(0, 10)] public int erodePixels = 2;
    [Range(0, 10)] public int dilatePixels = 0;

    [Header("Input Source")]
    public InputSourceType inputSourceType = InputSourceType.Webcam;
    public VideoClip videoClip;
    public bool loopVideo = true;

    [Header("Webcam Settings")]
    [Tooltip("Requested webcam width (actual may vary)")]
    public int webcamWidth = 1920;
    [Tooltip("Requested webcam height (actual may vary)")]
    public int webcamHeight = 1080;
    [Tooltip("Requested webcam FPS")]
    public int webcamFPS = 30;

    [Header("Camera Flip")]
    [Tooltip("Mirror camera horizontally (selfie/mirror effect)")]
    public bool mirrorCamera = false;

    [Header("Performance")]
    [Range(1, 5)] public int processEveryNFrames = 1;
    #endregion

    #region Private Fields
    // Model
    private Worker worker;
    private Model model;

    // Input source
    private WebCamTexture webcam;
    private VideoPlayer videoPlayer;
    private RenderTexture videoRT;
    private bool isVideoReady;

    // Textures (simplified - only essential ones)
    private RenderTexture inputRT;
    private RenderTexture flippedRT;
    private RenderTexture outputRT;
    private RenderTexture phaRT;
    private RenderTexture morphTempRT;
    private RenderTexture compositeRT;

    // Recurrent states
    private Tensor r1, r2, r3, r4;
    private int lastHeight = -1, lastWidth = -1;
    private float lastDsRatio = -1f;

    // Cached tensors
    private Tensor<float> srcTensor, dsrTensor;
    private int tensorW = -1, tensorH = -1;
    private float cachedDsr = -1f;

    // Compute shader
    private int kernelComposite = -1, kernelErode = -1, kernelDilate = -1;
    private bool useComputeShader;

    // CPU compositing material
    private Material cpuCompositeMaterial;

    // Shader property IDs (cached for performance)
    private static readonly int PropSourceTex = Shader.PropertyToID("_SourceTex");
    private static readonly int PropMaskTex = Shader.PropertyToID("_MaskTex");
    private static readonly int PropBackgroundTex = Shader.PropertyToID("_BackgroundTex");
    private static readonly int PropResult = Shader.PropertyToID("_Result");
    private static readonly int PropMorphInput = Shader.PropertyToID("_MorphInput");
    private static readonly int PropMorphOutput = Shader.PropertyToID("_MorphOutput");
    private static readonly int PropSourceSize = Shader.PropertyToID("_SourceSize");
    private static readonly int PropMaskSize = Shader.PropertyToID("_MaskSize");
    private static readonly int PropMorphSize = Shader.PropertyToID("_MorphSize");
    private static readonly int PropAlphaThreshold = Shader.PropertyToID("_AlphaThreshold");
    private static readonly int PropFeather = Shader.PropertyToID("_Feather");
    private static readonly int PropEdgeStrength = Shader.PropertyToID("_EdgeStrength");
    private static readonly int PropBackgroundColor = Shader.PropertyToID("_BackgroundColor");
    private static readonly int PropUseBackgroundTex = Shader.PropertyToID("_UseBackgroundTex");
    private static readonly int PropRadius = Shader.PropertyToID("_Radius");

    // State
    private int camWidth = 1920, camHeight = 1080;
    private int modelWidth, modelHeight;
    private int frameCount;
    private bool isProcessing, isShuttingDown, isCleanedUp;
    
    #endregion

    #region Constants
    private const int THREAD_GROUP_SIZE = 8;
    private const int DIMENSION_ALIGNMENT = 64;
    #endregion

    #region Unity Lifecycle
    void Start()
    {
        if (!InitializeModel()) { enabled = false; return; }
        InitializeComputeShader();
        InitializeInputSource();
        EnsureOutputRT();
        if (displayImage != null) displayImage.texture = outputRT;
    }

    void Update()
    {
        if (isShuttingDown) return;
        if (!IsInputReady()) return;

        frameCount++;
        if (!isProcessing && frameCount % processEveryNFrames == 0)
        {
            ProcessFrame();
        }
    }

    void OnDisable() { isShuttingDown = true; Cleanup(); }
    void OnDestroy() { Cleanup(); }
    #endregion

    #region Initialization
    private bool InitializeModel()
    {
        if (modelAsset == null) { Debug.LogError("[RVM] Model asset not assigned!"); return false; }

        model = ModelLoader.Load(modelAsset);
        if (model == null) { Debug.LogError("[RVM] Failed to load model!"); return false; }

        var backend = backendMode == BackendMode.GPU ? BackendType.GPUCompute : BackendType.CPU;
        worker = new Worker(model, backend);
        Debug.Log($"[RVM] Model loaded ({modelType}). Backend: {backend}");
        return true;
    }

    private void InitializeComputeShader()
    {
        // Initialize CPU composite material (needed for fallback)
        var shader = compositeShaderCPU != null ? compositeShaderCPU : Shader.Find("Hidden/RVM/Composite");
        if (shader != null)
        {
            cpuCompositeMaterial = new Material(shader);
        }
        else
        {
            Debug.LogWarning("[RVM] CPU composite shader not assigned! Assign RVMCompositeShader in Inspector.");
        }

        // CPU mode: disable compute shader (requires GPU)
        if (backendMode == BackendMode.CPU)
        {
            Debug.Log("[RVM] CPU mode: compute shader disabled, using shader-based compositing");
            useComputeShader = false;
            return;
        }

        if (!SystemInfo.supportsComputeShaders || compositeShader == null)
        {
            Debug.LogWarning("[RVM] Compute shader not available, using shader-based fallback");
            useComputeShader = false;
            return;
        }

        try
        {
            kernelComposite = compositeShader.FindKernel("Composite");
            kernelErode = compositeShader.FindKernel("Erode");
            kernelDilate = compositeShader.FindKernel("Dilate");
            useComputeShader = true;
            Debug.Log("[RVM] GPU compute shader initialized");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[RVM] Compute shader init failed: {e.Message}");
            useComputeShader = false;
        }
    }

    private void InitializeInputSource()
    {
        if (inputSourceType == InputSourceType.Webcam)
        {
            InitializeWebcam();
        }
        else
        {
            if (videoClip == null)
            {
                Debug.LogWarning("[RVM] No video clip, using webcam");
                inputSourceType = InputSourceType.Webcam;
                InitializeWebcam();
            }
            else
            {
                InitializeVideoPlayer();
            }
        }
    }

    private void InitializeWebcam()
    {
        if (WebCamTexture.devices.Length == 0)
        {
            Debug.LogError("[RVM] No webcam found!");
            return;
        }

        webcam = new WebCamTexture(WebCamTexture.devices[0].name, webcamWidth, webcamHeight, webcamFPS)
        {
            filterMode = FilterMode.Bilinear
        };
        webcam.Play();
        Debug.Log($"[RVM] Webcam: {webcam.deviceName} ({webcam.width}x{webcam.height} @ {webcamFPS}fps requested)");
    }

    private void InitializeVideoPlayer()
    {
        videoPlayer = gameObject.AddComponent<VideoPlayer>();
        videoPlayer.playOnAwake = false;
        videoPlayer.source = VideoSource.VideoClip;
        videoPlayer.clip = videoClip;
        videoPlayer.isLooping = loopVideo;
        videoPlayer.renderMode = VideoRenderMode.RenderTexture;
        videoPlayer.audioOutputMode = VideoAudioOutputMode.AudioSource;

        int vw = Mathf.Max((int)videoClip.width, 640);
        int vh = Mathf.Max((int)videoClip.height, 480);
        videoRT = CreateRT(vw, vh, false);
        videoPlayer.targetTexture = videoRT;

        videoPlayer.prepareCompleted += (source) => { isVideoReady = true; source.Play(); };
        videoPlayer.Prepare();
    }
    #endregion

    #region Input Source Helpers
    private bool IsInputReady()
    {
        if (inputSourceType == InputSourceType.Webcam)
            return webcam != null && webcam.isPlaying && webcam.width > 16;
        else
            return isVideoReady && videoPlayer != null && videoPlayer.isPlaying && videoRT != null;
    }

    private Texture GetInputTexture()
    {
        return inputSourceType == InputSourceType.Webcam ? webcam : videoRT;
    }

    private (int width, int height) GetInputDimensions()
    {
        if (inputSourceType == InputSourceType.Webcam)
            return (webcam?.width ?? 0, webcam?.height ?? 0);
        else
            return (videoRT?.width ?? 0, videoRT?.height ?? 0);
    }
    #endregion

    #region Texture Management (Simplified)
    private RenderTexture CreateRT(int w, int h, bool enableRandomWrite = true, RenderTextureFormat format = RenderTextureFormat.DefaultHDR)
    {
        var rt = new RenderTexture(w, h, 0, format)
        {
            enableRandomWrite = enableRandomWrite,
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        rt.Create();
        return rt;
    }

    private void ReleaseRT(ref RenderTexture rt)
    {
        if (rt != null) { rt.Release(); Destroy(rt); rt = null; }
    }

    private void EnsureOutputRT()
    {
        if (outputRT == null || outputRT.width != camWidth || outputRT.height != camHeight)
        {
            ReleaseRT(ref outputRT);
            outputRT = CreateRT(camWidth, camHeight);
        }
    }

    private void EnsureTextures(int cw, int ch, int mw, int mh)
    {
        // Camera-size textures
        if (inputRT == null || inputRT.width != cw || inputRT.height != ch)
        {
            ReleaseRT(ref inputRT);
            ReleaseRT(ref flippedRT);
            ReleaseRT(ref outputRT);
            ReleaseRT(ref compositeRT);
            inputRT = CreateRT(cw, ch, false);
            flippedRT = CreateRT(cw, ch, false);
            outputRT = CreateRT(cw, ch);
            compositeRT = CreateRT(cw, ch);
        }

        // Model-size textures
        if (phaRT == null || phaRT.width != mw || phaRT.height != mh)
        {
            ReleaseRT(ref phaRT);
            ReleaseRT(ref morphTempRT);
            phaRT = CreateRT(mw, mh, true, RenderTextureFormat.RFloat);
            morphTempRT = CreateRT(mw, mh, true, RenderTextureFormat.RFloat);
        }
    }
    #endregion

    #region Processing
    private void ProcessFrame()
    {
        if (isProcessing || isShuttingDown) return;
        isProcessing = true;

        try
        {
            var inputTexture = GetInputTexture();
            if (inputTexture == null) { isProcessing = false; return; }

            // Update dimensions
            var (cw, ch) = GetInputDimensions();
            if (cw != camWidth || ch != camHeight)
            {
                camWidth = cw;
                camHeight = ch;
                ResetRecurrentStates();
            }

            // Calculate model dimensions (aligned to 64)
            modelWidth = ((camWidth + DIMENSION_ALIGNMENT - 1) / DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT;
            modelHeight = ((camHeight + DIMENSION_ALIGNMENT - 1) / DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT;

            float dsRatio = CalculateDownsampleRatio();
            InitializeRecurrentStates(modelHeight, modelWidth, dsRatio);
            EnsureTextures(camWidth, camHeight, modelWidth, modelHeight);

            // Update display if output RT was recreated
            if (displayImage != null && displayImage.texture != outputRT)
                displayImage.texture = outputRT;

            // Blit input to inputRT, then apply mirror if needed
            Graphics.Blit(inputTexture, inputRT);

            // Apply mirror transformation
            RenderTexture processRT = inputRT;
            if (mirrorCamera)
            {
                Graphics.Blit(inputRT, flippedRT, new Vector2(-1, 1), new Vector2(1, 0));
                processRT = flippedRT;
            }

            // Ensure cached tensors
            EnsureCachedTensors(modelWidth, modelHeight, dsRatio);

            // Convert texture to tensor
            TextureConverter.ToTensor(processRT, srcTensor, new TextureTransform().SetTensorLayout(TensorLayout.NCHW));

            // Set inputs and run inference
            worker.SetInput("src", srcTensor);
            worker.SetInput("r1i", r1);
            worker.SetInput("r2i", r2);
            worker.SetInput("r3i", r3);
            worker.SetInput("r4i", r4);
            worker.SetInput("downsample_ratio", dsrTensor);
            worker.Schedule();

            // Update recurrent states
            UpdateRecurrentStates();

            // Get alpha output directly to RenderTexture (GPU-only, no CPU readback)
            var phaOutput = worker.PeekOutput("pha") as Tensor<float>;
            if (phaOutput == null) { isProcessing = false; return; }

            TextureConverter.RenderToTexture(phaOutput, phaRT, new TextureTransform().SetTensorLayout(TensorLayout.NCHW));

            // Composite
            if (useComputeShader)
                CompositeGPU(processRT);
            else
                CompositeCPU(processRT);
        }
        catch (Exception e)
        {
            if (!isShuttingDown) Debug.LogError($"[RVM] Error: {e}");
        }
        finally
        {
            isProcessing = false;
        }
    }

    private float CalculateDownsampleRatio()
    {
        if (!autoDownsampleRatio) return downsampleRatio;

        int maxDim = Mathf.Max(camWidth, camHeight);
        if (maxDim <= 512) return 1.0f;
        if (maxDim <= 1280) return 0.375f;
        if (maxDim <= 1920) return 0.25f;
        return 0.125f;
    }

    private void UpdateRecurrentStates()
    {
        // CopyOutput: GPU-only copy, no CPU readback.
        // Handles disposal/reallocation internally if shape changes.
        worker.CopyOutput("r1o", ref r1);
        worker.CopyOutput("r2o", ref r2);
        worker.CopyOutput("r3o", ref r3);
        worker.CopyOutput("r4o", ref r4);
    }
    #endregion

    #region Tensor Management
    private void EnsureCachedTensors(int w, int h, float dsr)
    {
        if (srcTensor == null || tensorW != w || tensorH != h)
        {
            srcTensor?.Dispose();
            srcTensor = new Tensor<float>(new TensorShape(1, 3, h, w));
            tensorW = w;
            tensorH = h;
        }

        if (dsrTensor == null || !Mathf.Approximately(cachedDsr, dsr))
        {
            dsrTensor?.Dispose();
            dsrTensor = new Tensor<float>(new TensorShape(1), new float[] { dsr });
            cachedDsr = dsr;
        }
    }

    private void InitializeRecurrentStates(int h, int w, float dsRatio)
    {
        if (r1 != null && h == lastHeight && w == lastWidth && Mathf.Approximately(dsRatio, lastDsRatio))
            return;

        DisposeRecurrentStates();

        // Calculate state dimensions
        int eh = Mathf.CeilToInt(h * dsRatio);
        int ew = Mathf.CeilToInt(w * dsRatio);
        eh = ((eh + 1) / 2) * 2;
        ew = ((ew + 1) / 2) * 2;

        int h1 = (eh + 1) / 2, w1 = (ew + 1) / 2;
        int h2 = (h1 + 1) / 2, w2 = (w1 + 1) / 2;
        int h3 = (h2 + 1) / 2, w3 = (w2 + 1) / 2;
        int h4 = (h3 + 1) / 2, w4 = (w3 + 1) / 2;

        // Channel config based on model type
        var (c1, c2, c3, c4) = modelType == ModelType.ResNet50
            ? (16, 32, 64, 128)
            : (16, 20, 40, 64);

        r1 = CreateZeroTensor(1, c1, h1, w1);
        r2 = CreateZeroTensor(1, c2, h2, w2);
        r3 = CreateZeroTensor(1, c3, h3, w3);
        r4 = CreateZeroTensor(1, c4, h4, w4);

        lastHeight = h;
        lastWidth = w;
        lastDsRatio = dsRatio;
    }

    private Tensor<float> CreateZeroTensor(int n, int c, int h, int w)
    {
        return new Tensor<float>(new TensorShape(n, c, h, w), new float[n * c * h * w]);
    }

    private void ResetRecurrentStates()
    {
        DisposeRecurrentStates();
        lastHeight = lastWidth = -1;
        lastDsRatio = -1f;
    }

    private void DisposeRecurrentStates()
    {
        r1?.Dispose(); r2?.Dispose(); r3?.Dispose(); r4?.Dispose();
        r1 = null; r2 = null; r3 = null; r4 = null;
    }
    #endregion

    #region GPU Compositing
    private void CompositeGPU(Texture inputTexture)
    {
        RenderTexture currentMask = phaRT;

        // Apply morphology
        if (erodePixels > 0)
        {
            ApplyMorphology(currentMask, morphTempRT, erodePixels, false);
            currentMask = morphTempRT;
        }

        if (dilatePixels > 0)
        {
            if (erodePixels > 0)
            {
                Graphics.Blit(morphTempRT, phaRT);
                currentMask = phaRT; // Read from phaRT (blitted erode result), not morphTempRT
            }
            ApplyMorphology(currentMask, morphTempRT, dilatePixels, true);
            currentMask = morphTempRT;
        }

        // Composite
        compositeShader.SetTexture(kernelComposite, PropSourceTex, inputTexture);
        compositeShader.SetTexture(kernelComposite, PropMaskTex, currentMask);
        compositeShader.SetTexture(kernelComposite, PropResult, compositeRT);
        compositeShader.SetInts(PropSourceSize, camWidth, camHeight);
        compositeShader.SetInts(PropMaskSize, currentMask.width, currentMask.height);
        compositeShader.SetFloat(PropAlphaThreshold, alphaThreshold);
        compositeShader.SetFloat(PropFeather, feather);
        compositeShader.SetFloat(PropEdgeStrength, edgeStrength);
        compositeShader.SetVector(PropBackgroundColor, backgroundColor);
        compositeShader.SetInt(PropUseBackgroundTex, backgroundImage != null ? 1 : 0);
        compositeShader.SetTexture(kernelComposite, PropBackgroundTex, backgroundImage != null ? backgroundImage : inputTexture);

        int tx = Mathf.CeilToInt(camWidth / (float)THREAD_GROUP_SIZE);
        int ty = Mathf.CeilToInt(camHeight / (float)THREAD_GROUP_SIZE);
        compositeShader.Dispatch(kernelComposite, tx, ty, 1);

        Graphics.Blit(compositeRT, outputRT);
    }

    private void CompositeCPU(Texture inputTexture)
    {
        if (cpuCompositeMaterial == null)
        {
            // Fallback: just copy input if material not available
            Graphics.Blit(inputTexture, outputRT);
            return;
        }

        // Set material properties
        cpuCompositeMaterial.SetTexture("_MainTex", inputTexture);
        cpuCompositeMaterial.SetTexture("_MaskTex", phaRT);
        cpuCompositeMaterial.SetColor("_BackgroundColor", backgroundColor);
        cpuCompositeMaterial.SetFloat("_AlphaThreshold", alphaThreshold);
        cpuCompositeMaterial.SetFloat("_Feather", feather);
        cpuCompositeMaterial.SetFloat("_EdgeStrength", edgeStrength);
        cpuCompositeMaterial.SetFloat("_UseBackgroundTex", backgroundImage != null ? 1f : 0f);
        if (backgroundImage != null)
            cpuCompositeMaterial.SetTexture("_BackgroundTex", backgroundImage);

        // Composite using shader
        Graphics.Blit(inputTexture, outputRT, cpuCompositeMaterial);
    }

    private void ApplyMorphology(RenderTexture input, RenderTexture output, int radius, bool dilate)
    {
        int kernel = dilate ? kernelDilate : kernelErode;

        compositeShader.SetTexture(kernel, PropMorphInput, input);
        compositeShader.SetTexture(kernel, PropMorphOutput, output);
        compositeShader.SetInts(PropMorphSize, input.width, input.height);
        compositeShader.SetInt(PropRadius, radius);

        int tx = Mathf.CeilToInt(input.width / (float)THREAD_GROUP_SIZE);
        int ty = Mathf.CeilToInt(input.height / (float)THREAD_GROUP_SIZE);
        compositeShader.Dispatch(kernel, tx, ty, 1);
    }
    #endregion

    #region Public API
    /// <summary>
    /// Gets the current source texture (processed input before compositing)
    /// Used by RVMHybridCore to access the original video frame
    /// </summary>
    public Texture GetSourceTexture()
    {
        // Return the processed input texture (mirrored if mirrorCamera is enabled)
        if (mirrorCamera && flippedRT != null)
            return flippedRT;
        return inputRT;
    }

    /// <summary>
    /// Gets the raw WebCamTexture for direct CPU-side pixel access.
    /// Returns null if the input source is not a webcam.
    /// </summary>
    public WebCamTexture GetWebCamTexture()
    {
        return webcam;
    }

    /// <summary>
    /// Gets the RVM alpha mask texture (for external processing)
    /// </summary>
    public RenderTexture GetAlphaMaskTexture()
    {
        return phaRT;
    }

    public void SwitchInputSource(InputSourceType newType)
    {
        if (newType == inputSourceType) return;
        StopInputSource();
        ResetRecurrentStates();
        inputSourceType = newType;
        InitializeInputSource();
    }

    /// <summary>
    /// Switch to a different webcam device by name.
    /// Stops the current webcam, creates a new WebCamTexture, and plays it.
    /// </summary>
    public void SwitchCamera(string deviceName)
    {
        if (string.IsNullOrEmpty(deviceName)) return;
        if (webcam != null && webcam.deviceName == deviceName) return;

        // Stop current webcam
        if (webcam != null) { webcam.Stop(); Destroy(webcam); webcam = null; }

        // Create and start new webcam
        webcam = new WebCamTexture(deviceName, webcamWidth, webcamHeight, webcamFPS)
        {
            filterMode = FilterMode.Bilinear
        };
        webcam.Play();
        ResetRecurrentStates();
        Debug.Log($"[RVM] Switched camera to: {deviceName}");
    }

    /// <summary>
    /// Switch to a different model at runtime.
    /// Disposes current worker, loads new model, creates new worker, and runs a warmup pass.
    /// </summary>
    public void SwitchModel(ModelAsset newAsset, ModelType newType)
    {
        if (newAsset == null) return;
        if (newAsset == modelAsset && newType == modelType) return;

        // Dispose current worker and recurrent states
        worker?.Dispose(); worker = null;
        DisposeRecurrentStates();
        srcTensor?.Dispose(); srcTensor = null;
        dsrTensor?.Dispose(); dsrTensor = null;
        lastHeight = lastWidth = -1;
        lastDsRatio = -1f;
        tensorW = tensorH = -1;
        cachedDsr = -1f;

        // Load new model
        modelAsset = newAsset;
        modelType = newType;
        model = ModelLoader.Load(modelAsset);
        var backend = backendMode == BackendMode.GPU ? BackendType.GPUCompute : BackendType.CPU;
        worker = new Worker(model, backend);

        Debug.Log($"[RVM] Switched model to {modelType}. Backend: {backend}");
    }

    public void LoadVideoClip(VideoClip clip)
    {
        if (clip == null) return;
        videoClip = clip;
        if (inputSourceType == InputSourceType.Video)
        {
            StopInputSource();
            ResetRecurrentStates();
            InitializeVideoPlayer();
        }
    }

    private void StopInputSource()
    {
        if (webcam != null) { webcam.Stop(); Destroy(webcam); webcam = null; }
        if (videoPlayer != null) { videoPlayer.Stop(); Destroy(videoPlayer); videoPlayer = null; }
        ReleaseRT(ref videoRT);
        isVideoReady = false;
    }
    #endregion

    #region Cleanup
    private void Cleanup()
    {
        if (isCleanedUp) return;
        isCleanedUp = true;

        StopInputSource();
        worker?.Dispose(); worker = null;
        DisposeRecurrentStates();
        srcTensor?.Dispose(); srcTensor = null;
        dsrTensor?.Dispose(); dsrTensor = null;
        if (cpuCompositeMaterial != null) { Destroy(cpuCompositeMaterial); cpuCompositeMaterial = null; }

        ReleaseRT(ref inputRT);
        ReleaseRT(ref flippedRT);
        ReleaseRT(ref outputRT);
        ReleaseRT(ref phaRT);
        ReleaseRT(ref morphTempRT);
        ReleaseRT(ref compositeRT);
    }
    #endregion
}
