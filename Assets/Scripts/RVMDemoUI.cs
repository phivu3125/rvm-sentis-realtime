using System.Collections.Generic;
using TMPro;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// Simple demo UI for RVM: camera picker dropdown, model switcher dropdown, and diagnostics text.
/// Attach to a GameObject in the scene and assign references in the Inspector.
/// </summary>
public class RVMDemoUI : MonoBehaviour
{
    [Header("References")]
    [Tooltip("The RVMCore component to control")]
    public RVMCore rvmCore;

    [Header("Camera Selection")]
    [Tooltip("Dropdown to select webcam device")]
    public TMP_Dropdown cameraDropdown;

    [Header("Mirror")]
    [Tooltip("Toggle to mirror/flip camera horizontally")]
    public UnityEngine.UI.Toggle mirrorToggle;

    [Header("Model Selection")]
    [Tooltip("Dropdown to select RVM model variant")]
    public TMP_Dropdown modelDropdown;

    [Header("Model Assets (assign all 4 ONNX models)")]
    public ModelAsset mobilenetV3_fp16;
    public ModelAsset mobilenetV3_fp32;
    public ModelAsset resnet50_fp16;
    public ModelAsset resnet50_fp32;

    [Header("Diagnostics")]
    [Tooltip("Text to display current camera, model, resolution, FPS")]
    public TMP_Text diagnosticsText;

    // Internal state
    private readonly List<string> cameraDeviceNames = new();
    private readonly List<ModelEntry> modelEntries = new();
    private float fpsTimer;
    private int fpsFrameCount;
    private float currentFPS;
    private bool isSwitching;

    private struct ModelEntry
    {
        public string displayName;
        public ModelAsset asset;
        public RVMCore.ModelType modelType;
    }

    void Start()
    {
        BuildModelList();
        PopulateModelDropdown();
        PopulateCameraDropdown();

        if (cameraDropdown != null)
            cameraDropdown.onValueChanged.AddListener(OnCameraSelected);
        if (modelDropdown != null)
            modelDropdown.onValueChanged.AddListener(OnModelSelected);
        if (mirrorToggle != null)
        {
            mirrorToggle.isOn = rvmCore != null && rvmCore.mirrorCamera;
            mirrorToggle.onValueChanged.AddListener(OnMirrorToggled);
        }
    }

    void Update()
    {
        UpdateFPS();
        UpdateDiagnostics();
    }

    void OnDestroy()
    {
        if (cameraDropdown != null)
            cameraDropdown.onValueChanged.RemoveListener(OnCameraSelected);
        if (modelDropdown != null)
            modelDropdown.onValueChanged.RemoveListener(OnModelSelected);
        if (mirrorToggle != null)
            mirrorToggle.onValueChanged.RemoveListener(OnMirrorToggled);
    }

    #region Camera Dropdown

    private void PopulateCameraDropdown()
    {
        if (cameraDropdown == null) return;

        cameraDropdown.ClearOptions();
        cameraDeviceNames.Clear();

        var devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            cameraDropdown.AddOptions(new List<string> { "No camera found" });
            cameraDropdown.interactable = false;
            return;
        }

        var options = new List<string>();
        for (int i = 0; i < devices.Length; i++)
        {
            string label = string.IsNullOrEmpty(devices[i].name) ? $"Camera {i}" : devices[i].name;
            options.Add(label);
            cameraDeviceNames.Add(devices[i].name);
        }

        cameraDropdown.AddOptions(options);
        cameraDropdown.interactable = true;

        // Select current camera if webcam is already running
        if (rvmCore != null)
        {
            var currentWebcam = rvmCore.GetWebCamTexture();
            if (currentWebcam != null)
            {
                int idx = cameraDeviceNames.IndexOf(currentWebcam.deviceName);
                if (idx >= 0) cameraDropdown.SetValueWithoutNotify(idx);
            }
        }
    }

    /// <summary>
    /// Refresh the camera list (call after permission is granted on WebGL).
    /// </summary>
    public void RefreshCameraList()
    {
        PopulateCameraDropdown();
    }

    private void OnCameraSelected(int index)
    {
        if (rvmCore == null || index < 0 || index >= cameraDeviceNames.Count) return;
        if (isSwitching) return;

        string deviceName = cameraDeviceNames[index];
        rvmCore.SwitchCamera(deviceName);
    }

    #endregion

    #region Mirror Toggle

    private void OnMirrorToggled(bool isOn)
    {
        if (rvmCore != null)
            rvmCore.mirrorCamera = isOn;
    }

    #endregion

    #region Model Dropdown

    private void BuildModelList()
    {
        modelEntries.Clear();

        if (mobilenetV3_fp16 != null)
            modelEntries.Add(new ModelEntry { displayName = "MobileNetV3 (FP16)", asset = mobilenetV3_fp16, modelType = RVMCore.ModelType.MobileNetV3 });
        if (mobilenetV3_fp32 != null)
            modelEntries.Add(new ModelEntry { displayName = "MobileNetV3 (FP32)", asset = mobilenetV3_fp32, modelType = RVMCore.ModelType.MobileNetV3 });
        if (resnet50_fp16 != null)
            modelEntries.Add(new ModelEntry { displayName = "ResNet50 (FP16)", asset = resnet50_fp16, modelType = RVMCore.ModelType.ResNet50 });
        if (resnet50_fp32 != null)
            modelEntries.Add(new ModelEntry { displayName = "ResNet50 (FP32)", asset = resnet50_fp32, modelType = RVMCore.ModelType.ResNet50 });
    }

    private void PopulateModelDropdown()
    {
        if (modelDropdown == null) return;

        modelDropdown.ClearOptions();

        if (modelEntries.Count == 0)
        {
            modelDropdown.AddOptions(new List<string> { "No models assigned" });
            modelDropdown.interactable = false;
            return;
        }

        var options = new List<string>();
        int currentIdx = 0;
        for (int i = 0; i < modelEntries.Count; i++)
        {
            options.Add(modelEntries[i].displayName);
            // Match the currently loaded model
            if (rvmCore != null && modelEntries[i].asset == rvmCore.modelAsset)
                currentIdx = i;
        }

        modelDropdown.AddOptions(options);
        modelDropdown.SetValueWithoutNotify(currentIdx);
        modelDropdown.interactable = true;
    }

    private void OnModelSelected(int index)
    {
        if (rvmCore == null || index < 0 || index >= modelEntries.Count) return;
        if (isSwitching) return;

        var entry = modelEntries[index];
        isSwitching = true;
        SetDropdownsInteractable(false);

        rvmCore.SwitchModel(entry.asset, entry.modelType);

        isSwitching = false;
        SetDropdownsInteractable(true);
    }

    private void SetDropdownsInteractable(bool interactable)
    {
        if (cameraDropdown != null) cameraDropdown.interactable = interactable;
        if (modelDropdown != null) modelDropdown.interactable = interactable;
    }

    #endregion

    #region Diagnostics

    private void UpdateFPS()
    {
        fpsFrameCount++;
        fpsTimer += Time.unscaledDeltaTime;
        if (fpsTimer >= 0.5f)
        {
            currentFPS = fpsFrameCount / fpsTimer;
            fpsFrameCount = 0;
            fpsTimer = 0f;
        }
    }

    private void UpdateDiagnostics()
    {
        if (diagnosticsText == null || rvmCore == null) return;

        var cam = rvmCore.GetWebCamTexture();
        string camName = cam != null ? cam.deviceName : "N/A";
        string camRes = cam != null ? $"{cam.width}x{cam.height}" : "N/A";
        string modelName = rvmCore.modelType.ToString();
        string backend = rvmCore.backendMode.ToString();

        diagnosticsText.text = $"Camera: {camName}\n" +
                               $"Resolution: {camRes}\n" +
                               $"Model: {modelName}\n" +
                               $"Backend: {backend}\n" +
                               $"Mirror: {(rvmCore.mirrorCamera ? "On" : "Off")}\n" +
                               $"FPS: {currentFPS:F1}";
    }

    #endregion
}
