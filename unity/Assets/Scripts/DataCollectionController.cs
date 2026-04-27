using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using TMPro;

public class DataCollectionController : MonoBehaviour
{
    [Header("Components")]
    public MapInitializer mapInitializer;
    public AgentDataRecorder dataRecorder;
    public SceneInitializationRecorder sceneRecorder;
    public GameObject agent;
    public GameObject parkgaragePrefab;
    public GameObject soundSource;
    public GameObject backgroundSound;
    public TextMeshProUGUI instructionText;
    public VRTeleporter teleporter;

    [Header("Settings")]
    public string participantId = "p01"; // 当前participant ID
    public string mapsJsonPath = "maps.json";

    private enum CollectionState
    {
        Empty,      // 空场景
        Waiting,    // 等待A键
        Collecting, // 收集数据
        MapFinished, // 当前map完成
        ShowingInstructions, // 显示指令文字，等待用户确认
        RecordingEstimate // 记录估计位置
    }

    private CollectionState currentState = CollectionState.Empty;

    // 数据管理
    private Dictionary<string, List<MapInitData>> participantMaps;
    private int currentMapIndex = 0;
    private List<MapInitData> currentParticipantMaps;

    // 保存路径管理
    private string currentSaveFolder;

    private const int MaxMapsPerParticipant = 270; // indices 0..269

    /// <summary>
    /// Root path used for saving participant/map data depending on build/editor.
    /// </summary>
    private string GetRootSavePath()
    {
#if UNITY_EDITOR
        return Application.streamingAssetsPath;
#else
        return Application.persistentDataPath;
#endif
    }

    /// <summary>
    /// Check whether all map folders 0..269 exist for a given participant.
    /// </summary>
    private bool IsParticipantFull(string pid)
    {
        string root = GetRootSavePath();
        string pDir = Path.Combine(root, pid);
        if (!Directory.Exists(pDir)) return false;
        for (int i = 0; i < MaxMapsPerParticipant; i++)
        {
            if (!Directory.Exists(Path.Combine(pDir, i.ToString())))
            {
                return false; // missing at least one index → not full
            }
        }
        return true; // all 0..269 exist
    }

    /// <summary>
    /// Return the first map index in 0..269 that does NOT exist for pid, or -1 if all exist.
    /// </summary>
    private int FindNextAvailableMapIndex(string pid)
    {
        string root = GetRootSavePath();
        string pDir = Path.Combine(root, pid);
        if (!Directory.Exists(pDir)) return 0;
        for (int i = 0; i < MaxMapsPerParticipant; i++)
        {
            string idxDir = Path.Combine(pDir, i.ToString());
            if (!Directory.Exists(idxDir))
            {
                return i;
            }
        }
        return -1; // full
    }

    /// <summary>
    /// Increment a participant ID like "p01" -> "p02" (preserves zero padding of numeric suffix).
    /// If no numeric suffix is found, appends 02.
    /// </summary>
    private string IncrementParticipantId(string pid)
    {
        // Find numeric tail
        int firstDigit = -1;
        for (int i = pid.Length - 1; i >= 0; i--)
        {
            if (!char.IsDigit(pid[i])) { firstDigit = i + 1; break; }
            if (i == 0) firstDigit = 0;
        }
        if (firstDigit < 0 || firstDigit >= pid.Length)
        {
            return pid + "02"; // fallback
        }
        string prefix = pid.Substring(0, firstDigit);
        string digits = pid.Substring(firstDigit);
        if (int.TryParse(digits, out int n))
        {
            n += 1;
            string padded = n.ToString(new string('0', digits.Length));
            return prefix + padded;
        }
        return pid + "02";
    }

    /// <summary>
    /// Resolve participantId and currentMapIndex based on existing save folders.
    /// - If current participant folder has all 0..269, increment participantId until a non-full one is found.
    /// - If participant exists but is not full, set currentMapIndex to first missing map index.
    /// - If participant does not exist, start at index 0.
    /// Also updates the instructionText to show the current participant ID.
    /// </summary>
    private void ResolveParticipantAndMapIndex()
    {
        string resolvedPid = participantId;
        int nextIdx;

        // Loop until we find a participant that is not full
        int safety = 0; // avoid infinite loop just in case
        while (IsParticipantFull(resolvedPid))
        {
            string prev = resolvedPid;
            resolvedPid = IncrementParticipantId(resolvedPid);
            safety++;
            if (safety > 100)
            {
                Debug.LogError("ResolveParticipantAndMapIndex safety break; too many participant folders are full.");
                break;
            }
            Debug.Log($"Participant '{prev}' is full (0..{MaxMapsPerParticipant - 1}). Trying '{resolvedPid}'...");
        }

        // Now figure out next map index for the resolved participant
        nextIdx = FindNextAvailableMapIndex(resolvedPid);
        if (nextIdx < 0)
        {
            // Full edge-case (e.g., folder created after the while) → increment once more
            string prev = resolvedPid;
            resolvedPid = IncrementParticipantId(resolvedPid);
            nextIdx = 0;
            Debug.LogWarning($"Participant '{prev}' became full; switching to '{resolvedPid}' at index 0.");
        }

        if (participantId != resolvedPid)
        {
            Debug.Log($"Participant ID changed: {participantId} → {resolvedPid}");
            participantId = resolvedPid;
        }
        currentMapIndex = nextIdx;
        Debug.Log($"Resolved participantId='{participantId}', starting currentMapIndex={currentMapIndex}");

        // Update instruction text to show participant ID
        UpdateInstructionText();
    }

    void Start()
    {
        var display = OVRManager.display; 
        display.RecenteredPose += RecenterPoseEventHandler;
        // Decide participantId and starting map index based on existing folders.
        ResolveParticipantAndMapIndex();

        // Optionally, show participant info from the start
        UpdateInstructionText();

        // Load maps for the resolved participantId
        LoadMapsData();

        parkgaragePrefab.SetActive(false);
        soundSource.SetActive(false);
        backgroundSound.SetActive(false);
        instructionText.gameObject.SetActive(false);
        dataRecorder.recordingInterval = 0.1f;
        HandleAButtonPress();
    }

    private void RecenterPoseEventHandler()
    {
        Debug.Log($":: recenter pose hander ::");
        if (currentState == CollectionState.ShowingInstructions)
        {
            StartNextMap();
        }
    }
    /// <summary>
    /// Updates the instruction text to show the current participant ID.
    /// Prepends the participant line to any existing instructions, ensuring it's always on top.
    /// </summary>
    private void UpdateInstructionText()
    {
        if (instructionText != null)
        {
            // instructionText.text = $"Participant: {participantId}\nMap: {currentMapIndex}...\n\n" +
            //                         "The target vehicle color is: \n\n" +
            //                         GetTargetVehicle(currentMapIndex).prefabType +
            //                         "\n\nPlease 'recenter' to start the next map";
        
            string original = instructionText.text;
            string withoutPrefix = original;

            // Remove existing Participant prefix if present
            int newlineIndex = original.IndexOf("\n\n");
            if (original.StartsWith("Participant:"))
            {
                if (newlineIndex >= 0 && newlineIndex < original.Length - 1)
                {
                    withoutPrefix = original.Substring(newlineIndex + 1);
                }
                else
                {
                    withoutPrefix = string.Empty;
                }
            }

            instructionText.text = $"Participant: {participantId}\nMap: {currentMapIndex}\n\n{withoutPrefix}";
        }
    }

    /// <summary>
    /// Update save folder based on current participant and map
    /// </summary>
    private void UpdateSaveFolder()
    {
#if UNITY_EDITOR
        currentSaveFolder = Path.Combine(Application.streamingAssetsPath, participantId, currentMapIndex.ToString());
#else
        currentSaveFolder = Path.Combine(Application.persistentDataPath, participantId, currentMapIndex.ToString());
#endif
        Debug.Log($"Save folder updated to: {currentSaveFolder}");

        // Ensure the save folder exists
        if (!System.IO.Directory.Exists(currentSaveFolder))
        {
            System.IO.Directory.CreateDirectory(currentSaveFolder);
            Debug.Log($"Created save folder: {currentSaveFolder}");
        }
    }

    void Update()
    {
        // if (Input.GetKeyDown(KeyCode.Space))
        // {
        //     Debug.Log("Space pressed, checking current state...");
        //     HandleAButtonPress();
        // }

        if (OVRInput.GetUp(OVRInput.Button.One))
        { 
            Debug.Log("A button pressed, checking current state...");
            HandleAButtonPress();
        }

        // Handle participant ID increment with Button.Four
        if (OVRInput.GetDown(OVRInput.Button.Four))
        {
            participantId = IncrementParticipantId(participantId);
            ResolveParticipantAndMapIndex();
            LoadMapsData();
            // UpdateInstructionText() will be called inside ResolveParticipantAndMapIndex
        }
    }
    
    private void LoadMapsData()
    {
# if UNITY_EDITOR
        string jsonPath = Path.Combine(Application.dataPath, "Resources", mapsJsonPath);
# else
        string jsonPath = Path.Combine(Application.persistentDataPath, mapsJsonPath);
# endif
        string jsonContent = File.ReadAllText(jsonPath);
        
        var participantMaps = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, List<MapInitData>>>(jsonContent); 
        if (participantMaps != null && participantMaps.ContainsKey(participantId))
        {
            currentParticipantMaps = participantMaps[participantId];
            Debug.Log($"Loaded {currentParticipantMaps.Count} maps for participant {participantId}");
        }
        else
        {
            Debug.LogError($"Failed to load maps for participant {participantId}");
        }
       
    }
    
    private void SetupDataRecorder()
    {
        // 更新保存路径
        UpdateSaveFolder();
        
        // 设置数据保存的文件夹名
        if (dataRecorder != null)
        {
            dataRecorder.SetSaveFolder(currentSaveFolder);
        }
        
        if (sceneRecorder != null)
        {
            sceneRecorder.SetSaveFolder(currentSaveFolder);
        }
    }

    [ContextMenu("Handle A Button Press")]
    private void HandleAButtonPress()
    {
        Debug.Log("A button pressed, handling state change...");
        switch (currentState)
        {
            case CollectionState.Empty:
                Debug.Log("Starting data collection...");
                if (currentMapIndex < currentParticipantMaps.Count)
                {
                    // 显示开始下一个map的指令文字
                    parkgaragePrefab.SetActive(false);
                    ShowNextMapInstructions();
                    MoveSoundSourceToTargetVehicle(soundSource);
                    return; // 不立即开始，等待用户再次按空格键
                }
                else
                {
                    Debug.Log("All maps completed! Ending program...");
                    // 所有map完成，直接结束程序
#if UNITY_EDITOR
                        UnityEditor.EditorApplication.isPlaying = false;
#else
                    Application.Quit();
#endif
                }
                break;
            case CollectionState.Waiting:
                Debug.Log("User confirmed to start map...");
                // 开始下一个map
                currentState = CollectionState.ShowingInstructions;
                ShowNextMapInstructions();
                MoveSoundSourceToTargetVehicle(soundSource);
                break;
            case CollectionState.ShowingInstructions:
                Debug.Log("User click recenter to start map...");
                MoveSoundSourceToTargetVehicle(soundSource);
                break;

            case CollectionState.Collecting:
                Debug.Log("Data collection ends...");
                FinishCurrentMap();
                break;

            case CollectionState.RecordingEstimate:
                Debug.Log("Recording estimate...");
                RecordEstimate();
                Debug.Log("Show next map...");
                if (currentMapIndex < currentParticipantMaps.Count)
                {
                    // 显示开始下一个map的指令文字
                    parkgaragePrefab.SetActive(false);
                    ShowNextMapInstructions();
                    MoveSoundSourceToTargetVehicle(soundSource);
                    return; // 不立即开始，等待用户再次按空格键
                }
                else
                {
                    Debug.Log("All maps completed! Ending program...");
                    // 所有map完成，直接结束程序
#if UNITY_EDITOR
                        UnityEditor.EditorApplication.isPlaying = false;
#else
                    Application.Quit();
#endif
                }
                break;

        }
    }
    
    private void StartNextMap()
    {
        // OVRManager.display.RecenterPose();
        MapInitData currentMap = currentParticipantMaps[currentMapIndex];
        Debug.Log($"Starting map {currentMapIndex + 1}/{currentParticipantMaps.Count}");
        
        // 隐藏指令文字
        if (instructionText != null)
        {
            instructionText.gameObject.SetActive(false);
            Debug.Log("Instruction text hidden - starting map");
        }
        
        // 更新保存路径并传递给recorder
        SetupDataRecorder();
        
        // 1. 使用现有的MapInitializer加载map
        parkgaragePrefab.SetActive(true);
        soundSource.SetActive(true);
        mapInitializer.InitializeMap(currentMap);
        sceneRecorder.RecordSceneData();
        bool background_noise = currentMap.background_noise;
        Debug.Log($"Background noise: {background_noise}");
        if (background_noise)
        {
            backgroundSound.SetActive(true);
        }
        
        
        // 2. 开始数据收集和保存（使用现有的AgentDataRecorder）
        dataRecorder.StartRecording();
        
        currentState = CollectionState.Collecting;
    }
    
    private void FinishCurrentMap()
    {
        if (dataRecorder != null)
        {
            dataRecorder.StopRecording();
        }
        
        if (mapInitializer != null)
        {
            mapInitializer.ClearCurrentMap();
        }

        teleporter.ToggleDisplay(true);
        
        // parkgaragePrefab.SetActive(false);
        soundSource.SetActive(false);
        backgroundSound.SetActive(false);
        
        currentMapIndex++;
        currentState = CollectionState.RecordingEstimate;
    }

    private void RecordEstimate()
    {
        Vector3? estimatePosition = teleporter.RecordPosition();
        if (estimatePosition != null)
        {
            Debug.Log($"Estimated position: {estimatePosition}");
            string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string saveFolder = Path.Combine(currentSaveFolder, $"estimate_{timestamp}.json");
            File.WriteAllText(saveFolder, JsonUtility.ToJson(estimatePosition));
        }
        else
        {
            Debug.Log("No estimate position found");
        }
        teleporter.ToggleDisplay(false);
        currentState = CollectionState.ShowingInstructions;
    }
    
    /// <summary>
    /// 显示下一个map的指令文字
    /// </summary>
    private void ShowNextMapInstructions()
    {
        if (instructionText != null)
        {
            if (currentState == CollectionState.Empty)
            {
                instructionText.text = $"Participant: {participantId}\nMap: {currentMapIndex}\n\n" +
                                  "Press 'A' to start";
                // 自动调节text位置到camera前面
                PositionTextInFrontOfCamera();

                instructionText.gameObject.SetActive(true);
                currentState = CollectionState.Waiting;
                Debug.Log($"Showing instructions for map {currentMapIndex + 1}, waiting for user confirmation");
            }
            else
            {
                instructionText.text = $"Participant: {participantId}\nMap: {currentMapIndex}\n\n" +
                                  "The target vehicle color is: \n\n" +
                                  GetTargetVehicle(currentMapIndex).prefabType +
                                  "\n\nPlease 'recenter' to start the next map";

                // 自动调节text位置到camera前面
                PositionTextInFrontOfCamera();

                instructionText.gameObject.SetActive(true);
                currentState = CollectionState.ShowingInstructions;
                Debug.Log($"Showing instructions for map {currentMapIndex + 1}, waiting for user confirmation");
            }
            
            

            // // Reset camera rotation Y to 0
            // Vector3 euler = agent.transform.eulerAngles;
            // agent.transform.eulerAngles = new Vector3(euler.x, 0f, euler.z);
            // Debug.Log("Agent Y rotation reset to 0");
            // OVRManager.display.RecenterPose();
        }
    }
    
    
    /// <summary>
    /// 自动调节text位置到camera前面
    /// </summary>
    private void PositionTextInFrontOfCamera()
    {
        if (instructionText == null) return;
        
        // 获取主camera
        Camera mainCamera = Camera.main;
        if (mainCamera == null) return;
        
        // 计算camera前方的位置（距离camera 2米）
        Vector3 cameraPosition = mainCamera.transform.position;
        Vector3 cameraForward = mainCamera.transform.forward;
        Vector3 textPosition = cameraPosition + cameraForward * 2f;
        
        // 确保text不会太低（至少与camera同高）
        if (textPosition.y < cameraPosition.y)
        {
            textPosition.y = cameraPosition.y;
        }
        
        // 获取Canvas组件
        Canvas canvas = instructionText.GetComponentInParent<Canvas>();
        if (canvas != null)
        {
            // 如果Canvas是World Space模式，直接移动Canvas
            if (canvas.renderMode == RenderMode.WorldSpace)
            {
                canvas.transform.position = textPosition;
                // 让Canvas面向camera
                canvas.transform.LookAt(mainCamera.transform);
                canvas.transform.Rotate(0, 180, 0);
                Debug.Log($"Moved Canvas to {textPosition}, facing camera");
            }
            else
            {
                // 如果是Screen Space模式，我们需要将世界坐标转换为屏幕坐标
                Vector3 screenPos = mainCamera.WorldToScreenPoint(textPosition);
                if (screenPos.z > 0) // 确保在camera前面
                {
                    // 将屏幕坐标转换为Canvas坐标
                    RectTransformUtility.ScreenPointToLocalPointInRectangle(
                        canvas.transform as RectTransform,
                        screenPos,
                        canvas.worldCamera,
                        out Vector2 localPoint);
                    
                    instructionText.rectTransform.anchoredPosition = localPoint;
                    Debug.Log($"Positioned text in Screen Space Canvas at screen position {screenPos}");
                }
            }
        }
        else
        {
            // 如果没有Canvas，直接移动text对象
            instructionText.transform.position = textPosition;
            instructionText.transform.LookAt(mainCamera.transform);
            instructionText.transform.Rotate(0, 180, 0);
            Debug.Log($"Moved text object directly to {textPosition}, facing camera");
        }
    }
    
    /// <summary>
    /// 获取目标车辆对象
    /// </summary>
    private VehicleInitData GetTargetVehicle(int mapIndex)
    {
        if (currentParticipantMaps != null && mapIndex < currentParticipantMaps.Count)
        {
            var map = currentParticipantMaps[mapIndex];
            var targetVehicle = map.vehicles?.FirstOrDefault(v => v.isTarget);
            return targetVehicle;
        }
        return null;
    }
    
    
    /// <summary>
    /// 移动sound source到目标车辆位置
    /// </summary>
    public void MoveSoundSourceToTargetVehicle(GameObject soundSource)
    {
        if (soundSource == null) return;
        
        var targetVehicle = GetTargetVehicle(currentMapIndex);
        if (targetVehicle != null)
        {
            // 创建Vector3位置
            Vector3 targetPosition = new Vector3(
                targetVehicle.position.x,
                targetVehicle.position.y,
                targetVehicle.position.z
            );
            
            // 移动sound source到目标车辆位置
            soundSource.transform.position = targetPosition;
            
            Debug.Log($"Moved sound source to target vehicle position: {targetPosition}");
        }
        else
        {
            Debug.LogWarning("No target vehicle found for current map");
        }
    }
}
