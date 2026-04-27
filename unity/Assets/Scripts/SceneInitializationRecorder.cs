using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

public class SceneInitializationRecorder : MonoBehaviour
{
    [Header("File Settings")]
    public string fileName = "scene_data";           // File name (without extension) - changed from "vehicle_positions"
    public string fileExtension = ".json";          // File extension
    public bool includeTimestamp = true;            // Include timestamp in filename
    
    [Header("Data Settings")]
    public bool includeRotation = true;             // Include rotation information
    public bool includeScale = true;                // Include scale information
    public bool recordAgentData = true;             // Whether to record agent position and rotation
    
    [Header("Agent Settings")]
    public GameObject agentObject;                   // Reference to the agent GameObject
    public string agentTag = "Agent";               // Tag to identify the agent
    
    [Header("Map Initializer Reference")]
    public MapInitializer mapInitializer;           // Reference to MapInitializer for agent initial data
    
    private string filePath;
    private string saveFolder = Application.streamingAssetsPath;  // Default fallback
    
    [System.Serializable]
    public class VehicleData
    {
        public string name;
        public string tag;
        public Vector3 position;
        public Vector3 rotation;
        public Vector3 scale;
        public bool isActive;
        public string timestamp;
        public int layer;
    }
    
    [System.Serializable]
    public class AgentData
    {
        public string name;
        public string tag;
        public Vector3 position;
        public Vector3 rotation;
        public Vector3 initialPosition;              // Initial position from MapInitializer
        public Vector3 initialRotation;             // Initial rotation from MapInitializer
        public bool isActive;
        public string timestamp;
        public int layer;
    }
    
    [System.Serializable]
    public class SceneData
    {
        public string recordTime;
        public int totalVehicles;
        public List<VehicleData> vehicles;
        public AgentData agent;                      // Agent information
    }
    
    void Start()
    {
        FindMapInitializer();
        FindAgentObject();
        // GenerateFilePath();
        // RecordSceneData();
    }
    
    /// <summary>
    /// Set save folder from external source (e.g., DataCollectionController)
    /// </summary>
    public void SetSaveFolder(string newSaveFolder)
    {
        if (!string.IsNullOrEmpty(newSaveFolder) && newSaveFolder != saveFolder)
        {
            saveFolder = newSaveFolder;
            Debug.Log($"SceneInitializationRecorder: Save folder set to {saveFolder}");
        }
    }
    
    /// <summary>
    /// Find MapInitializer in the scene
    /// </summary>
    private void FindMapInitializer()
    {
        if (mapInitializer == null)
        {
            mapInitializer = FindObjectOfType<MapInitializer>();
            if (mapInitializer != null)
            {
                Debug.Log($"VehiclePositionRecorder: Found MapInitializer: {mapInitializer.name}");
            }
            else
            {
                Debug.LogWarning("VehiclePositionRecorder: No MapInitializer found in scene");
            }
        }
    }
    
    /// <summary>
    /// Find the agent object in the scene
    /// </summary>
    private void FindAgentObject()
    {
        // If agentObject is not assigned, try to find it automatically
        if (agentObject == null)
        {
            // First try to find by tag
            GameObject[] agentObjects = GameObject.FindGameObjectsWithTag(agentTag);
            if (agentObjects.Length > 0)
            {
                agentObject = agentObjects[0];
                Debug.Log($"VehiclePositionRecorder: Found agent by tag '{agentTag}': {agentObject.name}");
            }
            else
            {
                // Try to find AgentDataRecorder and use its GameObject
                AgentDataRecorder agentRecorder = FindObjectOfType<AgentDataRecorder>();
                if (agentRecorder != null)
                {
                    agentObject = agentRecorder.gameObject;
                    Debug.Log($"VehiclePositionRecorder: Using AgentDataRecorder GameObject as agent: {agentObject.name}");
                }
                else
                {
                    // Try to find by name containing "Agent" or "Player"
                    GameObject[] allObjects = FindObjectsOfType<GameObject>();
                    foreach (GameObject obj in allObjects)
                    {
                        if (obj.name.ToLower().Contains("agent") || obj.name.ToLower().Contains("player"))
                        {
                            agentObject = obj;
                            Debug.Log($"VehiclePositionRecorder: Found agent by name: {agentObject.name}");
                            break;
                        }
                    }
                }
            }
        }
        
        if (agentObject == null)
        {
            Debug.LogWarning("VehiclePositionRecorder: No agent object found! Agent data will not be recorded.");
        }
    }
    
    /// <summary>
    /// Generate file path
    /// </summary>
    private void GenerateFilePath()
    {
        string timestamp = includeTimestamp ? "_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") : "";
        string fileNameWithTimestamp = fileName + timestamp + fileExtension;
        
        // Use current saveFolder (either set externally or default)
        Debug.Log($"SceneInitializationRecorder: Using save folder: {saveFolder}");
        
        // Ensure user subfolder exists
        if (!Directory.Exists(saveFolder))
        {
            Directory.CreateDirectory(saveFolder);
            Debug.Log($"Created user subfolder: {saveFolder}");
        }
        
        // Save to user subfolder
        filePath = Path.Combine(saveFolder, fileNameWithTimestamp);
        
        Debug.Log($"Scene data file will be saved to: {filePath}");
    }
    
    /// <summary>
    /// Get agent initial data from MapInitializer
    /// </summary>
    private Vector3 GetAgentInitialPosition()
    {
        if (mapInitializer != null)
        {
            MapInitData currentMapState = mapInitializer.GetCurrentMapState();
            if (currentMapState != null && currentMapState.agent != null)
            {
                Debug.Log($"VehiclePositionRecorder: Got agent initial position from MapInitializer: {currentMapState.agent.position}");
                return currentMapState.agent.position;
            }
        }
        return Vector3.zero;
    }
    
    private Vector3 GetAgentInitialRotation()
    {
        if (mapInitializer != null)
        {
            MapInitData currentMapState = mapInitializer.GetCurrentMapState();
            if (currentMapState != null && currentMapState.agent != null)
            {
                Debug.Log($"VehiclePositionRecorder: Got agent initial rotation from MapInitializer: {currentMapState.agent.rotation}");
                return currentMapState.agent.rotation;
            }
        }
        return Vector3.zero;
    }
    
    /// <summary>
    /// Record agent position and rotation
    /// </summary>
    private AgentData RecordAgentData()
    {
        if (agentObject == null || !recordAgentData)
        {
            return null;
        }
        
        // Get initial data from MapInitializer
        Vector3 initialPosition = GetAgentInitialPosition();
        Vector3 initialRotation = GetAgentInitialRotation();
        
        AgentData agentData = new AgentData
        {
            name = agentObject.name,
            tag = agentObject.tag,
            position = agentObject.transform.position,
            rotation = includeRotation ? agentObject.transform.rotation.eulerAngles : Vector3.zero,
            initialPosition = initialPosition,
            initialRotation = initialRotation,
            isActive = agentObject.activeInHierarchy,
            timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
            layer = agentObject.layer
        };
        
        string initialInfo = initialPosition != Vector3.zero ? 
            $" (Initial: {initialPosition}, {initialRotation})" : "";
        Debug.Log($"Recorded agent: {agentObject.name} at position {agentObject.transform.position}, rotation {agentObject.transform.rotation.eulerAngles}{initialInfo}");
        return agentData;
    }
    
    /// <summary>
    /// Record all scene data (vehicles and agent)
    /// </summary>
    public void RecordSceneData()
    {
        Debug.Log("Starting to record scene data (vehicles and agent)...");
        
        // Find all objects tagged as "Vehicles"
        GameObject[] allObjects = FindObjectsOfType<GameObject>();
        List<VehicleData> vehicles = new List<VehicleData>();
        
        foreach (GameObject obj in allObjects)
        {
            if (obj.CompareTag("Vehicles"))
            {
                VehicleData vehicleData = new VehicleData
                {
                    name = obj.name,
                    tag = obj.tag,
                    position = obj.transform.position,
                    rotation = includeRotation ? obj.transform.rotation.eulerAngles : Vector3.zero,
                    scale = includeScale ? obj.transform.localScale : Vector3.one,
                    isActive = obj.activeInHierarchy,
                    timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
                    layer = obj.layer
                };
                
                vehicles.Add(vehicleData);
                Debug.Log($"Recorded vehicle: {obj.name} at position {obj.transform.position}");
            }
        }
        
        // Record agent data
        AgentData agentData = RecordAgentData();
        
        // Create data object
        SceneData data = new SceneData
        {
            recordTime = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
            totalVehicles = vehicles.Count,
            vehicles = vehicles,
            agent = agentData
        };
        
        // Save to file
        SaveToFile(data);
        
        string agentInfo = agentData != null ? $" and agent '{agentData.name}'" : "";
        Debug.Log($"Recorded {vehicles.Count} vehicles{agentInfo} and saved to file.");
    }
    
    /// <summary>
    /// Save data to file
    /// </summary>
    private void SaveToFile(SceneData data)
    {
        try
        {
            GenerateFilePath();
            string json = JsonUtility.ToJson(data, true);
            File.WriteAllText(filePath, json);
            Debug.Log($"Scene data saved to: {filePath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to save scene data: {e.Message}");
        }
    }
    
    /// <summary>
    /// Public method to manually trigger recording
    /// </summary>
    public void ManualRecord()
    {
        RecordSceneData();
    }
    
    /// <summary>
    /// Update agent reference (useful if agent is spawned dynamically)
    /// </summary>
    public void UpdateAgentReference(GameObject newAgent)
    {
        agentObject = newAgent;
        Debug.Log($"VehiclePositionRecorder: Updated agent reference to: {newAgent.name}");
    }
    
    /// <summary>
    /// Update MapInitializer reference
    /// </summary>
    public void UpdateMapInitializerReference(MapInitializer newMapInitializer)
    {
        mapInitializer = newMapInitializer;
        Debug.Log($"VehiclePositionRecorder: Updated MapInitializer reference to: {newMapInitializer.name}");
    }
    
    // Legacy method name for backward compatibility
    [System.Obsolete("Use RecordSceneData() instead. This method will be removed in future versions.")]
    public void RecordAllVehicles()
    {
        RecordSceneData();
    }
}

