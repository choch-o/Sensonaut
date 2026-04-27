using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

[System.Serializable]
public class VehicleInitData
{
    public string name;
    public Vector3 position;
    public Vector3 rotation;
    public Vector3 scale = Vector3.one;
    public bool isActive = true;
    public string prefabType; // "black", "red", "white"
    public bool isTarget = false; // Indicates whether this vehicle is the target
}

[System.Serializable]
public class AgentInitData
{
    public Vector3 position;
    public Vector3 rotation;
}

[System.Serializable]
public class MapInitData
{
    public List<VehicleInitData> vehicles;
    public AgentInitData agent;
    public bool background_noise;
    public string mapName;
    public string description;
}

public class MapInitializer : MonoBehaviour
{
    [Header("Vehicle Prefabs")]
    public GameObject carBlackPrefab;  // Car black.prefab
    public GameObject carRedPrefab;    // Car red.prefab
    public GameObject carWhitePrefab;  // Car white.prefab

    [Header("Agent Settings")]
    public GameObject agent;    // Agent prefab
    public Transform mapRoot;         // Map root node

    [Header("Position File")]
    public string positionFileName = "vehicle_positions_20250813_141730.json";

    [Header("Audio Settings")]
    public AudioClip beeconClip; // Assigned in Inspector: beacon sound clip for target vehicles
    
    [Header("Debug")]
    public bool logInitialization = true;
    public bool showGizmos = true;
    
    private List<GameObject> spawnedVehicles = new List<GameObject>();
    private GameObject currentAgent;
    private List<VehicleData> availablePositions = new List<VehicleData>();
    
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
    public class VehiclePositionData
    {
        public string recordTime;
        public int totalVehicles;
        public List<VehicleData> vehicles;
    }
    
    void Start()
    {
        // Delayed initialization to ensure execution in main thread
        Invoke("InitializeMapRoot", 0.1f);
    }
    
    private void InitializeMapRoot()
    {
        // Ensure there's a map root node
        if (mapRoot == null)
        {
            mapRoot = new GameObject("MapRoot").transform;
        }
        
        // Load available vehicle positions
        // LoadAvailablePositions();
    }
    
    /// <summary>
    /// Load available vehicle positions
    /// </summary>
    private void LoadAvailablePositions()
    {
        try
        {
            string filePath = Path.Combine(Application.streamingAssetsPath, positionFileName);
            
            if (!File.Exists(filePath))
            {
                Debug.LogError($"Vehicle position file does not exist: {filePath}");
                return;
            }
            
            string json = File.ReadAllText(filePath);
            VehiclePositionData data = JsonUtility.FromJson<VehiclePositionData>(json);
            
            if (data != null && data.vehicles != null)
            {
                availablePositions = data.vehicles;
                Debug.Log($"Successfully loaded {availablePositions.Count} available vehicle positions");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to load vehicle position file: {e.Message}");
        }
    }
    
    /// <summary>
    /// Initialize map
    /// </summary>
    public void InitializeMap(MapInitData initData)
    {
        if (logInitialization)
        {
            Debug.Log($"=== Starting map initialization ===");
            Debug.Log($"Map name: {initData.mapName}");
            // Debug.Log($"Description: {initData.description}");
            Debug.Log($"Vehicle count: {initData.vehicles?.Count ?? 0}");
        }
        
        // Clear existing objects
        ClearCurrentMap();
        
        // Initialize vehicles
        if (initData.vehicles != null && initData.vehicles.Count > 0)
        {
            InitializeVehicles(initData.vehicles);
        }
        
        // Initialize Agent
        if (initData.agent != null)
        {
            InitializeAgent(initData.agent);
        }
        
        if (logInitialization)
        {
            Debug.Log($"=== Map initialization completed ===");
        }
    }
    
    /// <summary>
    /// Initialize vehicles
    /// </summary>
    private void InitializeVehicles(List<VehicleInitData> vehicles)
    {
        if (logInitialization)
        {
            Debug.Log($"Initializing {vehicles.Count} vehicles...");
        }
        
        foreach (var vehicleData in vehicles)
        {
            GameObject vehicle = CreateVehicle(vehicleData);
            if (vehicle != null)
            {
                spawnedVehicles.Add(vehicle);
                
                if (logInitialization)
                {
                    Debug.Log($"Created vehicle: {vehicle.name} at position {vehicleData.position}");
                }
            }
        }
    }
    
    /// <summary>
    /// Create a single vehicle
    /// </summary>
    private GameObject CreateVehicle(VehicleInitData vehicleData)
    {
        // Select prefab based on prefabType
        GameObject selectedPrefab = GetVehiclePrefab(vehicleData.prefabType);

        if (selectedPrefab == null)
        {
            Debug.LogError($"Cannot find vehicle prefab: {vehicleData.prefabType}");
            return null;
        }

        // Instantiate vehicle
        GameObject vehicle = Instantiate(selectedPrefab, mapRoot);

        // Set vehicle properties
        vehicle.name = vehicleData.name;
        vehicle.transform.position = vehicleData.position;
        vehicle.transform.rotation = Quaternion.Euler(vehicleData.rotation);
        vehicle.transform.localScale = vehicleData.scale;
        vehicle.SetActive(vehicleData.isActive);

        // Set tag to "Vehicles"
        vehicle.tag = "Vehicles";

        // // Ensure every vehicle has an AudioSource component
        // var audio = vehicle.GetComponent<AudioSource>();
        // if (audio == null)
        // {
        //     audio = vehicle.AddComponent<AudioSource>();
        // }
        // // Configure sensible defaults for 3D spatial audio
        // audio.spatialize = true;        // Enable spatialization
        // audio.spatialBlend = 1.0f;      // fully 3D
        // audio.playOnAwake = false;      // don't auto-play
        // audio.loop = true;              // optional: loop if used as beacon
        // audio.rolloffMode = AudioRolloffMode.Logarithmic;
        // audio.minDistance = 2f;
        // audio.maxDistance = 50f;
        // audio.setActive(false);

        // // Assign shared beacon clip to all vehicles (can be null if not set)
        // if (beeconClip != null)
        // {
        //     GetComponent<AudioSource>().clip = beeconClip;
        // }

        // // Only the target should have an active (enabled) AudioSource
        // // Others keep the component but disabled and stopped
        // if (vehicleData.isTarget)
        // {
        //     GetComponent<AudioSource>().enabled = true;
        //     GetComponent<AudioSource>().mute = false;
        //     if (GetComponent<AudioSource>().clip != null)
        //     {
        //         GetComponent<AudioSource>().Play();
        //     }
        // }
        // else
        // {
        //     if (GetComponent<AudioSource>().isPlaying) GetComponent<AudioSource>().Stop();
        //     GetComponent<AudioSource>().enabled = false;
        //     GetComponent<AudioSource>().mute = true; // extra safety
        // }

        // Append tag to target's name for easy identification
        if (vehicleData.isTarget && !vehicle.name.Contains("(Target)"))
        {
            vehicle.name = vehicle.name + " (Target)";
        }

        if (vehicleData.isTarget && logInitialization)
        {
            Debug.Log($"Marked target vehicle: {vehicle.name} with active AudioSource");
        }

        return vehicle;
    }
    
    /// <summary>
    /// Get vehicle prefab by type
    /// </summary>
    private GameObject GetVehiclePrefab(string prefabType)
    {
        switch (prefabType?.ToLower())
        {
            case "black":
                return carBlackPrefab;
            case "red":
                return carRedPrefab;
            case "white":
                return carWhitePrefab;
            default:
                Debug.LogWarning($"Unknown vehicle type: {prefabType}, using black vehicle as default");
                return carBlackPrefab;
        }
    }
    
    /// <summary>
    /// Get random vehicle configurations from available positions
    /// </summary>
    public List<VehicleInitData> GetRandomVehicleConfigurations(int count)
    {
        List<VehicleInitData> configurations = new List<VehicleInitData>();
        
        if (availablePositions.Count == 0)
        {
            Debug.LogWarning("No available vehicle positions");
            return configurations;
        }
        
        // Randomly select positions
        System.Random random = new System.Random();
        List<VehicleData> shuffledPositions = new List<VehicleData>(availablePositions);
        
        // Fisher-Yates shuffle algorithm
        for (int i = shuffledPositions.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            var temp = shuffledPositions[i];
            shuffledPositions[i] = shuffledPositions[j];
            shuffledPositions[j] = temp;
        }
        
        // Select first count positions
        int selectedCount = Math.Min(count, shuffledPositions.Count);
        string[] prefabTypes = { "black", "red", "white" };
        
        for (int i = 0; i < selectedCount; i++)
        {
            var position = shuffledPositions[i];
            var config = new VehicleInitData
            {
                name = position.name,
                position = position.position,
                rotation = position.rotation,
                scale = position.scale,
                isActive = position.isActive,
                prefabType = prefabTypes[random.Next(prefabTypes.Length)] // Randomly select color
            };
            configurations.Add(config);
        }
        
        return configurations;
    }
    
    /// <summary>
    /// Initialize Agent
    /// </summary>
    private void InitializeAgent(AgentInitData agentData)
    {
        if (logInitialization)
        {
            Debug.Log($"Initializing Agent at position {agentData.position}...");
        }

        if (agent != null)
        {
            // Use prefab
            // currentAgent = Instantiate(agentPrefab, mapRoot);
            // currentAgent = agentPrefab;
            agent.transform.SetParent(mapRoot);
        }
        else
        {
            // Create simple capsule as Agent
            agent = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            if (mapRoot != null)
            {
                agent.transform.SetParent(mapRoot);
            }
            agent.name = "Agent";
        }
        
        // Set Agent properties
        agent.transform.position = agentData.position;
        agent.transform.rotation = Quaternion.Euler(agentData.rotation);
        
        if (logInitialization)
        {
            Debug.Log($"Agent initialization completed: {agent.name}");
        }
    }
    
    /// <summary>
    /// Clear current map
    /// </summary>
    public void ClearCurrentMap()
    {
        if (logInitialization)
        {
            Debug.Log("Clearing current map...");
        }
        
        // Clear vehicles
        foreach (var vehicle in spawnedVehicles)
        {
            if (vehicle != null)
            {
                Destroy(vehicle);
            }
        }
        spawnedVehicles.Clear();
        
        // Clear Agent
        if (currentAgent != null)
        {
            Destroy(currentAgent);
            currentAgent = null;
        }
        
        if (logInitialization)
        {
            Debug.Log("Map clearing completed");
        }
    }
    
    /// <summary>
    /// Get current map state
    /// </summary>
    public MapInitData GetCurrentMapState()
    {
        var mapData = new MapInitData
        {
            mapName = "Current Map",
            description = "Current map state",
            vehicles = new List<VehicleInitData>(),
            agent = new AgentInitData()
        };
        
        // Get vehicle information
        GameObject[] allVehicles = GameObject.FindGameObjectsWithTag("Vehicles");
        foreach (var vehicle in allVehicles)
        {
            var audio = vehicle.GetComponent<AudioSource>();
            var vehicleData = new VehicleInitData
            {
                name = vehicle.name,
                position = vehicle.transform.position,
                rotation = vehicle.transform.rotation.eulerAngles,
                scale = vehicle.transform.localScale,
                isActive = vehicle.activeInHierarchy,
                isTarget = (audio != null && audio.enabled && !audio.mute)
            };
            mapData.vehicles.Add(vehicleData);
        }
        
        // Get Agent information
        if (currentAgent != null)
        {
            mapData.agent.position = currentAgent.transform.position;
            mapData.agent.rotation = currentAgent.transform.rotation.eulerAngles;
        }
        
        return mapData;
    }
    
    /// <summary>
    /// Visualize in scene
    /// </summary>
    void OnDrawGizmos()
    {
        if (!showGizmos) return;
        
        // Draw vehicle positions
        Gizmos.color = Color.green;
        GameObject[] vehicles = GameObject.FindGameObjectsWithTag("Vehicles");
        foreach (var vehicle in vehicles)
        {
            Gizmos.DrawWireSphere(vehicle.transform.position, 0.5f);
        }
        
        // Draw Agent position
        if (currentAgent != null)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(currentAgent.transform.position, 0.3f);
        }
    }
}
