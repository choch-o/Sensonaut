# Unity Audiovisual Search System

This project implements a comprehensive Python-Unity communication system for audiovisual search and agent control in 3D environments. The system features multi-modal data recording, vehicle position management, and real-time agent observations.

## 🏗️ Project Structure

```
├── control_client.py                    # Main Python client for Unity communication
├── Assets/
│   └── Samples/Meta XR Audio SDK/77.0.0/Example Scenes/scripts/
│       ├── MapInitializer.cs           # Map initialization and vehicle management
│       ├── SocketServer.cs             # Unity server for Python communication
│       ├── AgentControl.cs             # Agent movement and control
│       ├── VisualCapture.cs            # Visual observation capture
│       ├── AudioCapture.cs             # Audio recording and processing
│       ├── AgentDataRecorder.cs        # Multi-modal data recording system
│       ├── SceneInitializationRecorder.cs  # Scene data recording (vehicles + agent)
│       ├── MovementMap.cs              # Movement and navigation system
│       ├── LaunchManager.cs            # Application launch management
│       └── FirstPersonControl.cs       # First-person camera control
```

## 🎯 Core Features

### Multi-Modal Data Recording
- **Agent Data**: Position, rotation, visual observations, and audio recordings
- **Scene Data**: Complete scene snapshots including vehicle positions and agent data
- **User Management**: User ID-based folder organization for data separation

### Real-Time Communication
- **Python-Unity Bridge**: TCP socket communication for remote control
- **Command System**: Movement, observation, and map management commands
- **Data Streaming**: Real-time visual and audio data transmission

### Vehicle Management
- **Dynamic Spawning**: Position-based vehicle placement
- **Template System**: Multiple vehicle types (black, red, white)
- **Audio Integration**: Spatial audio with beacon sounds for target vehicles

## 📊 Data Organization

### File Structure
```
StreamingAssets/
├── user001/
│   ├── scene_data_YYYYMMDD_HHMMSS.json     # Complete scene snapshot
│   ├── agent_data_YYYYMMDD_HHMMSS_XXXX.json # Agent observation data
│   └── agent_audio_YYYYMMDD_HHMMSS_XXXX.wav # Audio recordings
├── user002/
│   ├── scene_data_YYYYMMDD_HHMMSS.json
│   ├── agent_data_YYYYMMDD_HHMMSS_XXXX.json
│   └── agent_audio_YYYYMMDD_HHMMSS_XXXX.wav
└── ...
```

### Data Types

#### 1. Scene Data (`scene_data_*.json`)
Complete scene snapshot including all vehicles and agent information:
```json
{
    "recordTime": "2025-01-13 14:30:00",
    "totalVehicles": 17,
    "vehicles": [
        {
            "name": "Car 1",
            "tag": "Vehicles",
            "position": {"x": 10.0, "y": 0.6, "z": 5.0},
            "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
            "isActive": true,
            "timestamp": "2025-01-13 14:30:00.123",
            "layer": 0
        }
    ],
    "agent": {
        "name": "Agent",
        "tag": "Agent",
        "position": {"x": 0.0, "y": 1.0, "z": 0.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "initialPosition": {"x": 0.0, "y": 1.0, "z": 0.0},
        "initialRotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "isActive": true,
        "timestamp": "2025-01-13 14:30:00.123",
        "layer": 0
    }
}
```

#### 2. Agent Data (`agent_data_*.json`)
Individual agent observation data points:
```json
{
    "timestamp": "2025-01-13 14:30:00.123",
    "position": {"x": 0.0, "y": 1.0, "z": 0.0},
    "rotation": {"x": 0.0, "y": 45.0, "z": 0.0},
    "audioFileName": "agent_audio_20250113_143000_0001.wav",
    "visualData": {
        "detectedObjects": [
            {
                "name": "Car_black_1",
                "tag": "Vehicles",
                "distance": 5.2,
                "angle": 45.0,
                "isVisible": true,
                "position": {"x": 5.0, "y": 0.0, "z": 5.0}
            }
        ]
    },
    "audioLevel": 0.75
}
```

#### 3. Audio Files (`agent_audio_*.wav`)
Corresponding audio recordings for each data point (WAV format, 48kHz, 16-bit, stereo).

## 🎮 Unity Scripts

### Core Data Recording System

#### AgentDataRecorder.cs
Multi-modal data recording system for agent observations.

**Key Features:**
- **Synchronized Recording**: Position, rotation, visual, and audio data
- **Individual File Saving**: Each data point saved as separate JSON + WAV files
- **User ID Management**: Automatic user subfolder creation and organization
- **Configurable Intervals**: Adjustable recording frequency (default: 2.0 seconds)

**Configuration:**
```csharp
[Header("File Settings")]
public string userId = "user001";              // User subfolder name

[Header("Recording Settings")]
public float recordingInterval = 2.0f;         // Recording frequency
public bool recordPosition = true;             // Record agent position
public bool recordRotation = true;             // Record agent rotation
public bool recordAudio = true;                // Record audio level
public bool recordVisualData = true;           // Record visual observations

[Header("Audio Settings")]
public bool recordAudioToFile = true;          // Save audio to WAV files

[Header("Components")]
public Camera agentCamera;                     // Agent's camera
public AudioListener audioListener;            // Audio listener
public VisualCapture visualCapture;            // Visual capture component
```

**Recording Workflow:**
1. **Every `recordingInterval` seconds**:
   - Stop previous audio recording
   - Start new audio recording
   - Capture visual observation
   - Record agent position and rotation
   - Save data point to JSON file
   - Record audio to WAV file

#### SceneInitializationRecorder.cs
Scene data recording system for vehicles and agent.

**Key Features:**
- **Complete Scene Snapshots**: Records all vehicles and agent data
- **MapInitializer Integration**: Gets agent initial position from MapInitializer
- **User ID Synchronization**: Automatically uses same user ID as AgentDataRecorder
- **Smart Agent Detection**: Auto-finds agent objects in scene

**Configuration:**
```csharp
[Header("File Settings")]
public string fileName = "scene_data";         // File name (without extension)
public string fileExtension = ".json";         // File extension
public bool includeTimestamp = true;           // Include timestamp in filename

[Header("Data Settings")]
public bool includeRotation = true;            // Include rotation information
public bool includeScale = true;               // Include scale information
public bool recordAgentData = true;            // Record agent position and rotation

[Header("Agent Settings")]
public GameObject agentObject;                  // Reference to agent GameObject
public string agentTag = "Agent";              // Tag to identify agent

[Header("Map Initializer Reference")]
public MapInitializer mapInitializer;          // Reference for agent initial data
```

**Agent Data Integration:**
- **Current Position**: Agent's current transform position and rotation
- **Initial Position**: Agent's initial position from MapInitializer
- **Automatic Detection**: Finds agent by tag, name, or AgentDataRecorder component

### Map and Vehicle Management

#### MapInitializer.cs
Handles map initialization and vehicle management.

**Key Features:**
- **Vehicle Templates**: Black, red, and white car prefabs
- **Position-Based Spawning**: Loads vehicle positions from JSON files
- **Agent Management**: Sets up agent initial position and rotation
- **Audio Integration**: Configures spatial audio for target vehicles

**Configuration:**
```csharp
[Header("Vehicle Prefabs")]
public GameObject carBlackPrefab;              // Black vehicle prefab
public GameObject carRedPrefab;                // Red vehicle prefab
public GameObject carWhitePrefab;              // White vehicle prefab

[Header("Agent Settings")]
public GameObject agent;                       // Agent prefab
public Transform mapRoot;                      // Map root node

[Header("Position File")]
public string positionFileName = "scene_data_YYYYMMDD_HHMMSS.json";
```

### Communication and Control

#### SocketServer.cs
TCP server for Python communication.

**Key Features:**
- **Command Processing**: Handles movement, observation, and map commands
- **Thread-Safe Communication**: Multi-threaded client handling
- **Auto-Detection**: Automatically finds MapInitializer component

#### AgentControl.cs
Agent movement and behavior control.

**Key Features:**
- **Movement Commands**: Move to position, turn left/right
- **State Management**: Position and rotation tracking
- **Action Execution**: Processes movement commands from Python

#### VisualCapture.cs
Visual observation capture system.

**Key Features:**
- **Object Detection**: Finds objects within camera view
- **Distance Calculation**: Calculates distances and angles to objects
- **Visibility Detection**: Determines if objects are visible
- **Categorization**: Tags objects by type (Vehicles, etc.)

### Utility Scripts

#### MovementMap.cs
Navigation and movement planning system.

#### LaunchManager.cs
Application launch and mode management.

#### FirstPersonControl.cs
Manual camera control for testing.

## 🐍 Python Client

### UnityControlClient
Main Python client for Unity communication.

**Key Features:**
- **Socket Communication**: TCP connection to Unity server
- **Map Management**: Initialize maps with vehicles and agent
- **Agent Control**: Send movement commands
- **Data Retrieval**: Get visual and audio observations

**Basic Usage:**
```python
from control_client import UnityControlClient

# Initialize client
client = UnityControlClient(host="127.0.0.1", port=5005)

# Connect to Unity
if client.connect():
    print("Connected to Unity server")
    
    # Initialize map
    client.initialize_map()
    
    # Run agent loop
    client.run_agent_loop(num_iterations=5)
    
    # Disconnect
    client.disconnect()
```

## 🚀 Getting Started

### Prerequisites

1. **Unity 2022.3 LTS or later**
2. **Python 3.8+**
3. **Required Python packages:**
   ```bash
   pip install matplotlib numpy
   ```

### Setup Instructions

1. **Unity Setup:**
   - Open the Unity project
   - Ensure all scripts are in the correct location
   - Configure MapInitializer with vehicle templates
   - Set up AgentDataRecorder with proper userId
   - Enable VehiclePositionRecorder component

2. **Python Setup:**
   - Install required Python packages
   - Ensure `control_client.py` is in your Python path
   - Configure Unity server IP and port if needed

3. **Configuration:**
   - Set userId in AgentDataRecorder
   - Configure recording intervals and data types
   - Set up vehicle templates in MapInitializer

### Running the System

1. **Start Unity:**
   - Open the project in Unity
   - Ensure VehiclePositionRecorder component is enabled
   - Run the scene

2. **Run Python Client:**
   ```python
   from control_client import UnityControlClient
   
   client = UnityControlClient()
   client.connect()
   client.initialize_map()
   client.run_agent_loop(num_iterations=5)
   ```

## 🔧 Configuration

### Unity Configuration

#### AgentDataRecorder Setup
```csharp
// Set user ID for data organization
public string userId = "user001";

// Configure recording settings
public float recordingInterval = 2.0f;
public bool recordPosition = true;
public bool recordRotation = true;
public bool recordAudio = true;
public bool recordVisualData = true;
public bool recordAudioToFile = true;

// Assign components
public Camera agentCamera;
public AudioListener audioListener;
public VisualCapture visualCapture;
```

#### VehiclePositionRecorder Setup
```csharp
// File naming
public string fileName = "scene_data";
public bool includeTimestamp = true;

// Data inclusion
public bool includeRotation = true;
public bool includeScale = true;
public bool recordAgentData = true;

// Agent settings
public string agentTag = "Agent";
// mapInitializer will be auto-detected
```

#### MapInitializer Setup
```csharp
// Vehicle templates
public GameObject carBlackPrefab;
public GameObject carRedPrefab;
public GameObject carWhitePrefab;

// Agent and map
public GameObject agent;
public Transform mapRoot;
public string positionFileName = "scene_data_YYYYMMDD_HHMMSS.json";
```

### Python Configuration

```python
# Connection settings
host = "127.0.0.1"
port = 5005

# File paths (update with actual file names)
position_file = "Assets/StreamingAssets/user001/scene_data_YYYYMMDD_HHMMSS.json"
```

## 🐛 Troubleshooting

### Common Issues

1. **No scene_data files generated:**
   - Ensure VehiclePositionRecorder component is enabled
   - Check that AgentDataRecorder has userId set
   - Verify MapInitializer is present in scene

2. **Data recording issues:**
   - Check userId configuration in AgentDataRecorder
   - Ensure VisualCapture component is assigned
   - Verify StreamingAssets folder is writable

3. **Agent data missing:**
   - Check that agent objects are properly tagged
   - Verify MapInitializer has agent reference
   - Ensure VehiclePositionRecorder can find agent

4. **File organization issues:**
   - Verify userId is consistent across components
   - Check that user subfolders are created
   - Ensure file naming conventions are followed

### Debug Information

Enable debug logging in Unity:
```csharp
// In MapInitializer
public bool logInitialization = true;
public bool showGizmos = true;

// In VehiclePositionRecorder
// Check Console for debug messages about file paths and data recording
```

## 📝 API Reference

### Key Methods

| Script | Method | Description |
|--------|--------|-------------|
| AgentDataRecorder | `RecordDataPoint()` | Record multi-modal data point |
| VehiclePositionRecorder | `RecordSceneData()` | Record complete scene snapshot |
| MapInitializer | `InitializeMap()` | Initialize map with vehicles |
| VisualCapture | `CaptureVisualObservation()` | Capture visual observations |
| SocketServer | `HandleClient()` | Handle client communication |

### Data Flow

1. **Scene Initialization**: MapInitializer sets up vehicles and agent
2. **Data Recording**: AgentDataRecorder records agent observations
3. **Scene Snapshots**: VehiclePositionRecorder captures complete scene state
4. **File Organization**: All data organized by user ID in StreamingAssets
5. **Python Access**: control_client.py can read and process recorded data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Note:** This system is designed for research and development purposes in audiovisual search applications. Ensure proper testing before using in production environments.
