using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

public class AgentDataRecorder : MonoBehaviour
{
    [Header("File Settings")]
    public string userId = "user001";  // User ID for creating subfolder
    public string mapId = "001"; // Map ID for data collection
    
    [Header("Recording Settings")]
    public float recordingInterval = 0.1f; // Recording interval and audio duration (seconds)
    public bool recordPosition = true;
    public bool recordRotation = true;
    public bool recordAudio = true;
    public bool recordVisualData = true;
    
    [Header("Audio Settings")]
    public bool recordAudioToFile = true; // Whether to record audio to file
    
    [Header("Components")]
    public Camera agentCamera;
    public AudioListener audioListener;
    public VisualCapture visualCapture;
    
    private float lastRecordTime;
    private int dataPointIndex = 0;  // Data point index
    private bool isRecording = false;
    
    // Recording related
    private string saveFolder;
    private FileStream audioFileStream;
    private string audioFilePath;
    private bool isAudioRecording = false;
    private Coroutine audioRecordingCoroutine;
    
    [System.Serializable]
    public class AgentDataPoint
    {
        public string timestamp;
        public Vector3 position;
        public Vector3 rotation;
        public string audioFileName; // Audio filename
        public VisualObservationData visualData;
    }
    
    // AgentDataCollection class removed because no longer needed
    
    void Start()
    {
        InitializeComponents();
    }
    
    /// <summary>
    /// Set save folder from external source (e.g., DataCollectionController)
    /// </summary>
    public void SetSaveFolder(string newSaveFolder)
    {
        if (!string.IsNullOrEmpty(newSaveFolder) && newSaveFolder != saveFolder)
        {
            saveFolder = newSaveFolder;
            Debug.Log($"AgentDataRecorder: Save folder set to {saveFolder}");
        }
    }
    
    void Update()
    {
        if (isRecording && Time.time - lastRecordTime >= recordingInterval)
        {
            RecordDataPoint();
            lastRecordTime = Time.time;
        }
    }

    
    private void InitializeComponents()
    {
        // Auto-get components
        if (agentCamera == null)
            agentCamera = GetComponent<Camera>();
        
        if (audioListener == null)
            audioListener = GetComponent<AudioListener>();
        
        if (visualCapture == null)
            visualCapture = GetComponent<VisualCapture>();
        
        if (agentCamera == null)
        {
            Debug.LogError("AgentDataRecorder: No camera found!");
        }
    }

    
    // GenerateFilePath method removed because each data point is saved individually
    
    public void StartRecording()
    {
        isRecording = true;
        lastRecordTime = Time.time;
        Debug.Log("Agent data recording started");
    }
    
    public void StopRecording()
    {
        isRecording = false;
        Debug.Log("Agent data recording stopped");
    }
    
    private void RecordDataPoint()
    {
        if (saveFolder == null)
        {
            saveFolder = Path.Combine(Application.streamingAssetsPath, userId, mapId);
            Debug.Log($"AgentDataRecorder: Default save folder initialized to {saveFolder}");
        }
        
        // Stop previous audio recording if it's still running
        if (recordAudioToFile && isAudioRecording)
        {
            StopAudioRecording();
        }
        
        // Start new audio recording for this data point
        if (recordAudioToFile)
        {
            StartAudioRecording();
        }
        
        // Capture visual observation if enabled
        if (recordVisualData && visualCapture != null)
        {
            visualCapture.CaptureVisualObservation();
        }
        
        var dataPoint = new AgentDataPoint
        {
            timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
            position = transform.position,
            rotation = transform.rotation.eulerAngles,
            audioFileName = recordAudioToFile ? GetCurrentAudioFileName() : null,
            visualData = recordVisualData && visualCapture != null ? visualCapture.currentVisualObservation : null,
        };
        
        // Save this data point immediately
        SaveDataPoint(dataPoint);
        
        dataPointIndex++;
        Debug.Log($"Recorded and saved data point {dataPointIndex}");
    }
    
    private string GetCurrentAudioFileName()
    {
        // Return current audio filename (without path)
        return Path.GetFileName(audioFilePath);
    }
    
    // Save single data point
    private void SaveDataPoint(AgentDataPoint dataPoint)
    {
        try
        {
            // Generate filename (with index)
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string fileName = $"agent_data_{timestamp}_{dataPointIndex:D4}";
            
            
            // Ensure user subfolder exists
            if (!Directory.Exists(saveFolder))
            {
                Directory.CreateDirectory(saveFolder);
            }
            
            // Save JSON file
            string jsonPath = Path.Combine(saveFolder, fileName + ".json");
            string json = JsonUtility.ToJson(dataPoint, true);
            File.WriteAllText(jsonPath, json);
            
            Debug.Log($"Data point saved: {jsonPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to save data point: {e.Message}");
        }
    }
    
    // Audio recording method
    private void StartAudioRecording()
    {
        if (isAudioRecording) return;
        
        try
        {
            // Generate audio filename (with index)
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string audioFileName = $"agent_audio_{timestamp}_{dataPointIndex:D4}.wav";
            
            audioFilePath = Path.Combine(saveFolder, audioFileName);
            
            // Ensure user subfolder exists
            if (!Directory.Exists(saveFolder))
            {
                Directory.CreateDirectory(saveFolder);
            }
            
            // Start audio recording
            audioFileStream = new FileStream(audioFilePath, FileMode.Create);
            WriteWavHeader();
            isAudioRecording = true;
            
            
            Debug.Log($"Audio recording started: {audioFilePath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to start audio recording: {e.Message}");
        }
    }
    
    private System.Collections.IEnumerator RecordAudioForDuration()
    {
        yield return new WaitForSeconds(recordingInterval);
        StopAudioRecording();
    }
    
    private void StopAudioRecording()
    {
        if (!isAudioRecording) return;
        
        try
        {
            UpdateWavHeader();
            audioFileStream?.Close();
            audioFileStream?.Dispose();
            isAudioRecording = false;
            
            Debug.Log($"Audio recording stopped: {audioFilePath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to stop audio recording: {e.Message}");
        }
    }
    
    // WAV file header handling
    private void WriteWavHeader()
    {
        if (audioFileStream == null) return;
        
        // Write 44-byte empty header (placeholder)
        byte[] emptyHeader = new byte[44];
        audioFileStream.Write(emptyHeader, 0, emptyHeader.Length);
    }
    
    private void UpdateWavHeader()
    {
        if (audioFileStream == null) return;
        
        try
        {
            long fileSize = audioFileStream.Length;
            audioFileStream.Seek(0, SeekOrigin.Begin);
            
            using (BinaryWriter writer = new BinaryWriter(audioFileStream))
            {
                writer.Write(System.Text.Encoding.UTF8.GetBytes("RIFF"));
                writer.Write((int)(fileSize - 8));
                writer.Write(System.Text.Encoding.UTF8.GetBytes("WAVE"));
                writer.Write(System.Text.Encoding.UTF8.GetBytes("fmt "));
                writer.Write(16);  // PCM
                writer.Write((short)1);  // Linear PCM
                writer.Write((short)2);  // channels
                writer.Write(48000);  // sample rate
                writer.Write(48000 * 2 * 2);  // byte rate
                writer.Write((short)(2 * 2));  // block align
                writer.Write((short)16);  // bits per sample
                writer.Write(System.Text.Encoding.UTF8.GetBytes("data"));
                writer.Write((int)(fileSize - 44));
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to update WAV header: {e.Message}");
        }
    }
    
    // Audio data writing (Unity's OnAudioFilterRead callback)
    void OnAudioFilterRead(float[] data, int channels)
    {
        if (!isAudioRecording || audioFileStream == null) return;
        
        try
        {
            // Convert float audio [-1,1] to 16-bit PCM
            byte[] byteData = new byte[data.Length * 2];
            int rescaleFactor = 32767;

            for (int i = 0; i < data.Length; i++)
            {
                short val = (short)(data[i] * rescaleFactor);
                byteData[i * 2] = (byte)(val & 0xff);
                byteData[i * 2 + 1] = (byte)((val >> 8) & 0xff);
            }

            audioFileStream.Write(byteData, 0, byteData.Length);
        }
        catch (Exception e)
        {
            Debug.LogError($"Error writing audio data: {e.Message}");
            StopAudioRecording();
        }
    }
    
}
