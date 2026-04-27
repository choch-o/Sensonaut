using System.IO;
using UnityEngine;
using System.Collections;

public class AudioCapture : MonoBehaviour
{
    [Header("Audio Recording Settings")]
    public string fileName = "recorded_audio.wav";
    public float recordingDuration = 0.1f;  // Fixed recording duration in seconds
    public bool autoStartRecording = true;

    
    private FileStream fileStream;
    private string fullPath;
    private bool isRecording = false;
    private Coroutine recordingCoroutine;

    void Start()
    {
        // Use Application.persistentDataPath for runtime file access
        fullPath = Path.Combine(Application.persistentDataPath, fileName);
        Debug.Log($"AudioCapture initialized. File path: {fullPath}");
        Debug.Log($"Persistent data path: {Application.persistentDataPath}");
        
        // Check if AudioListener exists
        AudioListener audioListener = GetComponent<AudioListener>();
        if (audioListener == null)
        {
            Debug.LogWarning("AudioListener not found, adding one...");
            gameObject.AddComponent<AudioListener>();
        }
        else
        {
            Debug.Log("AudioListener found");
        }
        
        if (autoStartRecording)
        {
            StartRecording();
        }
    }

    [ContextMenu("Start Recording")]
    public void StartRecording()
    {
        if (isRecording) return;
        
        try
        {
            // Close and dispose existing stream if it exists
            if (fileStream != null)
            {
                fileStream.Close();
                fileStream.Dispose();
                fileStream = null;
                Debug.Log("Closed and disposed existing file stream");
            }
            
            // Ensure directory exists
            string directory = Path.GetDirectoryName(fullPath);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
                Debug.Log($"Created directory: {directory}");
            }
            
            // Delete existing file if it exists
            if (File.Exists(fullPath))
            {
                File.Delete(fullPath);
                Debug.Log($"Deleted existing audio file: {fullPath}");
            }
            
            fileStream = new FileStream(fullPath, FileMode.Create);
            WriteWavHeader(); // Write placeholder header
            isRecording = true;
            Debug.Log($"Audio recording started: {fullPath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to start audio recording: {e.Message}");
        }
    }

    [ContextMenu("Stop Recording")]
    public void StopRecording()
    {
        if (!isRecording) return;
        
        try
        {
            UpdateWavHeader(); // Update WAV file header
            fileStream?.Close();
            fileStream?.Dispose();
            isRecording = false;
            
            // Check if file was created
            if (File.Exists(fullPath))
            {
                long fileSize = new FileInfo(fullPath).Length;
                Debug.Log($"Audio recording stopped. File size: {fileSize} bytes");
            }
            else
            {
                Debug.LogWarning("Audio recording stopped but file was not created");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to stop audio recording: {e.Message}");
        }
    }

    public void RecordForDuration()
    {
        // Stop any existing recording
        if (recordingCoroutine != null)
        {
            StopCoroutine(recordingCoroutine);
        }
        
        // Start new recording for fixed duration
        recordingCoroutine = StartCoroutine(RecordForDurationCoroutine());
    }

    private IEnumerator RecordForDurationCoroutine()
    {
        Debug.Log($"Starting recording for {recordingDuration} seconds...");
        
        // Start recording
        StartRecording();
        
        // Wait for fixed duration
        yield return new WaitForSeconds(recordingDuration);
        
        // Stop recording
        StopRecording();
        
        recordingCoroutine = null;
        Debug.Log("Recording coroutine completed");
    }

    [ContextMenu("Record for Duration")]
    public void RecordForDurationMenu()
    {
        RecordForDuration();
    }

    void OnAudioFilterRead(float[] data, int channels)
    {
        if (!isRecording || fileStream == null) return;
        
        try
        {
            // Convert float audio [-1,1] to 16bit PCM
            byte[] byteData = new byte[data.Length * 2];
            int rescaleFactor = 32767;

            for (int i = 0; i < data.Length; i++)
            {
                short val = (short)(data[i] * rescaleFactor);
                byteData[i * 2] = (byte)(val & 0xff);
                byteData[i * 2 + 1] = (byte)((val >> 8) & 0xff);
            }

            fileStream.Write(byteData, 0, byteData.Length);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error writing audio data: {e.Message}");
            StopRecording();
        }
    }

    void OnApplicationQuit()
    {
        StopRecording();
    }

    void OnDestroy()
    {
        StopRecording();
    }

    void WriteWavHeader()
    {
        // Write 44-byte empty header (placeholder)
        byte[] emptyHeader = new byte[44];
        fileStream.Write(emptyHeader, 0, emptyHeader.Length);
    }

    void UpdateWavHeader()
    {
        if (fileStream == null) return;
        
        try
        {
            long fileSize = fileStream.Length;

            fileStream.Seek(0, SeekOrigin.Begin);
            BinaryWriter writer = new BinaryWriter(fileStream);

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
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to update WAV header: {e.Message}");
        }
    }

    // Get the full path of the recorded audio file
    public string GetAudioFilePath()
    {
        return fullPath;
    }

    // Check if recording is active
    public bool IsRecording()
    {
        return isRecording;
    }
}
