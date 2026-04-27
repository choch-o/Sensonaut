using System.IO;
using UnityEngine;

[RequireComponent(typeof(AudioListener))]
public class BinauralRecorder : MonoBehaviour
{
    public string outputFile = "Assets/rir_output.wav";
    private MemoryStream leftChannel = new MemoryStream();
    private MemoryStream rightChannel = new MemoryStream();
    private bool recording = true;
    private int sampleRate;

    void Start()
    {
        sampleRate = AudioSettings.outputSampleRate;
        Debug.Log($"Recording at {sampleRate} Hz");
    }

    void OnAudioFilterRead(float[] data, int channels)
    {
        if (!recording || channels != 2) return;

        byte[] left = new byte[4];
        byte[] right = new byte[4];

        for (int i = 0; i < data.Length; i += 2)
        {
            System.Buffer.BlockCopy(System.BitConverter.GetBytes(data[i]), 0, left, 0, 4);
            System.Buffer.BlockCopy(System.BitConverter.GetBytes(data[i + 1]), 0, right, 0, 4);
            leftChannel.Write(left, 0, 4);
            rightChannel.Write(right, 0, 4);
        }
    }

    void OnDisable()
    {
        recording = false;
        SaveWav();
    }

    void SaveWav()
    {
        Debug.Log("Saving RIR to WAV file...");

        var leftData = leftChannel.ToArray();
        var rightData = rightChannel.ToArray();
        int totalSamples = leftData.Length / 4;

        using (var file = new FileStream(outputFile, FileMode.Create))
        using (var writer = new BinaryWriter(file))
        {
            int byteRate = sampleRate * 2 * 2;

            // Write WAV header
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(36 + totalSamples * 4); // File size - 8
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));
            writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            writer.Write(16); // Subchunk1Size
            writer.Write((short)1); // AudioFormat
            writer.Write((short)2); // NumChannels
            writer.Write(sampleRate);
            writer.Write(byteRate);
            writer.Write((short)4); // BlockAlign
            writer.Write((short)16); // BitsPerSample

            writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            writer.Write(totalSamples * 4); // NumSamples * NumChannels * BytesPerSample

            // Interleave stereo channels
            for (int i = 0; i < totalSamples; ++i)
            {
                writer.Write(leftData, i * 4, 4);
                writer.Write(rightData, i * 4, 4);
            }
        }

        Debug.Log($"Saved stereo RIR as {outputFile}");
    }
}
