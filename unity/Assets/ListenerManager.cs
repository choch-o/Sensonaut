using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ListenerManager : MonoBehaviour
{
    // Buffer for audio samples
    private float[] audioSamples = new float[1024];

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        // Read audio data from the AudioListener
        AudioListener.GetOutputData(audioSamples, 0);
        // Now audioSamples contains the latest audio data from the listener
    }
}
