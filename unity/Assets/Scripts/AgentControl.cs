using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AgentControl : MonoBehaviour
{
    private string currentAction = "stay";
    private AudioCapture audioCapture;
    private VisualCapture visualCapture;
    private MovementMap movementMap;
    private Rigidbody rb;

    void Start()
    {
        
        // Get AudioCapture component
        audioCapture = GetComponent<AudioCapture>();
        if (audioCapture == null)
        {
            Debug.LogWarning("AudioCapture component not found on this GameObject");
        }
        
        // Get VisualCapture component
        visualCapture = GetComponent<VisualCapture>();
        if (visualCapture == null)
        {
            Debug.LogWarning("VisualCapture component not found on this GameObject");
        }
        
        // Get MovementMap component
        movementMap = GetComponent<MovementMap>();
        if (movementMap == null)
        {
            // Try to find MovementMap in the scene
            movementMap = FindObjectOfType<MovementMap>();
            if (movementMap == null)
            {
                Debug.LogWarning("MovementMap component not found in scene");
            }
        }
    }

    public void SetAction(string jsonString)
    {
        var cmd = JsonUtility.FromJson<ActionCommand>(jsonString);
        currentAction = cmd.command.ToLower();

        switch (currentAction)
        {
            case "move_forward":
                Debug.Log("Moving forward");
                Vector3 newPos = new Vector3(cmd.target_position[0], transform.position.y, cmd.target_position[2]);
                transform.position = newPos;
                break;

            case "turn_left":
                Debug.Log($"Turning left by {cmd.angle} degrees");
                Debug.Log($"Before rotation: {transform.rotation.eulerAngles}");
               
                transform.Rotate(Vector3.up, -cmd.angle);
                
                Debug.Log($"After rotation: {transform.rotation.eulerAngles}");
                break;

            case "turn_right":
                Debug.Log($"Turning right by {cmd.angle} degrees");
                Debug.Log($"Before rotation: {transform.rotation.eulerAngles}");
                
                transform.Rotate(Vector3.up, cmd.angle);
                
                Debug.Log($"After rotation: {transform.rotation.eulerAngles}");
                break;

            case "commit":
                Debug.Log("Committing current action");
                // TODO: Commit the current action and move to the next episode
                break;

            case "stay":
                Debug.Log("Staying in place");
                // Do nothing, agent stays in place
                break;

            case "get_observation":
                // Capture both audio and visual observations immediately
                if (visualCapture != null)
                {
                    Debug.Log("Capturing visual observation immediately");
                    visualCapture.CaptureVisualObservation();
                    Debug.Log("Visual observation captured immediately");
                }
                else
                {
                    Debug.LogWarning("VisualCapture component not available for observation");
                }
                
                if (audioCapture != null)
                {
                    audioCapture.RecordForDuration();
                    Debug.Log("Started audio recording for observation");
                }
                else
                {
                    Debug.LogWarning("AudioCapture component not available for observation");
                }
                break;

            default:
                Debug.LogWarning($"Unknown action: {currentAction}");
                break;
        }

        currentAction = "stay";
    }
    
    public string GetVisualObservationJson()
    {
        if (visualCapture != null)
        {
            return visualCapture.GetVisualObservationJson();
        }
        return "{}";
    }
    
    public VisualObservationData GetVisualObservationData()
    {
        if (visualCapture != null)
        {
            return visualCapture.GetVisualObservationData();
        }
        return null;
    }


    [System.Serializable]
    public class ActionCommand
    {
        public string command;
        public float[] target_position;
        public float angle;
    }
}