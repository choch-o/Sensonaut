using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class VisualCapture : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera agentCamera; // Reference to the agent's camera
    
    [Header("Visual Observation Settings")]
    public int maxObjectsToDetect = 10; // Maximum number of objects to include in observation
    
    [Header("Debug Settings")]
    public bool showDebugGizmos = false; // Show FOV visualization in scene view
    public bool logDetectionInfo = true; // Log detection information to console
    public int coneSegments = 16; // Number of segments for cone visualization
    
    [Header("Observation Data")]
    public VisualObservationData currentVisualObservation;
    
    void Start()
    {
        // Try to find camera if not assigned
        if (agentCamera == null)
        {
            agentCamera = GetComponentInChildren<Camera>();
            if (agentCamera == null)
            {
                agentCamera = GetComponent<Camera>();
            }
        }
        
        if (agentCamera != null)
        {
            Debug.Log($"VisualCapture: Using camera FOV ({agentCamera.fieldOfView}°) for detection");
        }
        else
        {
            Debug.LogError("VisualCapture: No camera found!");
        }
    }
    
    /// <summary>
    /// Capture visual observation using camera-based or manual cone field of view
    /// </summary>
    public void CaptureVisualObservation()
    {
        currentVisualObservation = new VisualObservationData();
        currentVisualObservation.timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        currentVisualObservation.agentPosition = agentCamera.transform.position;
        currentVisualObservation.agentRotation = agentCamera.transform.rotation.eulerAngles;
        
        // Capture objects using camera-based or manual cone field of view
        List<DetectedObject> detectedObjects = CaptureConeFOV();
        
        // Sort by distance and limit the number of objects
        detectedObjects.Sort((a, b) => a.distance.CompareTo(b.distance));
        if (detectedObjects.Count > maxObjectsToDetect)
        {
            detectedObjects.RemoveRange(maxObjectsToDetect, detectedObjects.Count - maxObjectsToDetect);
        }
        
        currentVisualObservation.detectedObjects = detectedObjects.ToArray();
        
        if (logDetectionInfo)
        {
            Debug.Log($"VisualCapture: {detectedObjects.Count} vehicles detected using camera FOV ({agentCamera.fieldOfView}°) within {agentCamera.farClipPlane} units");
        }
    }
    
    /// <summary>
    /// Capture objects using camera's actual rendering information
    /// </summary>
    private List<DetectedObject> CaptureConeFOV()
    {
        List<DetectedObject> detectedObjects = new List<DetectedObject>();
        
        if (agentCamera == null)
        {
            Debug.LogError("VisualCapture: No camera assigned!");
            return detectedObjects;
        }
        
        // Use Camera's culling and rendering to detect objects
        detectedObjects = GetObjectsVisibleToCamera();
        
        if (logDetectionInfo)
        {
            Debug.Log($"VisualCapture: {detectedObjects.Count} vehicles detected using camera-based method");
        }
        
        return detectedObjects;
    }
    
    /// <summary>
    /// Get objects that are actually visible to the camera using rendering information
    /// </summary>
    private List<DetectedObject> GetObjectsVisibleToCamera()
    {
        List<DetectedObject> visibleObjects = new List<DetectedObject>();
        
        // Get all renderers in the scene
        Renderer[] allRenderers = FindObjectsOfType<Renderer>();
        
        if (logDetectionInfo)
        {
            Debug.Log($"VisualCapture: Found {allRenderers.Length} total renderers in scene");
        }
        
        foreach (Renderer renderer in allRenderers)
        {
            // Skip if this is part of the agent
            if (renderer.transform.IsChildOf(transform))
            {
                if (logDetectionInfo)
                {
                    // Debug.Log($"VisualCapture: Skipping agent's own renderer: {renderer.name}");
                }
                continue;
            }
            
                    // Only detect objects with tag "Vehicles"
        if (renderer.gameObject.tag != "Vehicles")
            {
                if (logDetectionInfo)
                {
                    // Debug.Log($"VisualCapture: Skipping non-vehicle object: {renderer.gameObject.name} (tag: {renderer.gameObject.tag})");
                }
                continue;
            }
            
            // Check if the renderer is visible to the camera (this includes FOV and distance)
            if (IsRendererVisibleToCamera(renderer))
            {
                Vector3 direction = renderer.transform.position - agentCamera.transform.position;
                float distance = direction.magnitude;
                float angle = Vector3.SignedAngle(agentCamera.transform.forward, direction, Vector3.up);
                
                // Get the collider for additional information (optional)
                Collider collider = renderer.GetComponent<Collider>();
                
                // If object is in camera frustum, it's considered visible
                bool isVisible = true;
                
                // Sanitize object name and tag
                string sanitizedName = SanitizeString(renderer.gameObject.name);
                string sanitizedTag = SanitizeString(renderer.gameObject.tag);
                
                // Check if we already have this vehicle (to avoid duplicates)
                bool isDuplicate = false;
                foreach (var existingObj in visibleObjects)
                {
                    if (existingObj.name == sanitizedName && 
                        Vector3.Distance(existingObj.position, renderer.transform.position) < 0.1f)
                    {
                        isDuplicate = true;
                                        if (logDetectionInfo)
                {
                    // Debug.Log($"VisualCapture: Skipping duplicate vehicle renderer for {sanitizedName}");
                }
                        break;
                    }
                }
                
                if (!isDuplicate)
                {
                    DetectedObject obj = new DetectedObject
                    {
                        name = sanitizedName,
                        tag = sanitizedTag,
                        distance = distance,
                        angle = angle,
                        position = renderer.transform.position,
                        size = renderer.bounds.size,
                        isVisible = isVisible,
                        layer = renderer.gameObject.layer
                    };
                    
                    visibleObjects.Add(obj);
                    
                    if (logDetectionInfo)
                    {
                        // Debug.Log($"VisualCapture: Detected vehicle {sanitizedName} at distance {distance:F2}, angle {angle:F2}, visible: {isVisible}, hasCollider: {collider != null}");
                    }
                }
            }
            else
            {
                if (logDetectionInfo)
                {
                    // Debug.Log($"VisualCapture: Vehicle {renderer.gameObject.name} is NOT visible to camera");
                }
            }
        }
        
        if (logDetectionInfo)
        {
            Debug.Log($"VisualCapture: Total vehicles found: {visibleObjects.Count}");
        }
        
        return visibleObjects;
    }
    
    /// <summary>
    /// Check if a renderer is visible to the camera using GeometryUtility
    /// This single check includes FOV angle and distance range
    /// </summary>
    private bool IsRendererVisibleToCamera(Renderer renderer)
    {
        if (renderer == null || !renderer.enabled || !renderer.gameObject.activeInHierarchy)
            return false;
            
        // This single check includes:
        // 1. FOV angle (fieldOfView)
        // 2. Distance range (nearClipPlane to farClipPlane)
        // 3. Camera frustum bounds
        bool inFrustum = GeometryUtility.TestPlanesAABB(GeometryUtility.CalculateFrustumPlanes(agentCamera), renderer.bounds);
        
        if (logDetectionInfo && inFrustum)
        {
            Debug.Log($"VisualCapture: {renderer.name} is in camera frustum");
        }
        
        return inFrustum;
    }
    

    
    /// <summary>
    /// Sanitize string to prevent JSON serialization issues
    /// </summary>
    private string SanitizeString(string input)
    {
        if (string.IsNullOrEmpty(input))
            return "Unknown";
            
        // Remove or replace problematic characters
        string sanitized = input
            .Replace("\"", "'")  // Replace quotes with apostrophes
            .Replace("\\", "/")  // Replace backslashes with forward slashes
            .Replace("\n", " ")  // Replace newlines with spaces
            .Replace("\r", " ")  // Replace carriage returns with spaces
            .Replace("\t", " "); // Replace tabs with spaces
            
        // Limit length to prevent very long strings
        if (sanitized.Length > 50)
        {
            sanitized = sanitized.Substring(0, 47) + "...";
        }
        
        return sanitized;
    }
    

    
    /// <summary>
    /// Get the current visual observation as JSON string
    /// </summary>
    public string GetVisualObservationJson()
    {
        if (currentVisualObservation != null)
        {
            try
            {
                string json = JsonUtility.ToJson(currentVisualObservation, true);
                
                // Validate JSON by trying to parse it back
                try
                {
                    JsonUtility.FromJson<VisualObservationData>(json);
                    return json;
                }
                catch (System.Exception parseError)
                {
                    Debug.LogError($"JSON validation failed: {parseError.Message}");
                    // Return a safe fallback JSON
                    return CreateSafeFallbackJson();
                }
            }
            catch (System.Exception jsonError)
            {
                Debug.LogError($"JSON serialization failed: {jsonError.Message}");
                return CreateSafeFallbackJson();
            }
        }
        return "{}";
    }
    
    /// <summary>
    /// Create a safe fallback JSON when serialization fails
    /// </summary>
    private string CreateSafeFallbackJson()
    {
        return "{\"timestamp\":\"" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + "\",\"agentPosition\":{\"x\":0,\"y\":0,\"z\":0},\"agentRotation\":{\"x\":0,\"y\":0,\"z\":0},\"detectedObjects\":[]}";
    }
    
    /// <summary>
    /// Get the current visual observation data
    /// </summary>
    public VisualObservationData GetVisualObservationData()
    {
        return currentVisualObservation;
    }
    
    /// <summary>
    /// Clear the current visual observation
    /// </summary>
    public void ClearVisualObservation()
    {
        currentVisualObservation = null;
    }
    
    /// <summary>
    /// Get objects within a specific distance
    /// </summary>
    public DetectedObject[] GetObjectsWithinDistance(float maxDistance)
    {        
        List<DetectedObject> nearbyObjects = new List<DetectedObject>();
        if (currentVisualObservation == null || currentVisualObservation.detectedObjects == null)
            return nearbyObjects.ToArray();
        foreach (DetectedObject obj in currentVisualObservation.detectedObjects)
        {
            if (obj.distance <= maxDistance)
            {
                nearbyObjects.Add(obj);
            }
        }
        
        return nearbyObjects.ToArray();
    }
    
    /// <summary>
    /// Get visible objects only
    /// </summary>
    public DetectedObject[] GetVisibleObjects()
    {
        List<DetectedObject> visibleObjects = new List<DetectedObject>();
        if (currentVisualObservation == null || currentVisualObservation.detectedObjects == null)
            return visibleObjects.ToArray();
        foreach (DetectedObject obj in currentVisualObservation.detectedObjects)
        {
            if (obj.isVisible)
            {
                visibleObjects.Add(obj);
            }
        }
        
        return visibleObjects.ToArray();
    }
    
    /// <summary>
    /// Get objects within a specific angle range (relative to forward direction)
    /// </summary>
    public DetectedObject[] GetObjectsInFieldOfView(float maxAngle)
    {   
        List<DetectedObject> fovObjects = new List<DetectedObject>();
        if (currentVisualObservation == null || currentVisualObservation.detectedObjects == null)
            return fovObjects.ToArray();
        foreach (DetectedObject obj in currentVisualObservation.detectedObjects)
        {
            if (Mathf.Abs(obj.angle) <= maxAngle)
            {
                fovObjects.Add(obj);
            }
        }
        
        return fovObjects.ToArray();
    }
    

    
    // Debug visualization
    void OnDrawGizmosSelected()
    {
        if (showDebugGizmos)
        {
            DrawConeFOVGizmos();
        }
    }
    
    /// <summary>
    /// Draw cone FOV visualization gizmos
    /// </summary>
    private void DrawConeFOVGizmos()
    {
        if (agentCamera == null) return;
        
        // Get current FOV parameters
        float currentFOV = agentCamera.fieldOfView;
        float currentDistance = agentCamera.farClipPlane;
        Vector3 cameraPosition = agentCamera.transform.position;
        Vector3 cameraForward = agentCamera.transform.forward;
        
        // Draw cone outline
        Gizmos.color = Color.cyan;
        DrawConeOutline(cameraPosition, cameraForward, currentDistance, currentFOV, coneSegments);
        
        // Draw detected objects
        if (currentVisualObservation != null && currentVisualObservation.detectedObjects != null)
        {
            foreach (DetectedObject obj in currentVisualObservation.detectedObjects)
            {
                Gizmos.color = Color.green;
                Gizmos.DrawLine(cameraPosition, obj.position);
                Gizmos.DrawWireCube(obj.position, obj.size);
            }
        }
    }
    
    /// <summary>
    /// Draw a cone outline for FOV visualization
    /// </summary>
    private void DrawConeOutline(Vector3 center, Vector3 direction, float distance, float angle, int segments)
    {
        float halfAngle = angle * 0.5f * Mathf.Deg2Rad;
        Vector3 right = Vector3.Cross(direction, Vector3.up).normalized;
        Vector3 up = Vector3.Cross(right, direction).normalized;
        
        // Draw cone base circle
        Vector3[] basePoints = new Vector3[segments];
        for (int i = 0; i < segments; i++)
        {
            float t = (float)i / segments * 2f * Mathf.PI;
            float x = Mathf.Sin(t) * Mathf.Sin(halfAngle) * distance;
            float y = Mathf.Cos(t) * Mathf.Sin(halfAngle) * distance;
            basePoints[i] = center + direction * distance * Mathf.Cos(halfAngle) + right * x + up * y;
        }
        
        // Draw base circle
        for (int i = 0; i < segments; i++)
        {
            Gizmos.DrawLine(basePoints[i], basePoints[(i + 1) % segments]);
        }
        
        // Draw lines from center to base
        for (int i = 0; i < segments; i++)
        {
            Gizmos.DrawLine(center, basePoints[i]);
        }
    }
}

[System.Serializable]
public class VisualObservationData
{
    public string timestamp;
    public Vector3 agentPosition;
    public Vector3 agentRotation;
    public DetectedObject[] detectedObjects;
}

[System.Serializable]
public class DetectedObject
{
    public string name;
    public string tag;
    public float distance;
    public float angle;
    public Vector3 position;
    public Vector3 size;
    public bool isVisible;
    public int layer;
}
