using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovementMap : MonoBehaviour
{
    [Header("Movement Map Settings")]
    public float gridSize = 1.0f; // Size of each grid cell
    public float agentRadius = 0.5f; // Radius of the agent for collision checking
    public LayerMask obstacleLayerMask = -1; // Layers to check for obstacles
    public bool showDebugGrid = true; // Show grid in scene view
    
    [Header("Map Bounds")]
    public Vector2 mapCenter = Vector2.zero; // Only XZ coordinates
    public Vector2 mapSize = new Vector2(50f, 50f); // Only XZ size (width, length)
    
    [Header("Runtime Data")]
    public bool[,] walkableGrid; // 2D grid of walkable positions
    public Vector2Int gridDimensions;
    public Vector2 gridOrigin; // Only XZ coordinates
    
    private bool isInitialized = false;
    
    void Start()
    {
        InitializeMovementMap();
    }
    
    /// <summary>
    /// Initialize the movement map by scanning the scene
    /// </summary>
    public void InitializeMovementMap()
    {
        Debug.Log("Initializing movement map...");
        
        // Calculate grid dimensions (X and Z only)
        gridDimensions = new Vector2Int(
            Mathf.CeilToInt(mapSize.x / gridSize),
            Mathf.CeilToInt(mapSize.y / gridSize) // mapSize.y represents Z
        );
        
        // Calculate grid origin (bottom-left corner, XZ only)
        gridOrigin = mapCenter - new Vector2(mapSize.x * 0.5f, mapSize.y * 0.5f);
        
        // Initialize the grid
        walkableGrid = new bool[gridDimensions.x, gridDimensions.y];
        
        // Scan the grid
        ScanGrid();
        
        isInitialized = true;
        Debug.Log($"Movement map initialized: {gridDimensions.x}x{gridDimensions.y} grid, {CountWalkableCells()} walkable cells");
    }
    
    /// <summary>
    /// Scan the grid to determine walkable areas
    /// </summary>
    private void ScanGrid()
    {
        int walkableCount = 0;
        
        for (int x = 0; x < gridDimensions.x; x++)
        {
            for (int z = 0; z < gridDimensions.y; z++)
            {
                Vector2 worldPos = GridToWorldPosition(x, z);
                walkableGrid[x, z] = IsPositionWalkable(worldPos);
                
                if (walkableGrid[x, z])
                {
                    walkableCount++;
                }
            }
        }
        
        Debug.Log($"Grid scan complete: {walkableCount}/{gridDimensions.x * gridDimensions.y} cells are walkable");
    }
    
    /// <summary>
    /// Check if a world position is walkable (XZ only)
    /// </summary>
    private bool IsPositionWalkable(Vector2 worldPos)
    {
        // Convert to Vector3 for physics check (Y = 0 for ground level)
        Vector3 worldPos3D = new Vector3(worldPos.x, 0, worldPos.y);
        
        // Use simple sphere overlap for 2D navigation
        Collider[] overlappingColliders = Physics.OverlapSphere(worldPos3D, agentRadius, obstacleLayerMask);
        
        // Position is walkable if no obstacles found
        return overlappingColliders.Length == 0;
    }
    
    /// <summary>
    /// Convert grid coordinates to world position (XZ only)
    /// </summary>
    public Vector2 GridToWorldPosition(int gridX, int gridZ)
    {
        return gridOrigin + new Vector2(
            gridX * gridSize + gridSize * 0.5f, 
            gridZ * gridSize + gridSize * 0.5f
        );
    }
    
    /// <summary>
    /// Convert world position to grid coordinates (XZ only)
    /// </summary>
    public Vector2Int WorldToGridPosition(Vector2 worldPos)
    {
        Vector2 localPos = worldPos - gridOrigin;
        return new Vector2Int(
            Mathf.FloorToInt(localPos.x / gridSize),
            Mathf.FloorToInt(localPos.y / gridSize)
        );
    }
    
    /// <summary>
    /// Check if a world position is walkable (XZ only)
    /// </summary>
    public bool IsWorldPositionWalkable(Vector2 worldPos)
    {
        if (!isInitialized)
        {
            Debug.LogWarning("Movement map not initialized!");
            return true;
        }
        
        Vector2Int gridPos = WorldToGridPosition(worldPos);
        
        // Check if grid position is within bounds
        if (gridPos.x < 0 || gridPos.x >= gridDimensions.x || 
            gridPos.y < 0 || gridPos.y >= gridDimensions.y)
        {
            return false;
        }
        
        return walkableGrid[gridPos.x, gridPos.y];
    }
    
    /// <summary>
    /// Find the nearest walkable position to a target position (XZ only)
    /// </summary>
    public Vector2 FindNearestWalkablePosition(Vector2 targetPosition, float maxSearchRadius = 10f)
    {
        if (!isInitialized)
        {
            Debug.LogWarning("Movement map not initialized!");
            return targetPosition;
        }
        
        // Check if target position is already walkable
        if (IsWorldPositionWalkable(targetPosition))
        {
            return targetPosition;
        }
        
        // Search in expanding circles
        float searchRadius = gridSize;
        int maxIterations = Mathf.CeilToInt(maxSearchRadius / gridSize);
        
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Search in a circle around the target
            int pointsInCircle = Mathf.Max(8, Mathf.CeilToInt(2 * Mathf.PI * searchRadius / gridSize));
            
            for (int i = 0; i < pointsInCircle; i++)
            {
                float angle = (2 * Mathf.PI * i) / pointsInCircle;
                Vector2 offset = new Vector2(
                    Mathf.Cos(angle) * searchRadius,
                    Mathf.Sin(angle) * searchRadius
                );
                
                Vector2 testPosition = targetPosition + offset;
                
                if (IsWorldPositionWalkable(testPosition))
                {
                    Debug.Log($"Found walkable position at {testPosition} (distance: {searchRadius:F2})");
                    return testPosition;
                }
            }
            
            searchRadius += gridSize;
        }
        
        Debug.LogWarning($"No walkable position found within {maxSearchRadius} units of {targetPosition}");
        return targetPosition;
    }
    
    /// <summary>
    /// Get a list of all walkable positions (XZ only)
    /// </summary>
    public List<Vector2> GetAllWalkablePositions()
    {
        List<Vector2> walkablePositions = new List<Vector2>();
        
        if (!isInitialized)
        {
            Debug.LogWarning("Movement map not initialized!");
            return walkablePositions;
        }
        
        for (int x = 0; x < gridDimensions.x; x++)
        {
            for (int z = 0; z < gridDimensions.y; z++)
            {
                if (walkableGrid[x, z])
                {
                    walkablePositions.Add(GridToWorldPosition(x, z));
                }
            }
        }
        
        return walkablePositions;
    }
    
    /// <summary>
    /// Count the number of walkable cells
    /// </summary>
    public int CountWalkableCells()
    {
        if (!isInitialized) return 0;
        
        int count = 0;
        for (int x = 0; x < gridDimensions.x; x++)
        {
            for (int z = 0; z < gridDimensions.y; z++)
            {
                if (walkableGrid[x, z])
                {
                    count++;
                }
            }
        }
        return count;
    }
    
    /// <summary>
    /// Get a random walkable position (XZ only)
    /// </summary>
    public Vector2 GetRandomWalkablePosition()
    {
        if (!isInitialized)
        {
            Debug.LogWarning("Movement map not initialized!");
            return Vector2.zero;
        }
        
        List<Vector2> walkablePositions = GetAllWalkablePositions();
        
        if (walkablePositions.Count == 0)
        {
            Debug.LogWarning("No walkable positions found!");
            return Vector2.zero;
        }
        
        return walkablePositions[Random.Range(0, walkablePositions.Count)];
    }

    
    // Debug visualization in scene view
    void OnDrawGizmos()
    {
        if (!showDebugGrid || !isInitialized) return;
        
        // Draw grid bounds (XZ plane only)
        Gizmos.color = Color.yellow;
        Vector3 mapCenter3D = new Vector3(mapCenter.x, 0, mapCenter.y);
        Vector3 mapSize3D = new Vector3(mapSize.x, 0.1f, mapSize.y);
        Gizmos.DrawWireCube(mapCenter3D, mapSize3D);
        
        // Draw walkable cells
        Gizmos.color = Color.green;
        for (int x = 0; x < gridDimensions.x; x++)
        {
            for (int z = 0; z < gridDimensions.y; z++)
            {
                if (walkableGrid[x, z])
                {
                    Vector2 pos2D = GridToWorldPosition(x, z);
                    Vector3 pos3D = new Vector3(pos2D.x, 0.05f, pos2D.y);
                    Gizmos.DrawWireCube(pos3D, new Vector3(gridSize * 0.8f, 0.1f, gridSize * 0.8f));
                }
            }
        }
        
        // Draw agent size at origin for reference (XZ plane only)
        Gizmos.color = Color.blue;
        Vector3 agentPos = mapCenter3D;
        Gizmos.DrawWireSphere(agentPos, agentRadius);
    }
}


