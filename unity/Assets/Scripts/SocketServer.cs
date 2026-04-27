using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;

[System.Serializable]
public class CommandData
{
    public string command;
    public string data;
}

public class SocketServer : MonoBehaviour
{
    private TcpListener listener;
    private Thread serverThread;
    private bool isRunning = false;
    private string pendingCommand = null;
    private string pendingResponse = null;
    private bool isPreparingResponse = false;
    
    // Map initialization related
    private bool isMapInitPending = false;
    private string pendingMapInitData = null;

    // Persistent receive buffer for incoming data
    private StringBuilder receiveBuffer = new StringBuilder();
    const int timeoutMs = 100;

    public AgentControl agent;
    public MapInitializer mapInitializer;

    void Start()
    {
        // Auto-find MapInitializer component
        if (mapInitializer == null)
        {
            mapInitializer = FindObjectOfType<MapInitializer>();
            if (mapInitializer != null)
            {
                Debug.Log("Found MapInitializer component automatically");
            }
            else
            {
                Debug.LogWarning("MapInitializer component not found. Please add a MapInitializer component to the scene or assign it manually to SocketServer.");
            }
        }
        
        serverThread = new Thread(ListenForClients);
        serverThread.IsBackground = true;
        serverThread.Start();
    }

    void Update()
    {
        // Handle commands in main thread
        if (!string.IsNullOrEmpty(pendingCommand))
        {
            Debug.Log($"Processing command: {pendingCommand}");

            if (agent != null)
            {
                agent.SetAction(pendingCommand);

                // If this was a get_observation command, prepare the response immediately
                if (pendingCommand.Contains("get_observation"))
                {
                    Debug.Log("Processing observation command immediately...");
                    string visualData = agent.GetVisualObservationJson();
                    pendingResponse = visualData;
                    isPreparingResponse = false;
                    Debug.Log($"Visual observation data prepared immediately, length: {visualData.Length}");
                    Debug.Log($"pendingResponse set to: {pendingResponse != null}");
                }
                else
                {
                    Debug.Log("Processing command in main thread...");
                    pendingResponse = "{\"status\": \"success\", \"message\": \"Commit, move, or turn successfully\"}";
                    isPreparingResponse = false;
                }
            }
            else
            {
                Debug.LogWarning("Agent not assigned to SocketServer");
            }
            pendingCommand = null;
        }
        
        // Handle map initialization in main thread
        if (isMapInitPending && !string.IsNullOrEmpty(pendingMapInitData))
        {
            Debug.Log("Processing map initialization in main thread...");
            
            if (mapInitializer != null)
            {
                try
                {
                    // Parse the map initialization data
                    var mapData = JsonUtility.FromJson<MapInitData>(pendingMapInitData);
                    
                    // Execute map initialization
                    mapInitializer.InitializeMap(mapData);
                    
                    pendingResponse = "{\"status\": \"success\", \"message\": \"Map initialized successfully\"}";
                    Debug.Log("Map initialized successfully");
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"Failed to initialize map: {e.Message}");
                    pendingResponse = "{\"status\": \"error\", \"message\": \"" + e.Message + "\"}";
                }
            }
            else
            {
                Debug.LogError("MapInitializer is null");
                pendingResponse = "{\"status\": \"error\", \"message\": \"MapInitializer not available\"}";
            }
            
            isMapInitPending = false;
            pendingMapInitData = null;
        }
    }

    private List<string> ExtractJsonObjects(StringBuilder buffer)
    {
        var result = new List<string>();
        int depth = 0;
        bool inString = false;
        bool escape = false;
        int start = -1;

        for (int i = 0; i < buffer.Length; i++)
        {
            char c = buffer[i];

            if (inString)
            {
                if (escape)
                {
                    escape = false;
                }
                else if (c == '\\')
                {
                    escape = true;
                }
                else if (c == '"')
                {
                    inString = false;
                }
                continue;
            }

            if (c == '"')
            {
                inString = true;
                continue;
            }

            if (c == '{')
            {
                if (depth == 0) start = i; // start of a new JSON object
                depth++;
            }
            else if (c == '}')
            {
                depth--;
                if (depth == 0 && start >= 0)
                {
                    int len = i - start + 1;
                    string obj = buffer.ToString(start, len);
                    result.Add(obj);

                    // Remove consumed part and reset scanning
                    buffer.Remove(0, i + 1);
                    i = -1;
                    start = -1;
                }
            }
        }

        return result;
    }

    void ListenForClients()
    {
        try
        {
            listener = new TcpListener(IPAddress.Any, 5005);
            listener.Start();
            isRunning = true;

            Debug.Log("Server started on port 5005"); 

            while (isRunning)
            {
                try
                {
                    TcpClient client = listener.AcceptTcpClient();
                    Debug.Log("Client connected");
                    
                    // Create new thread for each client
                    Thread clientThread = new Thread(() => HandleClient(client));
                    clientThread.IsBackground = true;
                    clientThread.Start();
                }
                catch (System.Exception e)
                {
                    if (isRunning)
                    {
                        Debug.LogError($"Error accepting client: {e.Message}");
                    }
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Server error: {e.Message}");
        }
    }

    void HandleClient(TcpClient client)
    {
        try
        {
            using (NetworkStream stream = client.GetStream())
            {
                byte[] buffer = new byte[4096]; // Increased buffer size
                while (isRunning)
                {
                    try
                    {
                        // Check if client is still connected
                        if (!IsClientConnected(client))
                        {
                            Debug.Log("Client disconnected");
                            break;
                        }

                        // Set timeout for read operation
                        stream.ReadTimeout = 10000; // 10 seconds
                        
                        int count = stream.Read(buffer, 0, buffer.Length);
                        if (count == 0)
                        {
                            Debug.Log("Client disconnected (no data received)");
                            break;
                        }

                        // Append this chunk to the persistent buffer
                        string chunk = Encoding.UTF8.GetString(buffer, 0, count);
                        receiveBuffer.Append(chunk);

                        // Extract all complete top-level JSON objects from the buffer
                        var messages = ExtractJsonObjects(receiveBuffer);
                        foreach (var cmd in messages)
                        {
                            if (string.IsNullOrWhiteSpace(cmd))
                                continue;

                            Debug.Log($"Received command: {cmd}");

                            string response = "ACK";

                            if (cmd.Contains("get_observation"))
                            {
                                Debug.Log("Scheduling observation capture for main thread...");
                                pendingResponse = null;
                                isPreparingResponse = true;
                                pendingCommand = cmd;

                                int waitedMs = 0;
                                while (isPreparingResponse && waitedMs < timeoutMs)
                                {
                                    Thread.Sleep(1);
                                    waitedMs += 1;
                                }

                                if (!isPreparingResponse && !string.IsNullOrEmpty(pendingResponse))
                                {
                                    response = pendingResponse;
                                }
                                else
                                {
                                    Debug.LogWarning("Observation capture timed out or returned empty");
                                    response = "{\"error\": \"Observation capture timeout\"}";
                                }
                                pendingResponse = null;
                            }
                            else if (cmd.Contains("initialize_map"))
                            {
                                Debug.Log("Scheduling map initialization for main thread...");
                                if (mapInitializer != null)
                                {
                                    try
                                    {
                                        var commandData = JsonUtility.FromJson<CommandData>(cmd);
                                        Debug.Log($"Parsed command: {commandData.command}, data: {commandData.data}");

                                        if (commandData != null && commandData.data != null)
                                        {
                                            pendingMapInitData = commandData.data;
                                            isMapInitPending = true;

                                            while (isMapInitPending)
                                            {
                                                Thread.Sleep(10);
                                            }

                                            if (pendingResponse != null)
                                            {
                                                response = pendingResponse;
                                                Debug.Log("Map initialization completed");
                                            }
                                            else
                                            {
                                                response = "{\"status\": \"error\", \"message\": \"Map initialization failed\"}";
                                                Debug.LogError("Map initialization failed - no response from main thread");
                                            }
                                        }
                                        else
                                        {
                                            Debug.LogError("Invalid command structure or missing data field");
                                            response = "{\"status\": \"error\", \"message\": \"Invalid command structure\"}";
                                        }
                                    }
                                    catch (System.Exception e)
                                    {
                                        Debug.LogError($"Failed to parse map data: {e.Message}");
                                        response = "{\"status\": \"error\", \"message\": \"" + e.Message + "\"}";
                                    }
                                }
                                else
                                {
                                    Debug.LogError("MapInitializer is null");
                                    response = "{\"status\": \"error\", \"message\": \"MapInitializer not available\"}";
                                }
                            }
                            else
                            {
                                pendingResponse = null;
                                isPreparingResponse = true;
                                pendingCommand = cmd;

                                int waitedMs = 0;
                                while (isPreparingResponse && waitedMs < timeoutMs)
                                {
                                    Thread.Sleep(1);
                                    waitedMs += 1;
                                }

                                if (!isPreparingResponse && !string.IsNullOrEmpty(pendingResponse))
                                {
                                    response = pendingResponse;
                                }
                                else
                                {
                                    Debug.LogWarning("Commit, turn, or move timed out or returned empty");
                                    response = "{\"error\": \"Commit, turn, or move timeout\"}";
                                }
                                pendingResponse = null;
                            }

                            if (SendResponseSafely(stream, response))
                            {
                                Debug.Log($"Response sent: {response}");
                                Debug.Log("Response sent successfully");
                            }
                            else
                            {
                                Debug.LogWarning("Failed to send response, client may have disconnected");
                                break;
                            }
                        }
                    }
                    catch (System.IO.IOException ioEx)
                    {
                        Debug.Log($"IO Error in client communication: {ioEx.Message}");
                        break;
                    }
                    catch (System.Exception ex)
                    {
                        Debug.LogError($"Error in client communication: {ex.Message}");
                        break;
                    }
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Client handling error: {e.Message}");
        }
        finally
        {
            client.Close();
            Debug.Log("Client connection closed");
        }
    }

    private bool IsClientConnected(TcpClient client)
    {
        try
        {
            if (client == null || !client.Connected)
                return false;

            // Check if the socket is still connected
            if (client.Client.Poll(0, SelectMode.SelectRead))
            {
                byte[] buff = new byte[1];
                if (client.Client.Receive(buff, SocketFlags.Peek) == 0)
                    return false;
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    private bool SendResponseSafely(NetworkStream stream, string response)
    {
        try
        {
            byte[] responseBytes = Encoding.UTF8.GetBytes(response);
            stream.Write(responseBytes, 0, responseBytes.Length);
            stream.Flush();
            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to send response: {e.Message}");
            return false;
        }
    }

    void OnApplicationQuit()
    {
        Debug.Log("Application quitting, stopping server...");
        isRunning = false;
        
        if (listener != null)
        {
            listener.Stop();
        }
        
        if (serverThread != null && serverThread.IsAlive)
        {
            serverThread.Join(1000); // Wait up to 1 second for thread to finish
        }
    }

    void OnDestroy()
    {
        Debug.Log("SocketServer destroyed, cleaning up...");
        isRunning = false;
        
        if (listener != null)
        {
            listener.Stop();
        }
        
        if (serverThread != null && serverThread.IsAlive)
        {
            serverThread.Join(1000); // Wait up to 1 second for thread to finish
        }
    }
}
