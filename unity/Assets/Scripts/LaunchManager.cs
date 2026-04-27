using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LaunchManager : MonoBehaviour
{
    public FirstPersonControl firstPersonControl;
    public MonoBehaviour agentController; 
    public SocketServer socketServer;

    void Start()
    {
        string[] args = System.Environment.GetCommandLineArgs();
        string mode = "python"; // Default to python control

        foreach (string arg in args)
        {
            if (arg.StartsWith("--control="))
            {
                mode = arg.Substring("--control=".Length);
                break;
            }
        }

        Debug.Log("Control mode selected: " + mode);

        if (mode == "python")
        {
            if (firstPersonControl != null) firstPersonControl.enabled = false;
            if (agentController != null) agentController.enabled = true;
            if (socketServer != null) socketServer.enabled = true;
        }
        else
        {
            if (firstPersonControl != null) firstPersonControl.enabled = true;
            if (agentController != null) agentController.enabled = false;
            if (socketServer != null) socketServer.enabled = false;
        }
    }
}
