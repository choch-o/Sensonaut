import socket
import json
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import random

class UnityControlClient:
    def __init__(self, host="127.0.0.1", port=5005):
        self.host = host
        self.port = port
        self.socket = None
        self.recording_duration = 2.0  # Should match Unity's recordingDuration
        self.last_visual_observation = None
        self.position_file = "Assets/StreamingAssets/vehicle_positions_20250808_230707.json"
        
    def connect(self):
        """Establish connection"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # Set timeout
            self.socket.connect((self.host, self.port))
            print(f"Connected to Unity server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            self.socket.close()
            self.socket = None
            print("Disconnected from Unity server")
    
    def load_vehicle_positions(self):
        """Load vehicle positions from specified file"""
        if not os.path.exists(self.position_file):
            print(f"❌ Vehicle position file does not exist: {self.position_file}")
            return []
        
        try:
            with open(self.position_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                vehicles = data.get('vehicles', [])
                print(f"✅ Successfully loaded {len(vehicles)} vehicle positions")
                return vehicles
        except Exception as e:
            print(f"❌ Failed to load vehicle position file: {e}")
            return []
    
    def create_vehicle_configurations(self, vehicle_specs):
        """Create vehicle configurations based on user-specified specifications"""
        available_positions = self.load_vehicle_positions()
        
        if len(available_positions) == 0:
            print("❌ No available vehicle positions")
            return []
        
        configurations = []
        used_positions = set()
        
        for spec in vehicle_specs:
            count = spec.get('count', 1)
            prefab_type = spec.get('prefab_type', 'black')
            position_indices = spec.get('position_indices', [])
            
            if prefab_type not in ['black', 'red', 'white']:
                prefab_type = 'black'
            
            # Select positions
            if position_indices:
                selected_positions = []
                for idx in position_indices:
                    if 0 <= idx < len(available_positions) and idx not in used_positions:
                        selected_positions.append(available_positions[idx])
                        used_positions.add(idx)
                
                # If specified positions are not enough, randomly select remaining positions
                while len(selected_positions) < count:
                    available_indices = [i for i in range(len(available_positions)) if i not in used_positions]
                    if not available_indices:
                        break
                    random_idx = random.choice(available_indices)
                    selected_positions.append(available_positions[random_idx])
                    used_positions.add(random_idx)
            else:
                # Randomly select positions
                available_indices = [i for i in range(len(available_positions)) if i not in used_positions]
                if len(available_indices) < count:
                    count = len(available_indices)
                
                random_indices = random.sample(available_indices, count)
                selected_positions = [available_positions[i] for i in random_indices]
                used_positions.update(random_indices)
            
            # Create vehicle configurations
            for position in selected_positions:
                config = {
                    "name": f"{prefab_type.capitalize()}_Car_{len(configurations)+1}",
                    "position": position.get('position', {}),
                    "rotation": position.get('rotation', {}),
                    "scale": position.get('scale', {"x": 1.0, "y": 1.0, "z": 1.0}),
                    "isActive": position.get('isActive', True),
                    "prefabType": prefab_type
                }
                configurations.append(config)
        
        return configurations
    
    def initialize_map(self, vehicle_specs=None, agent_position=None, agent_rotation=None):
        """Initialize map"""
        print("=== Initializing Map ===")
        
        # Ensure connection is established
        if not self.socket:
            if not self.connect():
                print("❌ Cannot connect to Unity server")
                return False
        
        if vehicle_specs is None:
            # Default configuration
            vehicle_specs = [
                {"count": 2, "prefab_type": "black", "position_indices": [0, 1]},
                {"count": 1, "prefab_type": "red"},
                {"count": 1, "prefab_type": "white", "position_indices": [5]}
            ]
        
        vehicles = self.create_vehicle_configurations(vehicle_specs)
        
        if len(vehicles) == 0:
            print("❌ Cannot create vehicle configurations")
            return False
        
        map_data = {
            "mapName": "Custom Map",
            "description": f"Custom scene with {len(vehicles)} vehicles",
            "vehicles": vehicles,
            "agent": {
                "position": agent_position or {"x": 0.0, "y": 1.0, "z": 0.0},
                "rotation": agent_rotation or {"x": 0.0, "y": 0.0, "z": 0.0}
            }
        }
        
        try:
            command = {
                "command": "initialize_map",
                "data": map_data
            }
            
            json_data = json.dumps(command)
            self.socket.sendall(json_data.encode("utf-8"))
            
            reply_text = self.receive_complete_response()
            print(f"Unity reply: {reply_text}")
            
            if reply_text.strip() == "ACK" or (reply_text.strip().startswith("{") and "success" in reply_text):
                print("✅ Map initialization successful")
                return True
            else:
                print("❌ Map initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ Map initialization error: {e}")
            return False
    
    def send_command(self, command):
        """Send command to Unity"""
        if not self.socket:
            if not self.connect():
                return False
        
        try:
            # Send JSON data
            json_data = json.dumps(command)
            self.socket.sendall(json_data.encode("utf-8"))
            
            # Receive reply - handle large responses
            reply_text = self.receive_complete_response()
            print(f"Unity reply: {reply_text}")
            
            # Debug: Check if reply looks like JSON
            print(f"Reply length: {len(reply_text)}")
            print(f"Reply starts with '{{': {reply_text.strip().startswith('{')}")
            print(f"Reply is not 'ACK': {reply_text != 'ACK'}")
            
            # Try to parse as JSON if it's not a simple ACK
            if reply_text != "ACK" and reply_text.strip().startswith("{"):
                try:
                    # Try to clean the JSON string first
                    cleaned_reply = self.clean_json_string(reply_text)
                    self.last_visual_observation = json.loads(cleaned_reply)
                    print("✓ Visual observation data received and parsed successfully")
                    print(f"✓ Data contains {len(self.last_visual_observation.get('detectedObjects', []))} objects")
                except json.JSONDecodeError as e:
                    print(f"✗ Failed to parse visual observation JSON: {e}")
                    print(f"✗ Raw reply: {repr(reply_text)}")
                    print(f"✗ Reply length: {len(reply_text)}")
                    
                    # Try to find the problematic character
                    if len(reply_text) > 0:
                        print(f"✗ First 100 chars: {repr(reply_text[:100])}")
                        if len(reply_text) > 100:
                            print(f"✗ Last 100 chars: {repr(reply_text[-100:])}")
                    
                    # Set empty observation to prevent crashes
                    self.last_visual_observation = {
                        "timestamp": "Error",
                        "agentPosition": {"x": 0, "y": 0, "z": 0},
                        "agentRotation": {"x": 0, "y": 0, "z": 0},
                        "detectedObjects": []
                    }
            else:
                print("✗ Reply is not JSON format (either 'ACK' or doesn't start with '{')")
            
            return True
            
        except socket.timeout:
            print("Timeout - reconnecting...")
            self.disconnect()
            return self.send_command(command)  # Retry
        except Exception as e:
            print(f"Send failed: {e}")
            self.disconnect()
            return False
    
    def receive_complete_response(self):
        """Receive complete response from Unity, handling large JSON data"""
        try:
            # Set a short timeout for the first read
            self.socket.settimeout(0.1)
            
            chunks = []
            total_data = b""
            
            while True:
                try:
                    chunk = self.socket.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total_data += chunk
                    
                    # Check if we have a complete JSON response
                    response_text = total_data.decode('utf-8', errors='ignore')
                    
                    # If it's a simple ACK, return immediately
                    if response_text.strip() == "ACK":
                        return "ACK"
                    
                    # If it starts with { and ends with }, it's likely complete
                    if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                        # Count braces to verify it's complete
                        brace_count = response_text.count('{') - response_text.count('}')
                        if brace_count == 0:
                            return response_text
                    
                except socket.timeout:
                    # No more data available
                    break
            
            # Return what we have
            return total_data.decode('utf-8', errors='ignore')
            
        except Exception as e:
            print(f"Error receiving response: {e}")
            return ""
        finally:
            # Reset timeout
            self.socket.settimeout(5.0)
    
    def clean_json_string(self, json_str):
        """Clean JSON string to fix common issues"""
        if not json_str:
            return "{}"
        
        # Remove any trailing incomplete data
        try:
            # Find the last complete JSON object
            brace_count = 0
            end_pos = -1
            
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break
            
            if end_pos >= 0:
                return json_str[:end_pos + 1]
            else:
                return "{}"
        except:
            return "{}"
    
    def get_observation(self):
        """Get observation (audio + visual) from Unity"""
        command = {"action": "get_observation"}
        success = self.send_command(command)
        
        if success:
            # Wait for audio recording to complete (visual data is now immediate)
            print(f"Waiting {self.recording_duration} seconds for audio recording...")
            time.sleep(self.recording_duration)
        
        return success
    
    def send_action(self, action_type, **kwargs):
        """Send action to Unity"""
        command = {"action": action_type}
        
        if action_type == "move_forward" and "target_position" in kwargs:
            command["target_position"] = kwargs["target_position"]
        
        return self.send_command(command)
    
    def move_to(self, x, z):
        """Move to specified position"""
        return self.send_action("move_forward", target_position=[float(x), 0.0, float(z)])
    
    def turn_left(self):
        """Turn left"""
        return self.send_action("turn_left")
    
    def turn_right(self):
        """Turn right"""
        return self.send_action("turn_right")
    
    def get_audio_file_path(self):
        """Get the path of the recorded audio file"""
        unity_data_path = os.path.expanduser("~/Library/Application Support/DefaultCompany/My project (1)/")
        audio_file_path = os.path.join(unity_data_path, "recorded_audio.wav")
        return audio_file_path
    
    def check_audio_file_exists(self):
        """Check if the audio file exists"""
        audio_path = self.get_audio_file_path()
        return os.path.exists(audio_path)
    
    def get_visual_observation(self):
        """Get the last visual observation data"""
        return self.last_visual_observation
    
    def process_visual_observation(self):
        """Process visual observation and extract useful information"""
        if not self.last_visual_observation:
            return None
        
        visual_data = self.last_visual_observation
        
        # Extract agent information
        agent_pos = visual_data.get('agentPosition', {})
        agent_rot = visual_data.get('agentRotation', {})
        
        # Process detected objects
        detected_objects = visual_data.get('detectedObjects', [])
        
        # Find closest objects
        closest_objects = []
        visible_objects = []
        
        for obj in detected_objects:
            distance = obj.get('distance', float('inf'))
            name = obj.get('name', 'Unknown')
            is_visible = obj.get('isVisible', False)
            
            closest_objects.append({
                'name': name,
                'distance': distance,
                'angle': obj.get('angle', 0)
            })
            
            if is_visible:
                visible_objects.append({
                    'name': name,
                    'distance': distance,
                    'angle': obj.get('angle', 0)
                })
        
        # Sort by distance
        closest_objects.sort(key=lambda x: x['distance'])
        visible_objects.sort(key=lambda x: x['distance'])
        
        return {
            'agent_position': agent_pos,
            'agent_rotation': agent_rot,
            'total_objects_detected': len(detected_objects),
            'closest_objects': closest_objects[:5],  # Top 5 closest
            'visible_objects': visible_objects[:5],  # Top 5 visible
            'timestamp': visual_data.get('timestamp', 'Unknown')
        }
    
    def process_observation(self):
        """Process both audio and visual observations and decide action"""
        # Process visual observation
        visual_info = self.process_visual_observation()
        
        if visual_info:
            print(f"Visual observation: {visual_info['total_objects_detected']} objects detected")
            if visual_info['closest_objects']:
                closest = visual_info['closest_objects'][0]
                print(f"Closest object: {closest['name']} at {closest['distance']:.2f} units")
        
        # Simple decision making based on visual information
        if visual_info and visual_info['closest_objects']:
            closest_obj = visual_info['closest_objects'][0]
            distance = closest_obj['distance']
            angle = closest_obj['angle']
            
            # If something is very close, turn away
            if distance < 2.0:
                if abs(angle) > 45:
                    return "turn_left", {}
                else:
                    return "turn_right", {}
            
            # If something is moderately close, move towards it
            elif distance < 5.0:
                # Calculate target position towards the object
                agent_pos = visual_info['agent_position']
                target_x = agent_pos.get('x', 0) + (distance - 1.0) * 0.5
                target_z = agent_pos.get('z', 0)
                return "move_forward", {"target_position": [target_x, 0.0, target_z]}
        
        # Default behavior: move forward
        return "move_forward", {"target_position": [5.0, 0.0, 0.0]}

    def print_visual_observation_details(self):
        """Print detailed visual observation data"""
        if not self.last_visual_observation:
            print("❌ No visual observation data available")
            return
        
        print("\n" + "="*60)
        print("📊 DETAILED VISUAL OBSERVATION DATA")
        print("="*60)
        
        # Print raw JSON
        print("🔍 RAW JSON DATA:")
        print(json.dumps(self.last_visual_observation, indent=2))
        
        # Print structured information
        print("\n📋 STRUCTURED INFORMATION:")
        print(f"Timestamp: {self.last_visual_observation.get('timestamp', 'Unknown')}")
        
        agent_pos = self.last_visual_observation.get('agentPosition', {})
        agent_rot = self.last_visual_observation.get('agentRotation', {})
        print(f"Agent Position: x={agent_pos.get('x', 0):.2f}, y={agent_pos.get('y', 0):.2f}, z={agent_pos.get('z', 0):.2f}")
        print(f"Agent Rotation: x={agent_rot.get('x', 0):.2f}, y={agent_rot.get('y', 0):.2f}, z={agent_rot.get('z', 0):.2f}")
        
        detected_objects = self.last_visual_observation.get('detectedObjects', [])
        print(f"\n🎯 DETECTED OBJECTS ({len(detected_objects)} total):")
        
        if detected_objects:
            print(f"{'Name':<20} {'Tag':<15} {'Distance':<10} {'Angle':<8} {'Visible':<8} {'Layer':<6}")
            print("-" * 80)
            
            for i, obj in enumerate(detected_objects):
                name = obj.get('name', 'Unknown')[:19]
                tag = obj.get('tag', 'Untagged')[:14]
                distance = obj.get('distance', 0)
                angle = obj.get('angle', 0)
                is_visible = "Yes" if obj.get('isVisible', False) else "No"
                layer = obj.get('layer', 0)
                
                print(f"{name:<20} {tag:<15} {distance:<10.2f} {angle:<8.1f} {is_visible:<8} {layer:<6}")
                
                # Print position and size for first few objects
                if i < 3:
                    pos = obj.get('position', {})
                    size = obj.get('size', {})
                    print(f"    Position: ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})")
                    print(f"    Size: ({size.get('x', 0):.2f}, {size.get('y', 0):.2f}, {size.get('z', 0):.2f})")
        else:
            print("❌ No objects detected - possible issues:")
            print("   • No objects within observation radius")
            print("   • VisualCapture component not properly configured")
            print("   • LayerMask excludes all objects")
            print("   • Objects don't have Collider components")
            print("   • Agent position is not in a scene with objects")
        
        print("="*60)

    def is_position_walkable(self, x, z):
        """Check if a world position is walkable using movement map data"""
        if not hasattr(self, 'movement_map_data') or not self.movement_map_data:
            return True  # Assume walkable if no map data
        
        try:
            # Get map parameters
            grid_origin = self.movement_map_data.get('gridOrigin', {})
            grid_size = self.movement_map_data.get('gridSize', 1.0)
            grid_dimensions = self.movement_map_data.get('gridDimensions', {})
            walkable_grid = self.movement_map_data.get('walkableGrid', [])
            
            # Convert world position to grid coordinates
            local_x = x - grid_origin.get('x', 0)
            local_z = z - grid_origin.get('z', 0)
            grid_x = int(local_x / grid_size)
            grid_z = int(local_z / grid_size)
            
            # Check bounds
            width = grid_dimensions.get('x', 0)
            height = grid_dimensions.get('y', 0)
            if grid_x < 0 or grid_x >= width or grid_z < 0 or grid_z >= height:
                return False
            
            # Check if position is walkable
            index = grid_x + grid_z * width
            if 0 <= index < len(walkable_grid):
                return walkable_grid[index]
            
            return False
        except Exception as e:
            print(f"Error checking walkable position: {e}")
            return True

    def find_safe_directions(self, x, z):
        """Find safe directions to move from current position"""
        if not hasattr(self, 'movement_map_data') or not self.movement_map_data:
            return [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Default directions
        
        safe_directions = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Right, Left, Forward, Back
        
        for dx, dz in directions:
            test_x = x + dx * 2.0  # Test 2 units away
            test_z = z + dz * 2.0
            if self.is_position_walkable(test_x, test_z):
                safe_directions.append((dx, dz))
        
        return safe_directions

    def choose_best_direction(self, safe_directions, obstacle_angle):
        """Choose the best direction to avoid an obstacle"""
        if not safe_directions:
            return "right"  # Default
        
        # Convert obstacle angle to direction preference
        # Positive angle = obstacle on right, negative = obstacle on left
        if obstacle_angle > 0:  # Obstacle on right, prefer left
            preferred_directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]  # Left, Back, Forward, Right
        else:  # Obstacle on left, prefer right
            preferred_directions = [(1, 0), (0, 1), (0, -1), (-1, 0)]  # Right, Forward, Back, Left
        
        # Find the first safe direction in preferred order
        for direction in preferred_directions:
            if direction in safe_directions:
                if direction == (-1, 0):  # Left
                    return "left"
                elif direction == (1, 0):  # Right
                    return "right"
                else:  # Forward or Back
                    return "right"  # Default to right turn
        
        return "right"  # Default

    def find_nearest_walkable_position(self, target_x, target_z, max_search_radius=5.0):
        """Find the nearest walkable position to a target"""
        if not hasattr(self, 'movement_map_data') or not self.movement_map_data:
            return (target_x, target_z)  # Return original if no map data
        
        # Search in expanding circles
        search_radius = 1.0
        while search_radius <= max_search_radius:
            # Test positions in a circle around target
            import math
            for angle in range(0, 360, 30):  # Test every 30 degrees
                radians = math.radians(angle)
                test_x = target_x + search_radius * math.cos(radians)
                test_z = target_z + search_radius * math.sin(radians)
                
                if self.is_position_walkable(test_x, test_z):
                    return (test_x, test_z)
            
            search_radius += 1.0
        
        return None  # No walkable position found

    def run_agent_loop(self, num_iterations=1):
        """Run the observation-action loop for the agent"""
        print(f"Starting agent control loop ({num_iterations} iteration{'s' if num_iterations > 1 else ''})...")
    
        
        for i in range(num_iterations):
            print(f"\n=== Iteration {i+1}/{num_iterations} ===")
            
            # Step 1: Get observation (audio + visual) from Unity
            print("1. Getting observation from Unity...")
            if not self.get_observation():
                print("Failed to get observation, skipping iteration")
                continue
            
            # Step 2: Check if audio file exists (optional)
            if self.check_audio_file_exists():
                print(f"   Audio file found: {self.get_audio_file_path()}")
            else:
                print("   Audio file not found yet")
            
            # Step 3: Check visual data
            print("2. Checking visual data...")
            visual_data = self.get_visual_observation()
            if visual_data:
                objects_count = len(visual_data.get('detectedObjects', []))
                print(f"   Visual: {objects_count} objects detected")
                
                # Print detailed visual observation data
                self.print_visual_observation_details()
            else:
                print("   Visual: No data received")
            
            # Step 4: Process observation and decide action
            print("3. Processing observation...")
            action_type, action_params = self.process_observation()
            print(f"   Decided action: {action_type}")
            
            # Step 5: Send action to Unity
            print("4. Sending action to Unity...")
            if action_type == "move_forward":
                self.move_to(action_params["target_position"][0], action_params["target_position"][2])
            elif action_type == "turn_left":
                self.turn_left()
            elif action_type == "turn_right":
                self.turn_right()
            
            # Wait before next iteration (only if more iterations to come)
            if i < num_iterations - 1:
                print("5. Waiting for next iteration...")
                time.sleep(3)
        
        print(f"\nAgent control loop completed ({num_iterations} iteration{'s' if num_iterations > 1 else ''})!")

# Example of correct control flow
if __name__ == "__main__":
    client = UnityControlClient()
    
    try:
        print("Unity Agent Control System")
        print("=" * 50)

        # Step 1: Initialize map before running agent loop
        print("\n=== Step 1: Map Initialization ===")
        map_initialized = client.initialize_map()
        
        if not map_initialized:
            print("⚠️ Map initialization failed, but continuing with agent loop...")
        else:
            print("✅ Map initialization completed successfully")
        
        # # Step 2: Run agent loop
        # print("\n=== Step 2: Running Agent Loop ===")
        # num_iterations = 1
        # client.run_agent_loop(num_iterations)
        
    finally:
        client.disconnect()
